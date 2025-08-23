import google.generativeai as genai
import json
import os
import time
from datetime import datetime
from google.api_core import exceptions
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from contextlib import asynccontextmanager

# --- Configuration ---
# It's recommended to set the API key as an environment variable for security.
# Example: export GOOGLE_API_KEY="your_api_key_here"
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAzBsWgFr2YItzRawoQb7fLX8LLomy7Ruc") 

genai.configure(api_key=API_KEY)

# --- Model and Safety Configuration ---
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
model = genai.GenerativeModel(
    'gemini-2.5-flash', 
    safety_settings=safety_settings
)

# --- Pydantic Models for API Input and Output ---
class PostInput(BaseModel):
    timestamp: str
    lat: float
    lon: float
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-08-23 14:30:00",
                "lat": 13.0827,
                "lon": 80.2707,
                "message": "URGENT!! Bridge near Central Station just collapsed due to the floods. My cousin sent a video. Everyone avoid that area!!"
            }
        }

class AnalysisResponse(BaseModel):
    final_action: str
    verdict: str
    summary: str
    confidence_score: int
    initial_analysis: Dict
    spatiotemporal_analysis: Dict

# This will hold the loaded DataFrame in memory
verified_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load verified reports into memory when the API server starts."""
    global verified_df
    verified_reports_filepath = 'verified.csv'
    print(f"Loading verified reports from '{verified_reports_filepath}'...")
    verified_df = load_verified_reports(verified_reports_filepath)
    if verified_df is None:
        print("CRITICAL ERROR: 'verified.csv' could not be loaded. The Spatiotemporal Agent will have no context.")
        # Create an empty DataFrame so the app doesn't crash on startup
        verified_df = pd.DataFrame(columns=['timestamp', 'lat', 'lon', 'message'])
        verified_df['timestamp'] = pd.to_datetime(verified_df['timestamp'])
    else:
        print("Context from verified reports loaded successfully.")
    yield
    # Cleanup logic can go here if needed (e.g., closing database connections)
    print("Shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Disaster Response Information Verification API",
    description="An API that uses a multi-agent system to analyze and verify social media posts during a crisis.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Core Functions (Agents and Helpers) ---
def get_llm_response(prompt: str) -> str | None:
    """Sends a prompt to the Gemini API and returns the text response, with exponential backoff."""
    max_retries = 5
    delay = 2
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response.parts:
                cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
                return cleaned_text
            return None
        except exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                print(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("Max retries reached. Failed to get response from Gemini API.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred with the Gemini API: {e}")
            return None
    return None

def initial_analysis_agent(message: str) -> dict:
    """Analyzes a message for urgency and misinformation indicators in a single call."""
    prompt = f"""
    You are a multi-faceted analysis agent for a disaster response team.
    Analyze the following social media message for BOTH urgency and misinformation indicators.
    The message is: "{message}"
    Respond in a single JSON format with the following keys:
    - "priority": Classify urgency as "Critical", "High", "Medium", or "Low".
    - "priority_reason": Briefly explain your priority reasoning.
    - "risk_level": Classify misinformation risk as "Very High", "High", "Medium", or "Low".
    - "flags": A list of specific misinformation red flags you identified.
    """
    response_text = get_llm_response(prompt)
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {
            "priority": "Unknown", "priority_reason": "Failed to parse LLM response.",
            "risk_level": "Unknown", "flags": ["Failed to parse LLM response."]
        }

def spatiotemporal_agent(message: str, timestamp: str, location: str, other_reports: str) -> dict:
    """Checks for consistency with other verified reports."""
    prompt = f"""
    You are a Spatiotemporal Consistency Agent. Your job is to determine if a specific report makes sense given other information.
    The primary report is:
    - Time: {timestamp}
    - Location: {location}
    - Text: "{message}"
    Here is a summary of other VERIFIED reports from around the same time and location:
    - {other_reports}
    Does the primary report contradict or align with the other reports?
    Respond in JSON format with two keys:
    1. "consistency": "Consistent", "Inconsistent", or "Unverifiable".
    2. "reason": Explain your reasoning.
    """
    response_text = get_llm_response(prompt)
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {"consistency": "Unknown", "reason": "Failed to parse LLM response."}

def synthesis_agent(initial_report: dict, spatiotemporal_report: dict, original_message: str) -> dict:
    """Synthesizes all reports into a final verdict and classifies the need."""
    prompt = f"""
    You are the Lead Analyst for a Crisis Information Verification Unit. You have received reports from your specialized agents about a social media message. Your job is to synthesize these reports, provide a final verdict, and classify the message's need.

    Original Message: "{original_message}"

    Agent Reports:
    - Initial Analysis Report (Urgency & Content Risk): {json.dumps(initial_report)}
    - Spatiotemporal Agent (Consistency Check): {json.dumps(spatiotemporal_report)}

    Based on the available reports, provide a final analysis.
    Respond in JSON format with four keys:
    1. "verdict": "Verified Official", "Likely True", "Unverified - High Priority", "Unverified - Low Priority", "Likely False", or "Misinformation/Harmful".
    2. "summary": A one-sentence summary of your conclusion, based ONLY on the provided reports.
    3. "confidence_score": A score from 0 to 100 in your verdict.
    4. "actionable_category": If the message is a credible request for help, classify it as "Needs Rescue", "Needs Food/Water", or "Needs Medical". If the message is not a request for help, is likely false, or is just informational, classify it as "Safe".
    """
    response_text = get_llm_response(prompt)
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {"verdict": "Error", "summary": "Failed to synthesize final report.", "confidence_score": 0, "actionable_category": "Safe"}

def get_relevant_context(input_post: dict, verified_df: pd.DataFrame) -> str:
    """Filters the verified reports DataFrame to find reports from the same day."""
    try:
        input_date = pd.to_datetime(input_post['timestamp']).date()
        relevant_reports_df = verified_df[verified_df['timestamp'].dt.date == input_date]
        if relevant_reports_df.empty:
            return "No verified reports available for this day."
        return "\n- ".join(relevant_reports_df['message'].tolist())
    except Exception as e:
        print(f"Could not determine relevant context: {e}")
        return "Error processing verified reports."

def load_verified_reports(filepath: str) -> pd.DataFrame | None:
    """Loads verified reports from CSV and returns a DataFrame."""
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        print(f"Error: Verified reports file not found at '{filepath}'.")
        return None
    except Exception as e:
        print(f"An error occurred loading the verified reports: {e}.")
        return None

def get_final_action(final_verdict: dict) -> str:
    """Returns the specific action category if the verdict is credible, otherwise returns 'Safe'."""
    verdict = final_verdict.get('verdict', 'Error')
    # Only return a specific need if the information is considered trustworthy
    if verdict in ["Verified Official", "Likely True", "Unverified - High Priority"]:
        return final_verdict.get("actionable_category", "Safe")
    else:
        return "Safe"

# --- Orchestrator ---
def run_analysis_pipeline(post_data: dict, verified_context_df: pd.DataFrame) -> dict:
    """Orchestrates the full agentic workflow for a single post."""
    message = post_data['message']
    timestamp = post_data['timestamp']
    location = f"Lat: {post_data['lat']}, Lon: {post_data['lon']}"
    
    print("\n--- Analyzing New Message ---")
    initial_report = initial_analysis_agent(message)
    print(f"Step 1: Initial Analysis -> Priority: {initial_report.get('priority')}, Risk: {initial_report.get('risk_level')}")
    
    relevant_context = get_relevant_context(post_data, verified_context_df)
    spatiotemporal_report = spatiotemporal_agent(message, timestamp, location, relevant_context)
    print(f"Step 2: Spatiotemporal -> Consistency: {spatiotemporal_report.get('consistency')}")
    
    final_verdict = synthesis_agent(initial_report, spatiotemporal_report, message)
    print(f"Step 3: Synthesis -> Final Verdict: {final_verdict.get('verdict')}")
    
    return {
        "initial_report": initial_report,
        "spatiotemporal_report": spatiotemporal_report,
        "final_verdict": final_verdict
    }

# --- API Endpoint ---
@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_message(post: PostInput):
    """
    Analyzes a social media post for urgency, misinformation, and spatiotemporal consistency.
    """
    if verified_df is None:
        raise HTTPException(status_code=503, detail="Server is not ready: Verified reports context is not loaded.")
    try:
        # post.model_dump() converts the Pydantic model to a dictionary
        analysis_results = run_analysis_pipeline(post.model_dump(), verified_df)
        
        final_verdict_json = analysis_results['final_verdict']
        final_action = get_final_action(final_verdict_json)
        
        return AnalysisResponse(
            final_action=final_action,
            verdict=final_verdict_json.get('verdict', 'Error'),
            summary=final_verdict_json.get('summary', 'N/A'),
            confidence_score=final_verdict_json.get('confidence_score', 0),
            initial_analysis=analysis_results['initial_report'],
            spatiotemporal_analysis=analysis_results['spatiotemporal_report']
        )
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# To run this API, save it as SocialDetect.py and run the following command in your terminal:
# uvicorn SocialDetect:app --reload