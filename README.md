POST http://127.0.0.1:8000/analyze/
body(JSON):
{
    "timestamp": "2025-08-23 09:00:00",
    "lat": 13.0827,
    "lon": 80.2707,
    "message": "[Official] IMD issues Red Alert for Chennai. Extremely heavy rainfall expected over the next 24 hours. #ChennaiRains"
}


res:

{
    "final_answer": "Positive",
    "verdict": "Verified Official",
    "summary": "The message claiming an official IMD Red Alert for Chennai due to extremely heavy rainfall is consistent with other verified reports, which include an identical IMD Red Alert and additional advisories for a severe weather event.",
    "confidence_score": 95,
    "initial_analysis": {
        "priority": "Critical",
        "priority_reason": "The message claims to be an 'Official' 'Red Alert' from the IMD (India Meteorological Department) regarding 'Extremely heavy rainfall' expected in 'the next 24 hours' for Chennai. This indicates an immediate, severe threat requiring urgent public awareness and potential action for safety.",
        "risk_level": "Medium",
        "flags": [
            "Unverified Source Claim: While the message states '[Official] IMD', this claim needs external verification. Misinformation often spreads through false attribution to credible sources.",
            "Lack of direct link to official source: The message does not provide a direct link to the IMD's official website, social media handle, or a specific official bulletin to easily verify its authenticity."
        ]
    },
    "spatiotemporal_analysis": {
        "consistency": "Consistent",
        "reason": "The primary report states that the IMD has issued a Red Alert for Chennai due to expected extremely heavy rainfall. This aligns perfectly with the other verified reports, which include an identical IMD Red Alert, advisories for residents to move to higher ground and prepare for evacuation due to flooding risks, and the suspension of Chennai Airport operations due to runway flooding. All reports collectively describe a severe weather event involving heavy rainfall and significant flooding in Chennai."
    }
}
