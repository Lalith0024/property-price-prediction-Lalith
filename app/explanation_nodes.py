# Explanation Nodes
# These helpers turn model outputs into short plain-English explanations for the user.
# Groq is used when configured, with a local fallback message if Groq is unavailable.

import json
import urllib.error
import urllib.request

from config import GRADE_LABELS, GROQ_API_URL, GROQ_MODEL, LABELS, RAW_NUMERIC_COLUMNS, config_value


# Return the Groq settings used by the explanation step.
def explanation_settings_from(secrets):
    api_key = config_value(secrets, "GROQ_API_KEY")
    return {
        "api_key": api_key,
        "api_url": config_value(secrets, "GROQ_API_URL") or GROQ_API_URL,
        "model": config_value(secrets, "GROQ_MODEL") or GROQ_MODEL,
    }


# Build a compact property summary so Groq receives only useful context.
def summarize_inputs(flow):
    if not flow:
        return {}

    summary = {LABELS[col]: flow["numeric_inputs"].get(col) for col in RAW_NUMERIC_COLUMNS}
    summary["Furnishing Status"] = flow.get("furnishing")
    summary["Neighborhood"] = flow.get("neighborhood")
    return summary


# Ask Groq for a short explanation of the prediction.
def explain_with_groq(flow, result, settings):
    prompt = {
        "property_inputs": summarize_inputs(flow),
        "prediction": {
            "predicted_price": round(result["price"], 2),
            "investment_grade": GRADE_LABELS.get(result["grade"], str(result["grade"])),
            "confidence": round(result["confidence"], 4),
        },
    }
    system_prompt = (
        "You explain real-estate ML predictions for a student project. "
        "Use 3 to 5 short bullet points. "
        "Do not claim exact model feature importance. Explain likely reasons using the provided inputs."
    )
    payload = {
        "model": settings["model"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(prompt)},
        ],
    }
    request = urllib.request.Request(
        settings["api_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "property-price-prediction-streamlit/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq explanation failed with status {exc.code}: {detail[:300]}") from exc

    return body["choices"][0]["message"]["content"].strip()


# Return either a Groq explanation or a clear fallback explanation.
def generate_explanation(flow, result, settings):
    if settings.get("api_key"):
        return explain_with_groq(flow, result, settings)

    grade = GRADE_LABELS.get(result["grade"], str(result["grade"]))
    return (
        f"The model predicts a price of Rs {result['price']:,.0f} and an investment grade of {grade}. "
        "This explanation is generated locally because GROQ_API_KEY is not configured."
    )
