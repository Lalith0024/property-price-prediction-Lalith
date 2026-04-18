# Input Nodes
# These helpers prepare property inputs for the LangGraph workflow.
# Prompt input is extracted with Groq when configured, while missing fields use training-data defaults.

import json
import re
import urllib.error
import urllib.request

from config import (
    FURNISH_MAP,
    GROQ_API_URL,
    GROQ_MODEL,
    INT_COLUMNS,
    LABELS,
    NEIGHBORHOODS,
    RAW_NUMERIC_COLUMNS,
    config_value,
)


# Return the Groq settings used by the prompt extraction node.
def input_settings_from(secrets):
    api_key = config_value(secrets, "GROQ_API_KEY")
    return {
        "api_key": api_key,
        "api_url": config_value(secrets, "GROQ_API_URL") or GROQ_API_URL,
        "model": config_value(secrets, "GROQ_MODEL") or GROQ_MODEL,
    }


# Normalize prompt text before the offline fallback parser checks patterns.
def clean_prompt_text(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# Try a list of regex patterns and return the first numeric match.
def extract_number(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


# Small offline fallback used only when GROQ_API_KEY is not configured.
def parse_prompt_with_rules(prompt):
    text = clean_prompt_text(prompt)
    number = r"(\d+(?:\.\d+)?)"

    numeric_patterns = {
        "Total_Square_Footage": [
            rf"(?:total\s*)?(?:square\s*footage|area|size|built[-\s]?up|carpet)\D{{0,30}}{number}\s*(?:sq\.?\s*ft|sqft|sft|square\s*feet)?",
            rf"{number}\s*(?:sq\.?\s*ft|sqft|sft|square\s*feet|square\s*foot)",
        ],
        "Bedrooms": [
            rf"{number}\s*(?:bhk|bed(?:room)?s?|br)\b",
            rf"(?:bhk|bed(?:room)?s?|br)\D{{0,12}}{number}",
        ],
        "Bathrooms": [
            rf"{number}\s*(?:bath(?:room)?s?|ba)\b",
            rf"(?:bath(?:room)?s?|ba)\D{{0,12}}{number}",
        ],
        "Age_of_Property": [
            rf"{number}\s*(?:years?|yrs?)\s*(?:old|age)",
            rf"(?:age(?:\s*of\s*property)?|property\s*age)\D{{0,20}}{number}",
        ],
        "Floor_Number": [
            rf"{number}(?:st|nd|rd|th)?\s*(?:floor|level)\b",
            rf"(?:floor|level)\D{{0,12}}{number}",
        ],
        "Distance_to_City_Center_km": [
            rf"{number}\s*(?:km|kilometers?|kilometres?)\s*(?:from|away\s*from|to)?\s*(?:city\s*(?:center|centre)|downtown)",
            rf"(?:city\s*(?:center|centre)|downtown)\D{{0,35}}{number}\s*(?:km|kilometers?|kilometres?)",
        ],
        "Proximity_to_Public_Transport_km": [
            rf"{number}\s*(?:km|kilometers?|kilometres?)\s*(?:from|away\s*from|to)?\s*(?:metro|public\s*transport|bus|railway|train)",
            rf"(?:metro|public\s*transport|bus|railway|train)\D{{0,35}}{number}\s*(?:km|kilometers?|kilometres?)",
        ],
        "Crime_Index": [rf"(?:crime(?:\s*index)?)\D{{0,18}}{number}"],
        "Air_Quality_Index": [rf"(?:aqi|air\s*quality(?:\s*index)?)\D{{0,18}}{number}"],
        "Neighborhood_Growth_Rate_%": [
            rf"(?:neighbou?rhood\s*)?(?:growth(?:\s*rate)?)\D{{0,18}}{number}\s*%?",
            rf"{number}\s*%\s*(?:neighbou?rhood\s*)?growth",
        ],
        "Price_per_SqFt": [
            rf"(?:price|rate)\s*(?:per|/)\s*(?:sq\.?\s*ft|sqft|sft|square\s*foot)\D{{0,22}}{number}",
            rf"{number}\s*(?:rs|inr)?\s*(?:per|/)\s*(?:sq\.?\s*ft|sqft|sft|square\s*foot)",
        ],
        "Annual_Property_Tax": [
            rf"(?:annual\s*)?(?:property\s*)?tax\D{{0,22}}{number}",
            rf"{number}\s*(?:rs|inr)?\s*(?:annual\s*)?(?:property\s*)?tax",
        ],
        "Estimated_Rental_Yield_%": [
            rf"(?:estimated\s*)?(?:rental\s*)?yield\D{{0,18}}{number}\s*%?",
            rf"{number}\s*%\s*(?:estimated\s*)?(?:rental\s*)?yield",
        ],
    }

    parsed = {}
    for col, patterns in numeric_patterns.items():
        value = extract_number(text, patterns)
        if value is not None:
            parsed[col] = value

    furnishing = None
    if re.search(r"\bunfurnished\b", text):
        furnishing = "Unfurnished"
    elif re.search(r"\bsemi[-\s]?furnished\b|\bsemi furnished\b", text):
        furnishing = "Semi-furnished"
    elif re.search(r"\bfully[-\s]?furnished\b|\bfull furnished\b|\bfurnished\b", text):
        furnishing = "Fully-furnished"

    neighborhood = None
    neighborhood_aliases = [
        ("IT Hub", r"\bit\s*hub\b|\btech\s*hub\b|\bit corridor\b"),
        ("Industrial", r"\bindustrial\b|\bfactory\b|\bmanufacturing\b"),
        ("Residential", r"\bresidential\b|\bhousing\b"),
        ("Suburban", r"\bsuburban\b|\bsuburb\b|\boutskirts\b"),
        ("Downtown", r"\bdowntown\b|\bcbd\b|\bcentral\b"),
    ]
    for label, pattern in neighborhood_aliases:
        if re.search(pattern, text):
            neighborhood = label
            break

    return parsed, furnishing, neighborhood


# Ask Groq to convert the user's text prompt into model-ready JSON fields.
def parse_prompt_with_api(prompt, settings):
    if not settings.get("api_key"):
        raise ValueError("API key is missing")

    system_prompt = (
        "You extract real-estate prediction inputs as JSON only. "
        "Return keys: numeric_inputs, furnishing_status, neighborhood. "
        "numeric_inputs must contain only fields from this list when present in the user's text: "
        f"{RAW_NUMERIC_COLUMNS}. "
        f"furnishing_status must be one of {list(FURNISH_MAP.keys())} or null. "
        f"neighborhood must be one of {NEIGHBORHOODS} or null. "
        "Do not invent missing values. Use numbers only, with km and percentages as plain numeric values."
    )
    payload = {
        "model": settings["model"],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
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
        raise RuntimeError(f"API request failed with status {exc.code}: {detail[:300]}") from exc

    content = body["choices"][0]["message"]["content"]
    extracted = json.loads(content)
    numeric = extracted.get("numeric_inputs") or {}
    numeric = {key: float(value) for key, value in numeric.items() if key in RAW_NUMERIC_COLUMNS and value is not None}

    furnishing = extracted.get("furnishing_status")
    if furnishing not in FURNISH_MAP:
        furnishing = None

    neighborhood = extracted.get("neighborhood")
    if neighborhood not in NEIGHBORHOODS:
        neighborhood = None

    return numeric, furnishing, neighborhood


# Use Groq when a key exists; otherwise use the local fallback parser.
def parse_prompt(prompt, settings):
    if settings.get("api_key"):
        numeric, furnishing, neighborhood = parse_prompt_with_api(prompt, settings)
        return numeric, furnishing, neighborhood, "Groq extracted", None

    numeric, furnishing, neighborhood = parse_prompt_with_rules(prompt)
    return numeric, furnishing, neighborhood, "Rule parser fallback", "GROQ_API_KEY is not configured."


# Merge extracted prompt values with defaults for fields the user did not mention.
def assemble_prompt_fields(prompt, defaults, default_furnishing, default_neighborhood, settings):
    parsed_numeric, parsed_furnishing, parsed_neighborhood, source_label, warning = parse_prompt(prompt, settings)
    numeric_inputs = {}
    sources = {}

    for col in RAW_NUMERIC_COLUMNS:
        value = parsed_numeric.get(col, defaults.get(col, 0.0))
        if col in INT_COLUMNS:
            value = int(round(value))
        else:
            value = float(value)
        numeric_inputs[col] = value
        sources[col] = source_label if col in parsed_numeric else "Training-data default"

    furnishing = parsed_furnishing or default_furnishing
    neighborhood = parsed_neighborhood or default_neighborhood
    sources["Furnishing_Status"] = source_label if parsed_furnishing else "Training-data default"
    sources["Neighborhood"] = source_label if parsed_neighborhood else "Training-data default"

    return {
        "prompt": prompt,
        "numeric_inputs": numeric_inputs,
        "furnishing": furnishing,
        "neighborhood": neighborhood,
        "sources": sources,
        "agent_source": source_label,
        "agent_warning": warning,
    }


# Build the review-table rows shown to the user before prediction.
def review_rows(flow):
    rows = []
    for col in RAW_NUMERIC_COLUMNS:
        rows.append(
            {
                "Field": LABELS[col],
                "Value": flow["numeric_inputs"][col],
                "Source": flow["sources"].get(col, "Training-data default"),
            }
        )
    rows.append(
        {
            "Field": "Furnishing Status",
            "Value": flow["furnishing"],
            "Source": flow["sources"].get("Furnishing_Status", "Training-data default"),
        }
    )
    rows.append(
        {
            "Field": "Neighborhood",
            "Value": flow["neighborhood"],
            "Source": flow["sources"].get("Neighborhood", "Training-data default"),
        }
    )
    return rows
