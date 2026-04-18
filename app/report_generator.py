# Report Generator
# This module converts graph outputs into one structured advisory report.
# The UI renders this report as the main result, while email remains an optional delivery step.

from config import ADVISORY_DESCRIPTIONS, ADVISORY_LABELS, GRADE_LABELS, LABELS, RAW_NUMERIC_COLUMNS

DISCLAIMER_TEXT = (
    "This application is a student project for educational purposes only. "
    "It does not constitute financial, legal, or investment advice."
)

SOURCE_LABELS = {
    "Groq extracted": "Prompt",
    "Rule parser fallback": "Prompt",
    "Training-data default": "Default",
    "Edited by user": "Edited",
}


# Format report values so numeric fields are readable without noisy decimals.
def format_report_value(value):
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# Convert internal source names into short user-facing report labels.
def display_source(source):
    return SOURCE_LABELS.get(source, source or "Default")


# Build the confirmed property table used by the dedicated report view.
def build_property_summary(flow):
    if not flow:
        return []

    rows = []
    sources = flow.get("sources", {})
    for col in RAW_NUMERIC_COLUMNS:
        rows.append(
            {
                "Field": LABELS[col],
                "Value": format_report_value(flow["numeric_inputs"].get(col)),
                "Source": display_source(sources.get(col)),
            }
        )

    rows.append(
        {
            "Field": "Furnishing Status",
            "Value": format_report_value(flow.get("furnishing")),
            "Source": display_source(sources.get("Furnishing_Status")),
        }
    )
    rows.append(
        {
            "Field": "Neighborhood",
            "Value": format_report_value(flow.get("neighborhood")),
            "Source": display_source(sources.get("Neighborhood")),
        }
    )
    return rows


# Count how many values came from prompt extraction, defaults, and user edits.
def build_source_summary(flow):
    summary = {"Prompt": 0, "Default": 0, "Edited": 0}
    if not flow:
        return summary

    for source in flow.get("sources", {}).values():
        label = display_source(source)
        if label in summary:
            summary[label] += 1
    return summary


# Convert probability output into rows that are easy to show in the report.
def build_probability_rows(result):
    return [
        {"Advisory Class": grade, "Probability": f"{prob:.1%}"}
        for grade, prob in result.get("probabilities", {}).items()
    ]


# Create the complete structured advisory report from graph state.
def build_advisory_report(flow, result, explanation, comparables=None):
    grade = int(result["grade"])
    recommendation = ADVISORY_LABELS.get(grade, str(grade))
    advisory_class = GRADE_LABELS.get(grade, str(grade))

    return {
        "title": "Structured Property Advisory Report",
        "summary": {
            "predicted_price": f"Rs {result['price']:,.0f}",
            "advisory_recommendation": recommendation,
            "advisory_class": advisory_class,
            "confidence": f"{result['confidence']:.1%}",
        },
        "recommendation": {
            "label": recommendation,
            "class": advisory_class,
            "meaning": ADVISORY_DESCRIPTIONS.get(grade, ""),
        },
        "model_evidence": [
            f"Predicted price: Rs {result['price']:,.0f}",
            f"Advisory class: {advisory_class}",
            f"Confidence: {result['confidence']:.1%}",
            f"Comparable rows used: {len(comparables or [])}",
        ],
        "property_summary": build_property_summary(flow),
        "source_summary": build_source_summary(flow),
        "probabilities": build_probability_rows(result),
        "comparables": comparables or [],
        "explanation": explanation or "",
        "risk_warning": (
            "Use this as an educational ML advisory report. Verify legal, financial, "
            "market, and property-condition details before making a real decision."
        ),
        "disclaimer": DISCLAIMER_TEXT,
    }
