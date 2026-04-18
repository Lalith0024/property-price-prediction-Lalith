# Notification Nodes
# These helpers send prediction outputs from the LangGraph workflow to the recipient entered in the UI.
# Sender credentials come from .env or Streamlit secrets, so the user only enters the receiver email.

import smtplib
import re
from email.message import EmailMessage
from email.utils import formataddr
from html import escape
from re import fullmatch

from config import GRADE_LABELS, LABELS, RAW_NUMERIC_COLUMNS, config_value

EXPLANATION_HIGHLIGHT_STYLE = (
    "background:#fff4d8;color:#6b4700;padding:2px 5px;"
    "border-radius:5px;font-weight:700;"
)

EXPLANATION_KEY_TERMS = (
    "predicted price",
    "investment grade",
    "confidence",
    "city center",
    "public transport",
    "square footage",
    "bedrooms",
    "bathrooms",
    "neighborhood growth",
    "crime index",
    "air quality",
    "price per sq ft",
    "annual tax",
    "rental yield",
    "furnishing status",
    "location",
    "accessibility",
    "prompt",
    "default",
    "edited",
)

SOURCE_LABELS = {
    "Groq extracted": "Prompt",
    "Rule parser fallback": "Prompt",
    "Training-data default": "Default",
    "Edited by user": "Edited",
}


# Load email settings from local .env, environment variables, or Streamlit secrets.
def email_settings_from(secrets):
    return {
        "host": config_value(secrets, "EMAIL_SMTP_HOST") or "smtp.gmail.com",
        "port": int(config_value(secrets, "EMAIL_SMTP_PORT") or 587),
        "sender": config_value(secrets, "EMAIL_SENDER"),
        "password": config_value(secrets, "EMAIL_APP_PASSWORD"),
        "from_name": config_value(secrets, "EMAIL_FROM_NAME") or "Property Predictor",
    }


# Check the recipient email format before trying SMTP.
def is_valid_email(email):
    return bool(fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or ""))


# Format values cleanly before placing them in email tables.
def format_value(value):
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# Convert internal source labels into user-friendly email badges.
def display_source(source):
    return SOURCE_LABELS.get(source, source or "Default")


# Build the full property summary for the email body.
def property_summary_rows(flow):
    if not flow:
        return []

    rows = []
    sources = flow.get("sources", {})
    for col in RAW_NUMERIC_COLUMNS:
        rows.append(
            {
                "field": LABELS[col],
                "value": format_value(flow["numeric_inputs"].get(col)),
                "source": display_source(sources.get(col)),
            }
        )
    rows.append(
        {
            "field": "Furnishing Status",
            "value": format_value(flow.get("furnishing")),
            "source": display_source(sources.get("Furnishing_Status")),
        }
    )
    rows.append(
        {
            "field": "Neighborhood",
            "value": format_value(flow.get("neighborhood")),
            "source": display_source(sources.get("Neighborhood")),
        }
    )
    return rows


# Strip simple Markdown markers so plain-text email stays readable.
def clean_markdown(text):
    cleaned = (text or "").replace("**", "")
    return re.sub(r"^\s*\*\s+", "- ", cleaned, flags=re.MULTILINE)


# Highlight important terms inside plain escaped text.
def highlight_plain_terms(text):
    highlighted = text
    for term in sorted(EXPLANATION_KEY_TERMS, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w)({re.escape(term)})(?!\w)", re.IGNORECASE)
        highlighted = pattern.sub(
            lambda match: f"<strong style='{EXPLANATION_HIGHLIGHT_STYLE}'>{match.group(1)}</strong>",
            highlighted,
        )
    return highlighted


# Convert Groq Markdown emphasis and key real-estate terms into safe highlighted HTML.
def highlight_explanation_text(text):
    parts = re.split(r"(\*\*.*?\*\*)", text or "")
    html_parts = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            phrase = escape(part[2:-2].strip())
            if phrase:
                html_parts.append(
                    f"<strong style='{EXPLANATION_HIGHLIGHT_STYLE}'>{phrase}</strong>"
                )
        else:
            html_parts.append(highlight_plain_terms(escape(part)))
    return "".join(html_parts)


# Detect whether an explanation line should be displayed as a bullet.
def explanation_line_parts(raw_line):
    line = raw_line.strip()
    if line.startswith("- ") or line.startswith("* "):
        return True, line[2:].strip()
    if line.startswith("-") or line.startswith("*"):
        return True, line[1:].strip()
    return False, line


# Format a plain-text fallback version of the prediction email.
def build_prediction_email_text(result, explanation, flow=None):
    grade = GRADE_LABELS.get(result["grade"], str(result["grade"]))
    lines = [
        "Hello,",
        "",
        "Your property prediction is ready.",
        "",
        f"Predicted Price: Rs {result['price']:,.0f}",
        f"Investment Grade: {grade}",
        f"Confidence: {result['confidence']:.1%}",
    ]

    summary = property_summary_rows(flow)
    if summary:
        lines.extend(["", "Property Summary:"])
        for row in summary:
            lines.append(f"- {row['field']}: {row['value']} ({row['source']})")

    if explanation:
        lines.extend(["", "Explanation:", clean_markdown(explanation)])

    lines.extend(["", "Thank you,", "Property Predictor"])
    return "\n".join(lines)


# Create one result card for the HTML email.
def metric_card(label, value):
    return f"""
    <td style="width:33.3%;padding:10px;">
      <table role="presentation" style="width:100%;border-collapse:collapse;background:#111111;border-radius:10px;">
        <tr>
          <td style="padding:16px;">
            <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#b7b7b7;">{escape(label)}</div>
            <div style="font-size:22px;line-height:1.3;font-weight:700;margin-top:8px;color:#ffffff;">{escape(value)}</div>
          </td>
        </tr>
      </table>
    </td>
    """


# Create a colored source badge for Prompt, Default, or Edited values.
def source_badge(source):
    colors = {
        "Prompt": ("#E8F3FF", "#0B5CAD"),
        "Default": ("#FFF4D8", "#8A5B00"),
        "Edited": ("#EAF8EA", "#247A2E"),
    }
    background, color = colors.get(source, ("#EEEEEE", "#333333"))
    return (
        f"<span style='display:inline-block;border-radius:999px;padding:4px 9px;"
        f"font-size:12px;font-weight:700;background:{background};color:{color};'>{escape(source)}</span>"
    )


# Build the property summary table for the HTML email.
def property_summary_table(flow):
    rows = property_summary_rows(flow)
    if not rows:
        return ""

    table_rows = ""
    for row in rows:
        table_rows += f"""
        <tr>
          <td style="padding:10px;border-bottom:1px solid #eeeeee;font-weight:600;">{escape(row['field'])}</td>
          <td style="padding:10px;border-bottom:1px solid #eeeeee;">{escape(row['value'])}</td>
          <td style="padding:10px;border-bottom:1px solid #eeeeee;">{source_badge(row['source'])}</td>
        </tr>
        """

    return f"""
    <h2 style="font-size:18px;margin:26px 0 10px;">Property Summary</h2>
    <p style="margin:0 0 12px;color:#555555;">Source tells you where each value came from.</p>
    <table style="width:100%;border-collapse:collapse;border:1px solid #eeeeee;border-radius:10px;overflow:hidden;">
      <thead>
        <tr style="background:#f7f6f3;text-align:left;">
          <th style="padding:10px;">Field</th>
          <th style="padding:10px;">Value</th>
          <th style="padding:10px;">Source</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    """


# Convert explanation bullets into clean HTML.
def explanation_html(explanation):
    if not explanation:
        return ""

    sections = []
    bullet_lines = []

    # Close the current bullet group before adding normal paragraphs.
    def flush_bullets():
        if bullet_lines:
            sections.append(
                "<ul style='padding-left:20px;margin:8px 0;color:#333333;'>"
                + "".join(bullet_lines)
                + "</ul>"
            )
            bullet_lines.clear()

    for raw_line in explanation.splitlines():
        is_bullet, line = explanation_line_parts(raw_line)
        if not line:
            continue
        formatted_line = highlight_explanation_text(line)
        if is_bullet:
            bullet_lines.append(f"<li style='margin:8px 0;'>{formatted_line}</li>")
        else:
            flush_bullets()
            sections.append(f"<p style='margin:8px 0;color:#333333;'>{formatted_line}</p>")

    flush_bullets()
    body = "".join(sections)
    return f"""
    <h2 style="font-size:18px;margin:26px 0 10px;">Quick Explanation</h2>
    <div style="background:#f7f6f3;border-radius:10px;padding:14px;">{body}</div>
    """


# Format a rich HTML email that is easy for users to scan quickly.
def build_prediction_email_html(result, explanation, flow=None):
    grade = GRADE_LABELS.get(result["grade"], str(result["grade"]))
    return f"""
    <!doctype html>
    <html>
      <body style="margin:0;padding:0;background:#f5f5f5;font-family:Arial,sans-serif;color:#1c1c1c;">
        <div style="max-width:760px;margin:0 auto;padding:24px;">
          <div style="background:#ffffff;border-radius:14px;padding:24px;border:1px solid #e7e7e7;">
            <div style="font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:#777777;font-weight:700;">
              Property Predictor
            </div>
            <h1 style="font-size:24px;margin:8px 0 8px;">Your property prediction is ready</h1>
            <p style="margin:0 0 18px;color:#555555;">
              We used your confirmed property details to estimate price and investment grade.
            </p>

            <table style="width:100%;border-collapse:collapse;margin:10px 0 18px;">
              <tr>
                {metric_card("Predicted Price", f"Rs {result['price']:,.0f}")}
                {metric_card("Investment Grade", grade)}
                {metric_card("Confidence", f"{result['confidence']:.1%}")}
              </tr>
            </table>

            <div style="background:#eef6ff;border-left:4px solid #0b5cad;border-radius:8px;padding:12px;margin:16px 0;">
              <strong>How to read this:</strong> Prompt values came from your input, Default values were filled by the app, and Edited values were changed by you before prediction.
            </div>

            {property_summary_table(flow)}
            {explanation_html(explanation)}

            <p style="margin-top:24px;color:#555555;">Thank you,<br><strong>Property Predictor</strong></p>
          </div>
        </div>
      </body>
    </html>
    """


# Send the prediction email through SMTP.
def send_prediction_email(recipient, result, explanation, flow, settings):
    if not is_valid_email(recipient):
        raise ValueError("Enter a valid recipient email address.")
    if not settings.get("sender") or not settings.get("password"):
        raise ValueError("EMAIL_SENDER and EMAIL_APP_PASSWORD must be configured in .env.")

    message = EmailMessage()
    message["Subject"] = "Your Property Prediction Result"
    message["From"] = formataddr((settings["from_name"], settings["sender"]))
    message["To"] = recipient
    message.set_content(build_prediction_email_text(result, explanation, flow))
    message.add_alternative(build_prediction_email_html(result, explanation, flow), subtype="html")

    with smtplib.SMTP(settings["host"], settings["port"]) as smtp:
        smtp.starttls()
        smtp.login(settings["sender"], settings["password"])
        smtp.send_message(message)

    return True


# Send CSV batch predictions as an email attachment.
def send_csv_predictions_email(recipient, output_df, settings):
    if not is_valid_email(recipient):
        raise ValueError("Enter a valid recipient email address.")
    if not settings.get("sender") or not settings.get("password"):
        raise ValueError("EMAIL_SENDER and EMAIL_APP_PASSWORD must be configured in .env.")

    message = EmailMessage()
    message["Subject"] = "Your Batch Property Predictions"
    message["From"] = formataddr((settings["from_name"], settings["sender"]))
    message["To"] = recipient
    message.set_content(
        "Hello,\n\nYour batch property prediction CSV is attached.\n\nThank you,\nProperty Predictor"
    )
    message.add_attachment(
        output_df.to_csv(index=False).encode("utf-8"),
        maintype="text",
        subtype="csv",
        filename="property_predictions.csv",
    )

    with smtplib.SMTP(settings["host"], settings["port"]) as smtp:
        smtp.starttls()
        smtp.login(settings["sender"], settings["password"])
        smtp.send_message(message)

    return True
