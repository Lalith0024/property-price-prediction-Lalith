# Property Graph
# This file defines the LangGraph state, nodes, and graph runners for the property workflow.
# Streamlit calls these runners so the UI stays the same while the backend becomes node-based.

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from config import GRADE_LABELS, RAW_NUMERIC_COLUMNS
from explanation_nodes import explanation_settings_from, generate_explanation
from input_nodes import assemble_prompt_fields
from notification_nodes import email_settings_from, send_csv_predictions_email, send_prediction_email
from prediction_nodes import find_comparable_properties, predict_one_property, raw_to_feature_frame, run_predict
from report_generator import build_advisory_report


class PropertyWorkflowState(TypedDict, total=False):
    mode: str
    prompt: str
    context: dict[str, Any]
    secrets: Any
    input_settings: dict[str, Any]
    explanation_settings: dict[str, Any]
    email_settings: dict[str, Any]
    numeric_inputs: dict[str, Any]
    furnishing: str
    neighborhood: str
    flow: dict[str, Any]
    result: dict[str, Any]
    explanation: str
    comparables: list[dict[str, Any]]
    report: dict[str, Any]
    recipient: str
    email_sent: bool
    input_df: Any
    output_df: Any
    review_ready: bool


# Create the flow object used by review, prediction, explanation, report, and email.
def prompt_input_node(state: PropertyWorkflowState) -> dict[str, Any]:
    context = state["context"]
    default_furnishing, default_neighborhood = context["default_categories"]
    flow = assemble_prompt_fields(
        state["prompt"],
        context["defaults"],
        default_furnishing,
        default_neighborhood,
        state["input_settings"],
    )
    return {"mode": "prompt", "flow": flow}


# Mark prompt extraction as ready for the Streamlit human-review popup.
def review_node(state: PropertyWorkflowState) -> dict[str, Any]:
    return {"review_ready": True}


# Wrap manual form values into the same flow shape used by prompt values.
def manual_input_node(state: PropertyWorkflowState) -> dict[str, Any]:
    return {
        "mode": "manual",
        "flow": {
            "prompt": "Manual input",
            "numeric_inputs": state["numeric_inputs"],
            "furnishing": state["furnishing"],
            "neighborhood": state["neighborhood"],
            "sources": {
                **{col: "Edited by user" for col in RAW_NUMERIC_COLUMNS},
                "Furnishing_Status": "Edited by user",
                "Neighborhood": "Edited by user",
            },
            "agent_source": "Manual input",
            "agent_warning": None,
        },
    }


# Run the saved XGBoost models for one confirmed property.
def prediction_node(state: PropertyWorkflowState) -> dict[str, Any]:
    flow = state["flow"]
    result = predict_one_property(
        flow["numeric_inputs"],
        flow["furnishing"],
        flow["neighborhood"],
        state["context"],
    )
    comparables = find_comparable_properties(
        state["context"]["df"],
        flow["neighborhood"],
        flow["numeric_inputs"]["Total_Square_Footage"],
        result["price"],
    )
    return {"result": result, "comparables": comparables}


# Ask Groq for a short user-facing explanation after prediction.
def explanation_node(state: PropertyWorkflowState) -> dict[str, Any]:
    explanation = generate_explanation(
        state["flow"],
        state["result"],
        state["explanation_settings"],
        state.get("comparables"),
    )
    return {"explanation": explanation}


# Assemble a structured report after prediction and explanation are complete.
def report_node(state: PropertyWorkflowState) -> dict[str, Any]:
    report = build_advisory_report(
        state["flow"],
        state["result"],
        state.get("explanation"),
        state.get("comparables"),
    )
    return {"report": report}


# Check whether a CSV contains encoded model columns or raw user-facing columns.
def csv_column_status(input_df, context):
    feature_columns = context["feature_columns"]
    has_encoded = all(column in input_df.columns for column in feature_columns)
    has_raw = (
        all(column in input_df.columns for column in RAW_NUMERIC_COLUMNS)
        and "Furnishing_Status" in input_df.columns
        and "Neighborhood" in input_df.columns
    )
    return has_encoded, has_raw


# Run the saved models for every row in an uploaded CSV file.
def csv_prediction_node(state: PropertyWorkflowState) -> dict[str, Any]:
    input_df = state["input_df"]
    context = state["context"]
    feature_columns = context["feature_columns"]
    has_encoded, has_raw = csv_column_status(input_df, context)

    if not has_encoded and not has_raw:
        raise ValueError("Column mismatch. Provide encoded feature columns or raw columns with Furnishing_Status and Neighborhood.")

    features = input_df[feature_columns].copy() if has_encoded else raw_to_feature_frame(
        input_df,
        feature_columns,
        context["defaults"],
    )
    prices, grades, probs = run_predict(
        features,
        context["reg_model"],
        context["reg_scaler"],
        context["clf_model"],
        context["clf_scaler"],
    )

    output_df = input_df.copy()
    output_df["Predicted_Price_INR"] = prices
    output_df["Predicted_Advisory_Class"] = grades
    output_df["Predicted_Advisory_Recommendation"] = [
        GRADE_LABELS.get(int(grade), str(int(grade))) for grade in grades
    ]
    for index in range(probs.shape[1]):
        output_df[f"Advisory_Class_{index}_Probability"] = probs[:, index]

    return {"mode": "csv", "output_df": output_df}


# Send one prediction result email through the notification node.
def result_email_node(state: PropertyWorkflowState) -> dict[str, Any]:
    send_prediction_email(
        state["recipient"],
        state["result"],
        state.get("explanation"),
        state.get("flow"),
        state.get("comparables"),
        state["email_settings"],
    )
    return {"email_sent": True}


# Send a CSV batch prediction attachment through the notification node.
def csv_email_node(state: PropertyWorkflowState) -> dict[str, Any]:
    send_csv_predictions_email(
        state["recipient"],
        state["output_df"],
        state["email_settings"],
    )
    return {"email_sent": True}


# Compile the prompt extraction graph that pauses before human review in Streamlit.
def build_prompt_review_graph():
    graph = StateGraph(PropertyWorkflowState)
    graph.add_node("input_node", prompt_input_node)
    graph.add_node("review_node", review_node)
    graph.add_edge(START, "input_node")
    graph.add_edge("input_node", "review_node")
    graph.add_edge("review_node", END)
    return graph.compile()


# Compile the single-property graph used after prompt review or manual input.
def build_single_prediction_graph(include_manual_input=False):
    graph = StateGraph(PropertyWorkflowState)
    if include_manual_input:
        graph.add_node("input_node", manual_input_node)
        graph.add_node("prediction_node", prediction_node)
        graph.add_edge(START, "input_node")
        graph.add_edge("input_node", "prediction_node")
    else:
        graph.add_node("prediction_node", prediction_node)
        graph.add_edge(START, "prediction_node")

    graph.add_node("explanation_node", explanation_node)
    graph.add_node("report_node", report_node)
    graph.add_edge("prediction_node", "explanation_node")
    graph.add_edge("explanation_node", "report_node")
    graph.add_edge("report_node", END)
    return graph.compile()


# Compile the CSV batch prediction graph.
def build_csv_prediction_graph():
    graph = StateGraph(PropertyWorkflowState)
    graph.add_node("csv_prediction_node", csv_prediction_node)
    graph.add_edge(START, "csv_prediction_node")
    graph.add_edge("csv_prediction_node", END)
    return graph.compile()


# Compile the single-result email graph.
def build_result_email_graph():
    graph = StateGraph(PropertyWorkflowState)
    graph.add_node("notification_node", result_email_node)
    graph.add_edge(START, "notification_node")
    graph.add_edge("notification_node", END)
    return graph.compile()


# Compile the CSV attachment email graph.
def build_csv_email_graph():
    graph = StateGraph(PropertyWorkflowState)
    graph.add_node("notification_node", csv_email_node)
    graph.add_edge(START, "notification_node")
    graph.add_edge("notification_node", END)
    return graph.compile()


PROMPT_REVIEW_GRAPH = build_prompt_review_graph()
CONFIRMED_PREDICTION_GRAPH = build_single_prediction_graph(include_manual_input=False)
MANUAL_PREDICTION_GRAPH = build_single_prediction_graph(include_manual_input=True)
CSV_PREDICTION_GRAPH = build_csv_prediction_graph()
RESULT_EMAIL_GRAPH = build_result_email_graph()
CSV_EMAIL_GRAPH = build_csv_email_graph()


# Run prompt extraction and return a state ready for Streamlit review.
def run_prompt_review_graph(prompt, context, input_settings):
    return PROMPT_REVIEW_GRAPH.invoke(
        {
            "mode": "prompt",
            "prompt": prompt,
            "context": context,
            "input_settings": input_settings,
        }
    )


# Run prediction and explanation after the user confirms prompt values.
def run_confirmed_prediction_graph(flow, context, secrets):
    return CONFIRMED_PREDICTION_GRAPH.invoke(
        {
            "mode": "prompt",
            "flow": flow,
            "context": context,
            "explanation_settings": explanation_settings_from(secrets),
        }
    )


# Run manual input through the same prediction and explanation nodes.
def run_manual_prediction_graph(numeric_inputs, furnishing, neighborhood, context, secrets):
    return MANUAL_PREDICTION_GRAPH.invoke(
        {
            "mode": "manual",
            "numeric_inputs": numeric_inputs,
            "furnishing": furnishing,
            "neighborhood": neighborhood,
            "context": context,
            "explanation_settings": explanation_settings_from(secrets),
        }
    )


# Run an uploaded CSV through the batch prediction graph.
def run_csv_prediction_graph(input_df, context):
    return CSV_PREDICTION_GRAPH.invoke(
        {
            "mode": "csv",
            "input_df": input_df,
            "context": context,
        }
    )


# Run the notification node for one property result.
def run_result_email_graph(recipient, result, explanation, flow, comparables, secrets):
    return RESULT_EMAIL_GRAPH.invoke(
        {
            "recipient": recipient,
            "result": result,
            "explanation": explanation,
            "flow": flow,
            "comparables": comparables,
            "email_settings": email_settings_from(secrets),
        }
    )


# Run the notification node for CSV batch results.
def run_csv_email_graph(recipient, output_df, secrets):
    return CSV_EMAIL_GRAPH.invoke(
        {
            "recipient": recipient,
            "output_df": output_df,
            "email_settings": email_settings_from(secrets),
        }
    )
