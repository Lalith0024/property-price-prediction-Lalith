from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "real_estate_clean.csv"
REG_MODEL_PATH = BASE_DIR / "models" / "xgb_regression_model.joblib"
REG_SCALER_PATH = BASE_DIR / "models" / "regression_scaler.joblib"
CLF_MODEL_PATH = BASE_DIR / "models" / "xgb_classification_model.joblib"
CLF_SCALER_PATH = BASE_DIR / "models" / "classification_scaler.joblib"
HOSTED_APP_URL = "https://property-price-prediction-real-estate.streamlit.app/"

FURNISH_MAP = {"Unfurnished": 0, "Semi-furnished": 1, "Fully-furnished": 2}
NEIGHBORHOODS = ["Downtown", "IT Hub", "Industrial", "Residential", "Suburban"]
INT_COLUMNS = {"Bedrooms", "Bathrooms", "Age_of_Property", "Floor_Number"}
GRADE_LABELS = {0: "0 - Low", 1: "1 - Medium", 2: "2 - High"}

FEATURE_COLUMNS_FALLBACK = [
    "Total_Square_Footage",
    "Bedrooms",
    "Bathrooms",
    "Age_of_Property",
    "Floor_Number",
    "Furnishing_Status",
    "Distance_to_City_Center_km",
    "Proximity_to_Public_Transport_km",
    "Crime_Index",
    "Air_Quality_Index",
    "Neighborhood_Growth_Rate_%",
    "Price_per_SqFt",
    "Annual_Property_Tax",
    "Estimated_Rental_Yield_%",
    "Neighborhood_IT Hub",
    "Neighborhood_Industrial",
    "Neighborhood_Residential",
    "Neighborhood_Suburban",
]

RAW_NUMERIC_COLUMNS = [
    "Total_Square_Footage",
    "Bedrooms",
    "Bathrooms",
    "Age_of_Property",
    "Floor_Number",
    "Distance_to_City_Center_km",
    "Proximity_to_Public_Transport_km",
    "Crime_Index",
    "Air_Quality_Index",
    "Neighborhood_Growth_Rate_%",
    "Price_per_SqFt",
    "Annual_Property_Tax",
    "Estimated_Rental_Yield_%",
]

LABELS = {
    "Total_Square_Footage": "Total Square Footage",
    "Bedrooms": "Bedrooms",
    "Bathrooms": "Bathrooms",
    "Age_of_Property": "Age of Property (years)",
    "Floor_Number": "Floor Number",
    "Distance_to_City_Center_km": "Distance to City Center (km)",
    "Proximity_to_Public_Transport_km": "Proximity to Public Transport (km)",
    "Crime_Index": "Crime Index",
    "Air_Quality_Index": "Air Quality Index",
    "Neighborhood_Growth_Rate_%": "Neighbourhood Growth Rate (%)",
    "Price_per_SqFt": "Price per Sq Ft",
    "Annual_Property_Tax": "Annual Property Tax",
    "Estimated_Rental_Yield_%": "Estimated Rental Yield (%)",
}

PROMPT_EXAMPLE = (
    "Example: 1450 sqft 3 BHK 2 bathrooms, 6 years old, 8th floor, "
    "semi furnished, IT Hub, 4.5 km from city center, 0.7 km from metro, "
    "crime index 32, AQI 88, growth 9%, price per sqft 7200, "
    "annual tax 85000, rental yield 4.2%."
)
