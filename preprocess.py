import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()

    # 🔥 Normalize column names (THIS SOLVES EVERYTHING)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print("Normalized columns:", df.columns.tolist())

    # Features (normalized names)
    feature_columns = [
        "tenure",
        "monthly_charge",
        "contract",
        "payment_method",
        "internet_service",
        "tech_support"
    ]

    # Target
    target_column = "churn_label"

    # Keep only existing columns
    existing_features = [c for c in feature_columns if c in df.columns]

    if target_column not in df.columns:
        raise ValueError("❌ Target column 'churn_label' not found in dataset")

    df = df[existing_features + [target_column]]

    # Drop missing values
    df.dropna(inplace=True)

    # Encode categorical features
    encoder = LabelEncoder()
    for col in existing_features:
        if df[col].dtype == "object":
            df[col] = encoder.fit_transform(df[col])

    # Encode target
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y