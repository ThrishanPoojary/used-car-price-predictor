
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

# Check the dataset folder first, then the project root
DATA_PATH = BASE_DIR / "dataset" / "car_details.csv"

if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "car_details.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "CSV not found. Check the dataset file location and name."
    )


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original records:", len(df))


# ==========================================
# 3. DATA CLEANING
# ==========================================

# Remove exact duplicate records
df = df.drop_duplicates().copy()

# Remove records with invalid selling prices
df = df[df["selling_price"] > 0].copy()

print("Records after cleaning:", len(df))


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

# Extract the car brand from its name
df["brand"] = df["name"].str.strip().str.split().str[0]

# Select input features
features = [
    "brand",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

X = df[features]
y = df["selling_price"]


# ==========================================
# 5. PREPROCESSING
# ==========================================

categorical_features = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 6. CREATE MACHINE LEARNING MODEL
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),

    ("regressor", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])


# ==========================================
# 7. SPLIT TRAINING AND TEST DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining your model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 9. EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE:  Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

print("=======================================")


# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

MODEL_PATH = BASE_DIR / "model.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)

