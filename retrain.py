# ============================================================
# STUDENT ACADEMIC PERFORMANCE FORECASTING
# Deep Learning - Feedforward Neural Network (FNN)
# Time-Series Forecasting Approach
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. PROJECT PATHS
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model" / "student-mat.csv"
MODEL_DIR = BASE_DIR / "model"

# Create model directory if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n==============================================")
print("     STUDENT PERFORMANCE FORECASTING")
print("==============================================")

print("\nLoading dataset...")

data = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\n==============================================")
print("     DATASET INFORMATION")
print("==============================================")

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst five rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum().sum())


# ============================================================
# 4. DEFINE FORECASTING TARGET
# ============================================================

# G1 = first assessment grade
# G2 = second assessment grade
# G3 = final grade
#
# The forecasting task is:
#
#       G1 + G2 + student information
#                    ↓
#              Deep Learning
#                    ↓
#              Forecast G3
#
# Therefore G3 is the numerical forecasting target.

TARGET_COLUMN = "G3"

X = data.drop(columns=[TARGET_COLUMN])

y = data[TARGET_COLUMN]


print("\n==============================================")
print("     FORECASTING TARGET")
print("==============================================")

print("Target column:", TARGET_COLUMN)
print("Target type: Numerical final grade")

print("\nTarget statistics:")
print(y.describe())


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\n==============================================")
print("     FEATURE INFORMATION")
print("==============================================")

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of categorical features:",
      len(categorical_features))

print("\nNumerical features:")
print(numerical_features)

print("\nNumber of numerical features:",
      len(numerical_features))


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

# We split the raw data BEFORE fitting the preprocessing
# components to avoid data leakage.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n==============================================")
print("     TRAIN / TEST SPLIT")
print("==============================================")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 7. DATA PREPROCESSING
# ============================================================

# Categorical variables:
# One-hot encoding
#
# Numerical variables:
# Passed through the preprocessor first, then standardized
# using StandardScaler.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# Fit ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the already fitted preprocessor
X_test_processed = preprocessor.transform(X_test)


print("\n==============================================")
print("     DATA PREPROCESSING")
print("==============================================")

print("Processed training shape:",
      X_train_processed.shape)

print("Processed testing shape:",
      X_test_processed.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train_processed)

# Transform test data
X_test_scaled = scaler.transform(X_test_processed)


print("\n==============================================")
print("     FEATURE SCALING")
print("==============================================")

print("Scaled training shape:",
      X_train_scaled.shape)

print("Scaled testing shape:",
      X_test_scaled.shape)


# Convert target values to NumPy arrays
y_train_array = y_train.to_numpy(dtype=np.float32)
y_test_array = y_test.to_numpy(dtype=np.float32)


# ============================================================
# 9. BUILD FNN DEEP LEARNING MODEL
# ============================================================

input_features = X_train_scaled.shape[1]


model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(input_features,)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        16,
        activation="relu"
    ),

    # Linear output because we are predicting
    # a numerical grade rather than a class.
    Dense(
        1,
        activation="linear"
    )
])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\n==============================================")
print("     FNN MODEL ARCHITECTURE")
print("==============================================")

model.summary()


# ============================================================
# 11. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n==============================================")
print("     MODEL TRAINING")
print("==============================================")

history = model.fit(
    X_train_scaled,
    y_train_array,

    validation_split=0.20,

    epochs=100,

    batch_size=16,

    callbacks=[early_stopping],

    verbose=1
)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

print("\n==============================================")
print("     MODEL EVALUATION")
print("==============================================")


# Predict G3 for the test dataset
y_pred = model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# RMSE
rmse = np.sqrt(
    mean_squared_error(
        y_test_array,
        y_pred
    )
)


# R-squared
r2 = r2_score(
    y_test_array,
    y_pred
)


# MAE
mae = np.mean(
    np.abs(
        y_test_array - y_pred
    )
)


print("\nForecasting Results:")

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R² Score: {r2:.4f}"
)

print(
    f"MAE: {mae:.4f}"
)


# ============================================================
# 14. SHOW SAMPLE FORECASTS
# ============================================================

print("\n==============================================")
print("     SAMPLE FORECASTS")
print("==============================================")

results = pd.DataFrame({
    "Actual_G3": y_test_array,
    "Forecasted_G3": y_pred
})

results["Error"] = (
    results["Actual_G3"]
    - results["Forecasted_G3"]
)

print(
    results.head(10).to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

HISTORY_PATH = MODEL_DIR / "training_history.csv"

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "student_performance_forecasting_model.keras"

model.save(
    MODEL_PATH
)


# ============================================================
# 17. SAVE PREPROCESSOR
# ============================================================

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


# ============================================================
# 18. SAVE SCALER
# ============================================================

SCALER_PATH = MODEL_DIR / "scaler.pkl"

joblib.dump(
    scaler,
    SCALER_PATH
)


# ============================================================
# 19. SAVE FEATURE INFORMATION
# ============================================================

FEATURE_INFO_PATH = MODEL_DIR / "feature_information.pkl"

feature_information = {
    "target": TARGET_COLUMN,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "input_features": input_features
}

joblib.dump(
    feature_information,
    FEATURE_INFO_PATH
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("     FORECASTING COMPLETED SUCCESSFULLY")
print("==============================================")

print(
    f"\nRMSE: {rmse:.4f}"
)

print(
    f"R² Score: {r2:.4f}"
)

print(
    f"MAE: {mae:.4f}"
)

print("\nFiles saved:")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Preprocessor:",
    PREPROCESSOR_PATH
)

print(
    "Scaler:",
    SCALER_PATH
)

print(
    "Training history:",
    HISTORY_PATH
)

print(
    "Feature information:",
    FEATURE_INFO_PATH
)

print("\nProcess finished successfully.")

# ============================================================
# SAVE MODEL PERFORMANCE METRICS
# ============================================================

import joblib


metrics = {
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2
}


joblib.dump(
    metrics,
    MODEL_DIR / "metrics.pkl"
)


print("Metrics saved successfully.")