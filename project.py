# ======================================================================
# AI Project: Impact of Pollinator Population on Crop Yield
# B.Tech 1st Year – Full Working Code with Algorithm Steps
# ======================================================================

# ------------------------------------------------------------
# STEP 1 — IMPORT LIBRARIES  (Algorithm Step 1)
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

print("STEP 1: Libraries imported successfully.")


# ------------------------------------------------------------
# STEP 2 — LOAD THE DATASET  (Algorithm Step 2)
# ------------------------------------------------------------
df = pd.read_csv("pollinators_dataset_100rows.csv")
print("\nSTEP 2: Dataset loaded successfully.")
print(df.head())


# ------------------------------------------------------------
# STEP 3 — DATA PREPROCESSING  (Algorithm Step 3)
# ------------------------------------------------------------

# Fill missing values if any
df = df.fillna(df.median(numeric_only=True))

# Smooth pollinator count with rolling average
df["pollinator_avg"] = df["pollinator_count"].rolling(3, min_periods=1).mean()

# Define features and target
X = df[["pollinator_avg", "rainfall", "temperature", "fertilizer_use"]]
y = df["crop_yield"]

print("\nSTEP 3: Preprocessing completed.")


# ------------------------------------------------------------
# STEP 4 — VISUALIZATION  (Algorithm Step 4)
# ------------------------------------------------------------

os.makedirs("figures", exist_ok=True)

# ============== Figure 1: Pollinators Over Years ===============
plt.figure(figsize=(10, 5))
plt.plot(df["year"], df["pollinator_count"], marker="o", linewidth=2)

# Trendline
z = np.polyfit(df["year"], df["pollinator_count"], 1)
p = np.poly1d(z)
plt.plot(df["year"], p(df["year"]), "r--", label="Trend Line")

plt.title("Pollinator Population Over Years")
plt.xlabel("Year")
plt.ylabel("Pollinator Count (thousands)")
plt.grid(True)
plt.legend()

# Annotation
plt.annotate("General decline in pollinators",
             xy=(df["year"].iloc[-1], df["pollinator_count"].iloc[-1]),
             xytext=(df["year"].min()+2, df["pollinator_count"].max()),
             arrowprops=dict(arrowstyle="->", color='red', lw=2))

plt.tight_layout()
plt.savefig("figures/pollinators_over_years.png")
plt.show()


# ============== Figure 2: Pollinator vs Yield ===============
plt.figure(figsize=(7, 7))
plt.scatter(df["pollinator_count"], df["crop_yield"], s=80, alpha=0.7)

# Trendline
z = np.polyfit(df["pollinator_count"], df["crop_yield"], 1)
p = np.poly1d(z)
plt.plot(df["pollinator_count"], p(df["pollinator_count"]), "g--", label="Trend Line")

plt.title("Pollinator Count vs Crop Yield")
plt.xlabel("Pollinator Count (thousands)")
plt.ylabel("Crop Yield (tonnes/ha)")
plt.grid(True)
plt.legend()

plt.annotate("Higher pollinators = higher yield",
             xy=(df["pollinator_count"].max(), df["crop_yield"].max()),
             xytext=(df["pollinator_count"].max()-2000, df["crop_yield"].max()+1),
             arrowprops=dict(arrowstyle="->", color='green', lw=2))

plt.tight_layout()
plt.savefig("figures/pollinators_vs_yield.png")
plt.show()


print("\nSTEP 4: Visualizations created.")


# ------------------------------------------------------------
# STEP 5 — TRAIN-TEST SPLIT (Algorithm Step 5)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nSTEP 5: Train-test split completed.")


# ------------------------------------------------------------
# STEP 6 — TRAIN THE MODEL (Algorithm Step 6)
# ------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nSTEP 6: Model trained successfully.")


# ------------------------------------------------------------
# STEP 7 — PREDICT (Algorithm Step 7)
# ------------------------------------------------------------
y_pred = model.predict(X_test)
print("\nSTEP 7: Predictions generated.")


# ------------------------------------------------------------
# STEP 8 — EVALUATION (Algorithm Step 8)
# ------------------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nSTEP 8: Model Evaluation")
print("RMSE:", rmse)
print("R² Score:", r2)


# ------------------------------------------------------------
# STEP 9 — ACTUAL vs PREDICTED PLOT (Algorithm Step 9)
# ------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.7, s=70)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")

plt.title("Actual vs Predicted Crop Yield")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.grid(True)

plt.annotate("Closer to red line = better prediction",
             xy=(y_test.mean(), y_pred.mean()),
             xytext=(y_test.mean()-1, y_pred.mean()+1),
             arrowprops=dict(arrowstyle="->", color='blue', lw=2))

plt.tight_layout()
plt.savefig("figures/actual_vs_predicted.png")
plt.show()

print("\nSTEP 9: Actual vs Predicted plot completed.")
print("\nALL STEPS COMPLETED SUCCESSFULLY 🎉")
