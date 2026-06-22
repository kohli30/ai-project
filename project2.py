# ==================================================================
# FINAL WORKING MODEL WITH ALL DATASETS (NO ERRORS)
# ==================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ================================================================
# STEP 1 — Load datasets
# ================================================================
pest = pd.read_csv("pesticides.csv")
rain = pd.read_csv("rainfall.csv")
honey = pd.read_csv("US_honey_dataset_updated.csv")
yield_df = pd.read_csv("yield_df.csv")
poll100 = pd.read_csv("pollinators_dataset_100rows.csv")

# ================================================================
# STEP 2 — Clean columns
# ================================================================
def clean_cols(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

pest = clean_cols(pest)
rain = clean_cols(rain)
honey = clean_cols(honey)
yield_df = clean_cols(yield_df)
poll100 = clean_cols(poll100)

# ================================================================
# STEP 3 — Rename pollinators dataset for safe merging
# ================================================================
poll100 = poll100.rename(columns={
    "country": "area",
    "pollinator_count": "pollinator_colonies",
    "rainfall": "pollinator_rainfall",
    "temperature": "pollinator_temperature",
    "fertilizer_use": "fertilizer_use",
    "crop_yield": "pollinator_yield"
})

# ================================================================
# STEP 4 — Merge datasets
# ================================================================
merged = yield_df.copy()

# rainfall
merged = merged.merge(
    rain[["area", "year", "average_rain_fall_mm_per_year"]],
    on=["area", "year"], how="left"
)

# pesticides
merged = merged.merge(
    pest[["area", "year", "pesticides_tonnes"]],
    on=["area", "year"], how="left"
)

# pollinators
merged = merged.merge(
    poll100[["area", "year", "pollinator_colonies", "pollinator_rainfall",
             "pollinator_temperature", "fertilizer_use"]],
    on=["area", "year"], how="left"
)

# USA honey dataset (state → country)
honey_agg = honey.groupby("year")["pollinator_colonies"].sum().reset_index()
honey_agg["area"] = "USA"

merged = merged.merge(
    honey_agg, on=["area", "year"], how="left", suffixes=("", "_usa")
)

# Combine USA pollinator data if needed
merged["pollinator_colonies"] = merged["pollinator_colonies"].fillna(
    merged["pollinator_colonies_usa"]
)

# ================================================================
# STEP 5 — Fill missing numeric values
# ================================================================
merged = merged.fillna(merged.median(numeric_only=True))

# ================================================================
# STEP 6 — Feature selection using REAL column names
# ================================================================
features = [
    "average_rain_fall_mm_per_year_x",
    "pesticides_tonnes_x",
    "avg_temp",
    "pollinator_colonies",
    "pollinator_temperature",
    "fertilizer_use"
]

target = "yield_value"

X = merged[features]
y = merged[target]

# ================================================================
# STEP 7 — Train model
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ================================================================
# STEP 8 — Prediction function
# ================================================================
def predict_yield(country, year):
    row = merged[(merged["area"].str.lower() == country.lower()) &
                 (merged["year"] == year)]

    if row.empty:
        print("\n❌ No data found for this country/year.")
        return

    row_x = row[features].iloc[0].to_frame().T
    pred = model.predict(row_x)[0]

    print(f"\n🌾 Predicted crop yield for {country} in {year}: {pred:.2f}")
    print("Features used:", row_x.to_dict(orient="records")[0])

# ================================================================
# STEP 9 — Menu
# ================================================================
while True:
    print("\n======== MENU ========")
    print("1. Predict crop yield")
    print("2. Exit")

    ch = input("Enter choice: ")

    if ch == "1":
        country = input("Enter country: ")
        year = int(input("Enter year: "))
        predict_yield(country, year)
    else:
        break
