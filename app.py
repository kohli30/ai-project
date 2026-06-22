import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

st.set_page_config(layout="wide", page_title="Pollinators → Crop Yield Dashboard")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_csv_safe(path):
    return pd.read_csv(path)

def clean_cols(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

@st.cache_data
def load_and_merge_all():
    # file names (must exist in same folder as app.py)
    files = {
        "pesticides": "pesticides.csv",
        "rainfall": "rainfall.csv",
        "honey": "US_honey_dataset_updated.csv",
        "yield": "yield_df.csv",
        "poll100": "pollinators_dataset_100rows.csv"
    }

    # load
    pest = load_csv_safe(files["pesticides"])
    rain = load_csv_safe(files["rainfall"])
    honey = load_csv_safe(files["honey"])
    yield_df = load_csv_safe(files["yield"])
    poll100 = load_csv_safe(files["poll100"])

    # clean columns
    pest = clean_cols(pest)
    rain = clean_cols(rain)
    honey = clean_cols(honey)
    yield_df = clean_cols(yield_df)
    poll100 = clean_cols(poll100)

    # harmonize poll100
    rename_map = {}
    if "country" in poll100.columns:
        rename_map["country"] = "area"
    if "pollinator_count" in poll100.columns:
        rename_map["pollinator_count"] = "pollinator_colonies"
    if "rainfall" in poll100.columns:
        rename_map["rainfall"] = "pollinator_rainfall"
    if "temperature" in poll100.columns:
        rename_map["temperature"] = "pollinator_temperature"
    if "fertilizer_use" in poll100.columns:
        rename_map["fertilizer_use"] = "fertilizer_use"
    poll100 = poll100.rename(columns=rename_map)

    # Merge order: base = yield_df
    merged = yield_df.copy()

    # merge rainfall (area+year if possible else year)
    if {"area","year","average_rain_fall_mm_per_year"}.issubset(rain.columns):
        merged = merged.merge(rain[["area","year","average_rain_fall_mm_per_year"]], on=["area","year"], how="left")
    else:
        if "year" in rain.columns and "average_rain_fall_mm_per_year" in rain.columns:
            merged = merged.merge(rain[["year","average_rain_fall_mm_per_year"]], on="year", how="left")

    # merge pesticides
    if {"area","year","pesticides_tonnes"}.issubset(pest.columns):
        merged = merged.merge(pest[["area","year","pesticides_tonnes"]], on=["area","year"], how="left")
    else:
        if "year" in pest.columns and "pesticides_tonnes" in pest.columns:
            merged = merged.merge(pest[["year","pesticides_tonnes"]], on="year", how="left")

    # merge poll100
    poll_cols = [c for c in ["area","year","pollinator_colonies","pollinator_rainfall","pollinator_temperature","fertilizer_use"] if c in poll100.columns]
    if poll_cols:
        merged = merged.merge(poll100[poll_cols], on=["area","year"], how="left")

    # honey: aggregate states to USA if present
    if {"year","pollinator_colonies","state"}.issubset(honey.columns):
        honey_agg = honey.groupby("year")["pollinator_colonies"].sum().reset_index()
        honey_agg["area"] = "usa"
        merged = merged.merge(honey_agg[["area","year","pollinator_colonies"]], on=["area","year"], how="left", suffixes=("","_usa"))
        # fill pollinator_colonies with USA aggregated where missing and area == 'usa'
        merged["pollinator_colonies"] = merged.get("pollinator_colonies").fillna(merged.get("pollinator_colonies_usa"))

    # final: normalize numeric and fill missing with medians
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.fillna(merged[numeric_cols].median())

    # return merged & original frames for reference
    return merged, {"pest": pest, "rain": rain, "honey": honey, "yield": yield_df, "poll100": poll100}

@st.cache_data
def train_models(merged_df, features, target):
    X = merged_df[features]
    y = merged_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

    # metrics
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    metrics = {
        "lr": {
            "rmse": np.sqrt(mean_squared_error(y_test, lr_pred)),
            "r2": r2_score(y_test, lr_pred)
        },
        "rf": {
            "rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
            "r2": r2_score(y_test, rf_pred)
        }
    }
    return {"lr": lr, "rf": rf, "metrics": metrics}

# ---------------------------
# App layout
# ---------------------------
st.title("Pollinators → Crop Yield (AI) — Dashboard")
st.markdown("Interactive dashboard: merge datasets, train models, and predict crop yield using pollinator and environmental features.")

merged, sources = load_and_merge_all()

st.sidebar.header("Model / Data Controls")
st.sidebar.write("Uploaded files will be loaded from local folder. Make sure CSVs exist in same folder as app.py.")

# show merged preview
if st.sidebar.checkbox("Show merged preview (first 10 rows)", True):
    st.dataframe(merged.head(10))

# determine available feature candidates
st.sidebar.markdown("### Choose features (default recommended)")
available_cols = merged.columns.tolist()
# recommended mapping based on earlier conversation
recommended = []
if "average_rain_fall_mm_per_year" in available_cols:
    recommended.append("average_rain_fall_mm_per_year")
elif "average_rain_fall_mm_per_year_x" in available_cols:
    recommended.append("average_rain_fall_mm_per_year_x")

if "pesticides_tonnes" in available_cols:
    recommended.append("pesticides_tonnes")
elif "pesticides_tonnes_x" in available_cols:
    recommended.append("pesticides_tonnes_x")

if "avg_temp" in available_cols:
    recommended.append("avg_temp")
if "pollinator_colonies" in available_cols:
    recommended.append("pollinator_colonies")
if "pollinator_temperature" in available_cols:
    recommended.append("pollinator_temperature")
if "fertilizer_use" in available_cols:
    recommended.append("fertilizer_use")

target_col = "yield_value" if "yield_value" in available_cols else ("crop_yield" if "crop_yield" in available_cols else None)
if target_col is None:
    st.error("No recognized target column found (yield_value or crop_yield). Check your CSVs.")
    st.stop()

st.sidebar.write("Detected target column: ", target_col)

# show recommended features and let user modify
st.sidebar.write("Recommended features (you can change):")
features_selected = st.sidebar.multiselect("Select features", options=available_cols, default=recommended)
if not features_selected:
    st.sidebar.error("Select at least one feature.")
    st.stop()

# training controls
algo = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest"], index=1)
do_train = st.sidebar.button("Train models (use cached when same data)")

# Train automatically (cached)
with st.spinner("Training models..."):
    models_info = train_models(merged, features_selected, target_col)
    st.success("Models trained.")

# show metrics
st.subheader("Model Performance")
cols = st.columns(2)
cols[0].metric("Linear Regression RMSE", f"{models_info['metrics']['lr']['rmse']:.3f}")
cols[0].metric("Linear Regression R²", f"{models_info['metrics']['lr']['r2']:.3f}")
cols[1].metric("Random Forest RMSE", f"{models_info['metrics']['rf']['rmse']:.3f}")
cols[1].metric("Random Forest R²", f"{models_info['metrics']['rf']['r2']:.3f}")

# ---------------------------
# Prediction area
# ---------------------------
st.subheader("Predict Crop Yield")

mode = st.radio("Choose input mode", ["Country + Year (auto lookup)", "Manual inputs"])

if mode == "Country + Year (auto lookup)":
    country_list = merged["area"].dropna().unique().tolist()
    country_choice = st.selectbox("Select Country", options=sorted(country_list))
    year_choice = st.selectbox("Select Year", options=sorted(merged["year"].unique().tolist()))
    if st.button("Predict (Country + Year)"):
        row = merged[(merged["area"].astype(str).str.lower() == str(country_choice).lower()) & (merged["year"] == int(year_choice))]
        if row.empty:
            st.error("No data found for this country+year combination.")
        else:
            X_input = row[features_selected].iloc[0].to_frame().T
            model_chosen = models_info['rf'] if algo == "Random Forest" else models_info['lr']
            pred = model_chosen.predict(X_input)[0]
            st.success(f"Predicted crop yield for {country_choice}, {year_choice}: {pred:.2f}")
            st.write("Features used:")
            st.json(X_input.to_dict(orient="records")[0])

else:
    st.markdown("Enter feature values:")
    manual_vals = {}
    for f in features_selected:
        manual_vals[f] = st.number_input(f, value=float(merged[f].median()))
    if st.button("Predict (Manual)"):
        X_input = pd.DataFrame([manual_vals])
        model_chosen = models_info['rf'] if algo == "Random Forest" else models_info['lr']
        pred = model_chosen.predict(X_input)[0]
        st.success(f"Predicted crop yield (manual): {pred:.2f}")
        st.write("Features used:")
        st.json(X_input.to_dict(orient="records")[0])

# ---------------------------
# Plots
# ---------------------------
st.subheader("Plots / Visuals")

# 1. Pollinator count vs Yield scatter
fig1, ax1 = plt.subplots()
ax1.scatter(merged["pollinator_colonies"], merged[target_col])
ax1.set_xlabel("Pollinator colonies")
ax1.set_ylabel("Crop yield")
ax1.set_title("Pollinator colonies vs Crop yield")
st.pyplot(fig1)

# 2. Actual vs Predicted (Random Forest)
model_rf = models_info['rf']
preds_full = model_rf.predict(merged[features_selected])
fig2, ax2 = plt.subplots()
ax2.scatter(merged[target_col], preds_full)
ax2.set_xlabel("Actual yield")
ax2.set_ylabel("Predicted yield")
ax2.set_title("Actual vs Predicted (Random Forest)")
st.pyplot(fig2)

# 3. Time-series: average pollinators by year
ts = merged.groupby("year")["pollinator_colonies"].mean().reset_index()
fig3, ax3 = plt.subplots()
ax3.plot(ts["year"], ts["pollinator_colonies"])
ax3.set_xlabel("Year")
ax3.set_ylabel("Average pollinator colonies")
ax3.set_title("Average pollinator colonies by year")
st.pyplot(fig3)

# ---------------------------
# Export prediction (optional)
# ---------------------------
st.subheader("Export / Download")
if st.button("Save merged dataset as CSV"):
    out_path = Path("merged_dataset.csv")
    merged.to_csv(out_path, index=False)
    st.success(f"Saved merged dataset to {out_path}")

st.markdown("---")
st.caption("Notes: This app assumes CSV files are in the same directory as app.py. If column names differ, the app attempts safe merging but you may need to adjust files or column names.")
