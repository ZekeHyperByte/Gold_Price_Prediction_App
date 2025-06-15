import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
# IMPORTANT: These paths are relative to your app.py file
MODEL_FILE_PATH_LR_ABS = 'linear_regression_abs_gold_price_model.joblib'
MODEL_FILE_PATH_XGB_ABS = 'xgboost_tuned_abs_gold_price_model.joblib' # Tuned absolute price model
MODEL_FILE_PATH_LR_RETURNS = 'linear_regression_returns_gold_price_model.joblib'
MODEL_FILE_PATH_XGB_RETURNS = 'xgboost_returns_tuned_gold_price_model.joblib'

FEATURES_FILE_PATH_ABS = 'model_features_abs.joblib' # Feature list for absolute price models
FEATURES_FILE_PATH_RETURNS = 'model_features_for_returns.joblib' # Feature list for returns models

GOLD_PRICES_2013_2023_CSV = 'gold prices.csv'
GOLD_PRICES_2024_CSV = 'Gold Futures Historical Data (23.01.24-22.11.24).csv'

# --- 1. Load Models and Data (Cached for Efficiency) ---
@st.cache_resource # Use st.cache_resource for models as they are loaded once
def load_all_models_and_features():
    models = {}
    features_sets = {}
    
    try:
        models['Linear Regression (Abs Price)'] = joblib.load(MODEL_FILE_PATH_LR_ABS)
        features_sets['Linear Regression (Abs Price)'] = joblib.load(FEATURES_FILE_PATH_ABS)

        try:
            models['XGBoost (Abs Price - Tuned)'] = joblib.load(MODEL_FILE_PATH_XGB_ABS)
            features_sets['XGBoost (Abs Price - Tuned)'] = joblib.load(FEATURES_FILE_PATH_ABS)
        except FileNotFoundError:
            st.warning(f"Optional model '{MODEL_FILE_PATH_XGB_ABS}' not found. Skipping its loading.")

        models['Linear Regression (Returns)'] = joblib.load(MODEL_FILE_PATH_LR_RETURNS)
        features_sets['Linear Regression (Returns)'] = joblib.load(FEATURES_FILE_PATH_RETURNS)

        models['XGBoost (Returns - Tuned)'] = joblib.load(MODEL_FILE_PATH_XGB_RETURNS)
        features_sets['XGBoost (Returns - Tuned)'] = joblib.load(FEATURES_FILE_PATH_RETURNS)
        
        st.success("All individual models and feature lists loaded successfully!")
        return models, features_sets
    except FileNotFoundError as e:
        st.error(f"Error loading required model/feature files: {e}. Please ensure all .joblib files are in the app directory.")
        st.stop() # Stop the app if crucial files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model/features: {e}")
        st.stop()
        
@st.cache_data # Use st.cache_data for data loading/preprocessing
def load_and_prepare_data():
    try:
        df_2013_2023 = pd.read_csv(GOLD_PRICES_2013_2023_CSV)
        df_2024 = pd.read_csv(GOLD_PRICES_2024_CSV)

        df_2013_2023.columns = [col.replace(' ', '_').lower() for col in df_2013_2023.columns]
        df_2024.columns = [col.replace(' ', '_').lower() for col in df_2024.columns]

        if 'close/last' in df_2013_2023.columns:
            df_2013_2023.rename(columns={'close/last': 'close'}, inplace=True)
        if 'price' in df_2024.columns:
            df_2024.rename(columns={'price': 'close'}, inplace=True)
        if 'vol.' in df_2024.columns:
            df_2024.rename(columns={'vol.': 'volume'}, inplace=True)

        df_2013_2023['date'] = pd.to_datetime(df_2013_2023['date'])
        df_2024['date'] = pd.to_datetime(df_2024['date'])

        numeric_cols_2024 = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols_2024:
            if col in df_2024.columns:
                df_2024[col] = pd.to_numeric(df_2024[col].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')

        df_2013_2023 = df_2013_2023.sort_values(by='date')
        df_2024 = df_2024.sort_values(by='date')

        common_cols_ohcl = ['date', 'open', 'high', 'low', 'close']
        combined_df = pd.concat([df_2013_2023[common_cols_ohcl], df_2024[common_cols_ohcl]]).sort_values(by='date').reset_index(drop=True)
        combined_df = combined_df.set_index('date')
        combined_df = combined_df.asfreq('B')
        combined_df = combined_df.ffill().bfill()

        combined_df['close_lag1'] = combined_df['close'].shift(1)
        combined_df['open_lag1'] = combined_df['open'].shift(1)
        combined_df['close_lag2'] = combined_df['close'].shift(2)
        combined_df['open_lag2'] = combined_df['open'].shift(2)
        combined_df['daily_return'] = combined_df['close'].pct_change()
        combined_df['daily_return_lag1'] = combined_df['daily_return'].shift(1)

        combined_df['SMA_5'] = combined_df['close'].rolling(window=5).mean()
        combined_df['SMA_20'] = combined_df['close'].rolling(window=20).mean()
        combined_df['volatility_10d'] = combined_df['close'].rolling(window=10).std()

        combined_df['day_of_week'] = combined_df.index.dayofweek
        combined_df['month'] = combined_df.index.month
        combined_df['year'] = combined_df.index.year
        combined_df['day_of_year'] = combined_df.index.dayofyear
        combined_df.dropna(inplace=True)

        return combined_df
    except FileNotFoundError:
        st.error(f"Error: Data CSV files not found. Please ensure '{GOLD_PRICES_2013_2023_CSV}' and '{GOLD_PRICES_2024_CSV}' are in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()

# --- 2. Feature Engineering Function for Prediction ---
def prepare_features_for_prediction(
    prediction_date,
    last_actual_ohlc_date,
    last_actual_open,
    last_actual_high,
    last_actual_low,
    last_actual_close,
    historical_df,
    features_list
):
    try:
        prediction_date_ts = pd.to_datetime(prediction_date)
        last_actual_ohlc_date_ts = pd.to_datetime(last_actual_ohlc_date)

        new_input_df = pd.DataFrame(index=[prediction_date_ts])
        
        if prediction_date_ts in historical_df.index:
            new_input_df['open'] = historical_df.loc[prediction_date_ts, 'open']
            new_input_df['high'] = historical_df.loc[prediction_date_ts, 'high']
            new_input_df['low'] = historical_df.loc[prediction_date_ts, 'low']
        else:
            new_input_df['open'] = last_actual_close
            new_input_df['high'] = last_actual_close * 1.005
            new_input_df['low'] = last_actual_close * 0.995

        new_input_df['close_lag1'] = last_actual_close
        new_input_df['open_lag1'] = last_actual_open
        
        two_days_ago_date_ts = last_actual_ohlc_date_ts - pd.Timedelta(days=1)
        while two_days_ago_date_ts not in historical_df.index and two_days_ago_date_ts >= historical_df.index.min():
            two_days_ago_date_ts -= pd.Timedelta(days=1)

        if two_days_ago_date_ts in historical_df.index:
            new_input_df['close_lag2'] = historical_df.loc[two_days_ago_date_ts, 'close_lag1']
            new_input_df['open_lag2'] = historical_df.loc[two_days_ago_date_ts, 'open_lag1']
            new_input_df['daily_return_lag1'] = historical_df.loc[two_days_ago_date_ts, 'daily_return']
        else:
            new_input_df['close_lag2'] = np.nan
            new_input_df['open_lag2'] = np.nan
            new_input_df['daily_return_lag1'] = np.nan

        end_date_for_lookbacks = last_actual_ohlc_date_ts
        start_date_for_lookbacks = end_date_for_lookbacks - pd.Timedelta(days=40)
        
        recent_history_for_lookbacks = historical_df.loc[
            (historical_df.index >= start_date_for_lookbacks) & (historical_df.index <= end_date_for_lookbacks),
            'close'
        ].copy()
        
        sma_5 = recent_history_for_lookbacks.rolling(window=5).mean().iloc[-1] if len(recent_history_for_lookbacks) >= 5 else np.nan
        sma_20 = recent_history_for_lookbacks.rolling(window=20).mean().iloc[-1] if len(recent_history_for_lookbacks) >= 20 else np.nan
        volatility_10d = recent_history_for_lookbacks.rolling(window=10).std().iloc[-1] if len(recent_history_for_lookbacks) >= 10 else np.nan

        new_input_df['SMA_5'] = sma_5
        new_input_df['SMA_20'] = sma_20
        new_input_df['volatility_10d'] = volatility_10d

        new_input_df['day_of_week'] = prediction_date_ts.dayofweek
        new_input_df['month'] = prediction_date_ts.month
        new_input_df['year'] = prediction_date_ts.year
        new_input_df['day_of_year'] = prediction_date_ts.dayofyear

        if 'daily_return' in features_list:
            close_two_days_ago_for_dr = historical_df.loc[two_days_ago_date_ts, 'close'] if two_days_ago_date_ts in historical_df.index else np.nan
            
            if not pd.isna(close_two_days_ago_for_dr) and close_two_days_ago_for_dr != 0:
                daily_return_val = (last_actual_close - close_two_days_ago_for_dr) / close_two_days_ago_for_dr
            else:
                daily_return_val = np.nan

            new_input_df['daily_return'] = daily_return_val

        X_predict = new_input_df[features_list].copy()
        
        if X_predict.isnull().any().any():
            st.warning("Warning: NaN values detected in prediction input after feature preparation. Imputing with nearest valid observation.")
            X_predict = X_predict.ffill().bfill()
            if X_predict.isnull().any().any():
                st.error("Critical: NaNs still present after imputation. Cannot predict.")
                return None
        
        return X_predict

    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return None

# --- Main Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Gold Price Predictor")

st.title("💰 Gold Price Predictor: Unraveling Gold's Future Trends")
st.markdown("""
Welcome to the Gold Price Predictor! This interactive tool uses advanced Machine Learning models
to forecast gold prices, helping you understand potential future trends or analyze historical model performance.
""")

st.info("""
**How it works:**
Our models learn from historical gold price data (Open, High, Low, Close, and Volume).
They leverage various features like past prices (lagged values), moving averages, and time-based factors (day of week, month, year)
to make predictions.
""")

# Load models, features, and data
models, features_sets = load_all_models_and_features()
combined_df = load_and_prepare_data()

latest_data_date = combined_df.index.max().date()
min_hist_date = combined_df.index.min().date()

# Sidebar for common info or settings
st.sidebar.header("About This App")
st.sidebar.info(
    "This application showcases gold price prediction using various machine learning models trained on historical data. "
    "Use it to explore future price forecasts or analyze how models performed on past data."
)
st.sidebar.markdown(f"**Data Coverage:** {min_hist_date.strftime('%Y-%m-%d')} to {latest_data_date.strftime('%Y-%m-%d')}")

# Add "Ensemble Model" to the list of available models for selection
available_models_for_selection = list(models.keys())
available_models_for_selection.insert(0, "Ensemble (LR Abs + XGBoost Returns)")

selected_model_name = st.sidebar.selectbox(
    "Select Prediction Model:",
    available_models_for_selection,
    index=0 # Default to Ensemble Model
)

# Determine the actual model(s) and feature set based on selection
selected_model_obj = None
selected_features = None
is_returns_model = False # Flag to determine if prediction is a return that needs conversion

if selected_model_name == "Ensemble (LR Abs + XGBoost Returns)":
    lr_abs_model = models.get('Linear Regression (Abs Price)')
    xgb_returns_tuned_model = models.get('XGBoost (Returns - Tuned)')
    
    if lr_abs_model is None or xgb_returns_tuned_model is None:
        st.error("Ensemble model components not loaded. Please check model files.")
        st.stop()
    
    selected_model_obj = {'lr_abs': lr_abs_model, 'xgb_returns': xgb_returns_tuned_model}
    
    selected_features_lr_abs = features_sets['Linear Regression (Abs Price)']
    selected_features_xgb_returns = features_sets['XGBoost (Returns - Tuned)']

    is_returns_model = False

elif "Returns" in selected_model_name:
    selected_model_obj = models[selected_model_name]
    selected_features = features_sets[selected_model_name]
    is_returns_model = True
else: # Absolute Price Models
    selected_model_obj = models[selected_model_name]
    selected_features = features_sets[selected_model_name]
    is_returns_model = False

st.sidebar.markdown(f"**Current Model:** `{selected_model_name}`")
st.sidebar.markdown("""
**Model Types Explained:**
* **Abs Price Models:** Predict the gold price directly.
* **Returns Models:** Predict the percentage change in gold price, then convert that change into an absolute price. (Often performs better for tree-based models like XGBoost in time series).
* **Ensemble Model:** Combines the strengths of multiple individual models for a potentially more robust prediction.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Disclaimer:** *This tool is for educational and demonstrative purposes only. Gold price predictions are highly speculative and should not be used as financial advice or for real-world trading decisions.*")


# --- Tabs for Prediction Types and Model Performance ---
tab_live, tab_historical, tab_performance = st.tabs(["🚀 Predict Tomorrow's Price", "🕰️ Analyze Historical Price", "📈 Model Performance Overview"])

with tab_live:
    st.header("🚀 Predict Tomorrow's Gold Price")
    st.markdown("""
    Use this section to get a forecast for the **next trading day's gold price**.
    Input the most recent actual Open, High, Low, and Close prices available.
    """)

    tomorrow_date = latest_data_date + timedelta(days=1)
    while tomorrow_date.weekday() > 4:
        tomorrow_date += timedelta(days=1)

    st.subheader(f"Prediction for: {tomorrow_date.strftime('%Y-%m-%d')}")

    st.markdown(f"**Most Recent Actual Gold Price Data (as of {latest_data_date.strftime('%Y-%m-%d')}):**")
    st.info("""
    These are the trading prices of gold from the last completed trading day:
    * **Open:** The price at which gold trading opened.
    * **High:** The highest price gold reached during the day.
    * **Low:** The lowest price gold reached during the day.
    * **Close:** The final trading price of gold for the day.
    
    You can adjust these values to explore hypothetical scenarios!
    """)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        live_open = st.number_input("Open ($)", value=float(combined_df['open'].iloc[-1]), format="%.2f", key="live_open")
    with col2:
        live_high = st.number_input("High ($)", value=float(combined_df['high'].iloc[-1]), format="%.2f", key="live_high")
    with col3:
        live_low = st.number_input("Low ($)", value=float(combined_df['low'].iloc[-1]), format="%.2f", key="live_low")
    with col4:
        live_close = st.number_input("Close ($)", value=float(combined_df['close'].iloc[-1]), format="%.2f", key="live_close")

    if st.button("Predict Tomorrow's Price", key="predict_live_button"):
        if live_close <= 0:
            st.error("Close price must be positive to make a meaningful prediction.")
        else:
            with st.spinner("Calculating prediction..."):
                predicted_absolute_price_live = None # Initialize
                predicted_return_display = "N/A" # Initialize

                if selected_model_name == "Ensemble (LR Abs + XGBoost Returns)":
                    X_predict_live_lr_abs = prepare_features_for_prediction(
                        prediction_date=tomorrow_date, last_actual_ohlc_date=latest_data_date,
                        last_actual_open=live_open, last_actual_high=live_high,
                        last_actual_low=live_low, last_actual_close=live_close,
                        historical_df=combined_df, features_list=features_sets['Linear Regression (Abs Price)']
                    )
                    
                    X_predict_live_xgb_returns = prepare_features_for_prediction(
                        prediction_date=tomorrow_date, last_actual_ohlc_date=latest_data_date,
                        last_actual_open=live_open, last_actual_high=live_high,
                        last_actual_low=live_low, last_actual_close=live_close,
                        historical_df=combined_df, features_list=features_sets['XGBoost (Returns - Tuned)']
                    )

                    if X_predict_live_lr_abs is not None and X_predict_live_xgb_returns is not None:
                        lr_abs_pred = selected_model_obj['lr_abs'].predict(X_predict_live_lr_abs)[0]
                        xgb_returns_pred_val = selected_model_obj['xgb_returns'].predict(X_predict_live_xgb_returns)[0]
                        xgb_returns_abs_pred = live_close * (1 + xgb_returns_pred_val)
                        
                        predicted_absolute_price_live = (lr_abs_pred + xgb_returns_abs_pred) / 2
                        predicted_return_display = "Combined (N/A)"
                    else:
                        st.error("Could not prepare features for Ensemble prediction. Please check inputs.")
                        predicted_absolute_price_live = None
                else: # Individual Models
                    X_predict_live = prepare_features_for_prediction(
                        prediction_date=tomorrow_date,
                        last_actual_ohlc_date=latest_data_date,
                        last_actual_open=live_open,
                        last_actual_high=live_high,
                        last_actual_low=live_low,
                        last_actual_close=live_close,
                        historical_df=combined_df,
                        features_list=selected_features
                    )
                    
                    if X_predict_live is not None:
                        predicted_value = selected_model_obj.predict(X_predict_live)[0]

                        if is_returns_model:
                            predicted_absolute_price_live = live_close * (1 + predicted_value)
                            predicted_return_display = f"{predicted_value*100:.4f}%"
                        else:
                            predicted_absolute_price_live = predicted_value
                            predicted_return_display = "N/A (Abs Price Model)"
                
                # --- Display Results ---
                if predicted_absolute_price_live is not None:
                    st.subheader("Prediction Results:")
                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        st.metric(label="Predicted Daily Return", value=predicted_return_display)
                    with col_pred2:
                        st.metric(label="Predicted Gold Price", value=f"${predicted_absolute_price_live:.2f}",
                                  delta=f"${predicted_absolute_price_live - live_close:.2f} vs last Close")
                    st.markdown(f"*(Based on {latest_data_date.strftime('%Y-%m-%d')} Close: ${live_close:.2f})*")

                    # --- Visualization for Live Prediction ---
                    st.subheader("Visualizing Tomorrow's Forecast:")
                    plot_start_date = latest_data_date - timedelta(days=60)
                    if plot_start_date < combined_df.index.min().date():
                        plot_start_date = combined_df.index.min().date()

                    plot_df = combined_df.loc[plot_start_date:latest_data_date].copy()
                    
                    viz_df = plot_df[['close']].copy()
                    viz_df['Type'] = 'Actual'
                    
                    prediction_point_df = pd.DataFrame({'close': [predicted_absolute_price_live], 'Type': 'Predicted'},
                                                    index=[pd.to_datetime(tomorrow_date)])
                    viz_df = pd.concat([viz_df, prediction_point_df])

                    last_actual_point_for_plot = plot_df['close'].iloc[-1]
                    last_actual_point_date_for_plot = plot_df.index[-1]
                    
                    connector_df = pd.DataFrame({
                        'close': [last_actual_point_for_plot, predicted_absolute_price_live],
                        'Type': 'Prediction Path'
                    }, index=[last_actual_point_date_for_plot, pd.to_datetime(tomorrow_date)])

                    final_viz_df = pd.concat([viz_df, connector_df])
                    
                    fig_live = px.line(final_viz_df, x=final_viz_df.index, y='close', color='Type',
                                       title='Actual Gold Price and Tomorrow\'s Forecast',
                                       labels={'close': 'Gold Price ($)', 'index': 'Date', 'color': 'Data Type'},
                                       color_discrete_map={'Actual': 'blue', 'Predicted': 'red', 'Prediction Path': 'red'})

                    fig_live.update_layout(hovermode="x unified", showlegend=True)
                    st.plotly_chart(fig_live, use_container_width=True)
                    st.markdown("""
                    This chart illustrates the recent historical gold price movements (blue line)
                    and your selected model's forecast for tomorrow (red dot, connected by a dashed red line).
                    """)


                else:
                    st.error("Could not prepare features for prediction. Please check inputs and ensure enough historical data is available.")


with tab_historical:
    st.header("🕰️ Analyze Historical Price Prediction")
    st.markdown("""
    Select a historical date from the model's test set range to see how accurately the model would have predicted its price.
    The model will use data *prior* to your selected date to make its prediction, and then show you the actual price for comparison.
    """)

    test_set_start_date = pd.to_datetime('2024-08-02').date()
    test_set_end_date = latest_data_date

    historical_prediction_date = st.date_input(
        "Choose a historical trading day (from 2024-08-02 onwards):",
        value=test_set_start_date,
        min_value=test_set_start_date,
        max_value=test_set_end_date,
        key="hist_date_input"
    )
    
    historical_prediction_dt = pd.to_datetime(historical_prediction_date)

    st.markdown("---")

    if st.button("✨ Predict & Analyze Historical Price", key="predict_historical_button"):
        if historical_prediction_dt not in combined_df.index:
            st.warning(f"⚠️ No trading data available for {historical_prediction_date.strftime('%Y-%m-%d')}. Please select an actual trading day.")
        else:
            with st.spinner(f"Calculating prediction for {historical_prediction_date.strftime('%Y-%m-%d')}..."):
                predicted_absolute_price_hist = None # Initialize

                try:
                    actual_current_day_ohlc = combined_df.loc[historical_prediction_dt]
                    
                    last_actual_ohlc_date = historical_prediction_dt - pd.Timedelta(days=1)
                    while last_actual_ohlc_date not in combined_df.index and last_actual_ohlc_date >= combined_df.index.min():
                        last_actual_ohlc_date -= pd.Timedelta(days=1)
                    
                    if last_actual_ohlc_date < combined_df.index.min() or \
                       (last_actual_ohlc_date - pd.Timedelta(days=30)) < combined_df.index.min():
                        st.error(f"❌ Not enough historical data before {historical_prediction_date.strftime('%Y-%m-%d')} to make a prediction. Please select a later date.")
                        st.stop()

                    last_actual_open = combined_df.loc[last_actual_ohlc_date, 'open']
                    last_actual_high = combined_df.loc[last_actual_ohlc_date, 'high']
                    last_actual_low = combined_df.loc[last_actual_ohlc_date, 'low']
                    last_actual_close = combined_df.loc[last_actual_ohlc_date, 'close']

                except KeyError as ke:
                    st.error(f"❌ Data not found for {historical_prediction_date.strftime('%Y-%m-%d')} or its preceding days. Key: {ke}. Is this date within the loaded data range?")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Error fetching historical data: {e}")
                    st.stop()

                if selected_model_name == "Ensemble (LR Abs + XGBoost Returns)":
                    X_predict_hist_lr_abs = prepare_features_for_prediction(
                        prediction_date=historical_prediction_dt, last_actual_ohlc_date=last_actual_ohlc_date,
                        last_actual_open=last_actual_open, last_actual_high=last_actual_high,
                        last_actual_low=last_actual_low, last_actual_close=last_actual_close,
                        historical_df=combined_df, features_list=features_sets['Linear Regression (Abs Price)']
                    )
                    X_predict_hist_xgb_returns = prepare_features_for_prediction(
                        prediction_date=historical_prediction_dt, last_actual_ohlc_date=last_actual_ohlc_date,
                        last_actual_open=last_actual_open, last_actual_high=last_actual_high,
                        last_actual_low=last_actual_low, last_actual_close=last_actual_close,
                        historical_df=combined_df, features_list=features_sets['XGBoost (Returns - Tuned)']
                    )

                    if X_predict_hist_lr_abs is not None and X_predict_hist_xgb_returns is not None:
                        lr_abs_pred_hist = selected_model_obj['lr_abs'].predict(X_predict_hist_lr_abs)[0]
                        xgb_returns_pred_hist_val = selected_model_obj['xgb_returns'].predict(X_predict_hist_xgb_returns)[0]
                        xgb_returns_abs_pred_hist = last_actual_close * (1 + xgb_returns_pred_hist_val)
                        
                        predicted_absolute_price_hist = (lr_abs_pred_hist + xgb_returns_abs_pred_hist) / 2
                        predicted_return_display_hist = "Combined (N/A)"
                    else:
                        st.error("Could not prepare features for Ensemble historical prediction.")
                        predicted_absolute_price_hist = None
                else: # Individual Models
                    X_predict_historical = prepare_features_for_prediction(
                        prediction_date=historical_prediction_dt,
                        last_actual_ohlc_date=last_actual_ohlc_date,
                        last_actual_open=last_actual_open,
                        last_actual_high=last_actual_high,
                        last_actual_low=last_actual_low,
                        last_actual_close=last_actual_close,
                        historical_df=combined_df,
                        features_list=selected_features
                    )

                    if X_predict_historical is not None:
                        predicted_value_hist = selected_model_obj.predict(X_predict_historical)[0]

                        if is_returns_model:
                            predicted_absolute_price_hist = last_actual_close * (1 + predicted_value_hist)
                            predicted_return_display_hist = f"{predicted_value_hist*100:.4f}%"
                        else:
                            predicted_absolute_price_hist = predicted_value_hist
                            predicted_return_display_hist = "N/A (Abs Price Model)"
                
                # --- Display Results ---
                if predicted_absolute_price_hist is not None:
                    st.subheader(f"📊 Prediction Results for {historical_prediction_date.strftime('%Y-%m-%d')}:")
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    if selected_model_name == "Ensemble (LR Abs + XGBoost Returns)":
                        with col_res1:
                            st.metric(label="Predicted Gold Price", value=f"${predicted_absolute_price_hist:.2f}")
                        with col_res2:
                            st.empty() # No direct return for ensemble
                    elif is_returns_model:
                        with col_res1:
                            st.metric(label="Predicted Daily Return", value=predicted_return_display_hist)
                        with col_res2:
                            st.metric(label="Predicted Gold Price", value=f"${predicted_absolute_price_hist:.2f}")
                    else: # Absolute Price Model
                        with col_res1:
                            st.metric(label="Predicted Gold Price", value=f"${predicted_absolute_price_hist:.2f}")
                        with col_res2:
                            st.empty()
                    
                    actual_gold_price_hist = actual_current_day_ohlc['close']
                    error_abs = predicted_absolute_price_hist - actual_gold_price_hist
                    error_percent = (error_abs / actual_gold_price_hist) * 100 if actual_gold_price_hist != 0 else 0

                    with col_res3:
                        st.metric(label="Actual Gold Price", value=f"${actual_gold_price_hist:.2f}",
                                  delta=f"${error_abs:.2f} ({error_percent:.2f}%) Error")
                    
                    st.info(f"""
                    * **Prediction based on data available up to:** **{last_actual_ohlc_date.strftime('%Y-%m-%d')}**
                    * **Closing price on {last_actual_ohlc_date.strftime('%Y-%m-%d')}:** **${last_actual_close:.2f}**
                    """)
                    
                    st.subheader("📈 Visualizing Historical Performance:")
                    if abs(error_abs) <= 5:
                        st.success("✨ **Excellent Prediction!** The model's forecast was very close to the actual price. ✨")
                    elif abs(error_abs) <= 20:
                        st.info("👍 **Good Prediction.** The model's forecast was reasonably accurate.")
                    else:
                        st.warning("⚠️ **Considerable Deviation.** The model's prediction showed a larger difference from the actual price.")

                    plot_range_days_before = 30
                    plot_range_days_after = 5
                    
                    plot_start_date_hist = historical_prediction_dt - pd.Timedelta(days=plot_range_days_before)
                    plot_end_date_hist = historical_prediction_dt + pd.Timedelta(days=plot_range_days_after)
                    
                    plot_start_dt_hist = pd.to_datetime(plot_start_date_hist)
                    plot_end_dt_hist = pd.to_datetime(plot_end_date_hist)
                    
                    if plot_start_dt_hist < combined_df.index.min():
                        plot_start_dt_hist = combined_df.index.min()
                    if plot_end_dt_hist > combined_df.index.max():
                        plot_end_dt_hist = combined_df.index.max()

                    plot_df_hist = combined_df.loc[plot_start_dt_hist:plot_end_dt_hist].copy()

                    viz_df_hist = plot_df_hist[['close']].copy()
                    viz_df_hist.rename(columns={'close': 'Actual'}, inplace=True)
                    
                    viz_df_hist.loc[historical_prediction_dt, 'Predicted'] = predicted_absolute_price_hist

                    last_actual_before_pred_series = viz_df_hist.loc[viz_df_hist.index < historical_prediction_dt, 'Actual']
                    if not last_actual_before_pred_series.empty:
                        last_actual_before_pred_hist = last_actual_before_pred_series.iloc[-1]
                        last_actual_before_pred_date_hist = last_actual_before_pred_series.index[-1]
                    else:
                        last_actual_before_pred_hist = np.nan
                        last_actual_before_pred_date_hist = np.nan


                    fig_hist = px.line(viz_df_hist, x=viz_df_hist.index, y=['Actual', 'Predicted'],
                                       title=f'Actual vs. Predicted Gold Price for {historical_prediction_date.strftime("%Y-%m-%d")}',
                                       labels={'value': 'Gold Price ($)', 'index': 'Date', 'variable': 'Data Type'},
                                       color_discrete_map={'Actual': 'blue', 'Predicted': 'orange'},
                                       line_dash_map={'Predicted': 'dot'})


                    if not pd.isna(last_actual_before_pred_hist):
                        fig_hist.add_trace(go.Scatter(x=[last_actual_before_pred_date_hist, historical_prediction_dt],
                                                     y=[last_actual_before_pred_hist, predicted_absolute_price_hist],
                                                     mode='lines', name='Prediction Link',
                                                     line=dict(dash='dot', color='red', width=3),
                                                     showlegend=False))
                    
                    fig_hist.add_trace(go.Scatter(x=[historical_prediction_dt], y=[actual_gold_price_hist],
                                                 mode='markers', name='Actual Price (on Date)',
                                                 marker=dict(size=10, color='green', symbol='circle'),
                                                 showlegend=True))


                    fig_hist.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    st.markdown("""
                    This chart shows the historical gold price (blue line) and the model's specific prediction for the selected date (orange dot).
                    The green circle indicates the *actual* gold price on that date for easy comparison.
                    The red dashed line connects the last actual price to the predicted price.
                    """)


                else:
                    st.error("Could not prepare features for historical prediction. Please check selected date.")
with tab_performance:
    st.header("📈 Model Performance Overview")
    st.markdown("""
    This section presents a detailed comparison of all trained machine learning models based on their performance
    on the unseen **test dataset**. Lower MAE/RMSE indicate better accuracy, and higher R-squared (closer to 1.0)
    is better. A negative R-squared means the model is worse than simply predicting the mean.
    """)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    split_date_dt = pd.to_datetime('2024-08-01')

    test_df = combined_df[combined_df.index > split_date_dt]

    features_abs_model_names = features_sets['Linear Regression (Abs Price)']
    target_abs_col = 'close'
    X_test_abs = test_df[features_abs_model_names].copy()
    y_test_abs = test_df[target_abs_col].copy()

    features_returns_model_names = features_sets['XGBoost (Returns - Tuned)']
    target_returns_col = 'daily_return'
    X_test_returns = test_df[features_returns_model_names].copy()
    y_test_returns = test_df[target_returns_col].copy()

    performance_results = []
    plot_data_comparison = pd.DataFrame({'Actual': y_test_abs})
    prediction_category_counts = []

    st.subheader("Performance Metrics Table:")
    st.info("""
    * **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values. Lower is better.
    * **RMSE (Root Mean Squared Error):** Square root of the average squared difference. Penalizes larger errors more. Lower is better.
    * **R-squared:** Proportion of variance in actual values predictable from features. Closer to 1.0 is better. Negative R-squared means the model is worse than simply predicting the mean.
    """)

    # --- Generate predictions and metrics for all individual models ---
    for model_name, model_obj in models.items():
        if "Abs Price" in model_name:
            X_test_current = X_test_abs
            y_test_current_abs = y_test_abs
            predictions = model_obj.predict(X_test_current)
            predicted_abs_price_for_plot = pd.Series(predictions, index=X_test_current.index)
        else: # Returns model
            X_test_current = X_test_returns
            y_test_current_returns = y_test_returns
            predictions = model_obj.predict(X_test_current)
            predicted_abs_price_for_plot = X_test_current['close_lag1'] * (1 + pd.Series(predictions, index=X_test_current.index))
        
        mae = mean_absolute_error(y_test_abs, predicted_abs_price_for_plot)
        rmse = np.sqrt(mean_squared_error(y_test_abs, predicted_abs_price_for_plot))
        r2 = r2_score(y_test_abs, predicted_abs_price_for_plot)
        
        performance_results.append({
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R-squared': r2
        })
        
        plot_data_comparison[model_name] = predicted_abs_price_for_plot

        # --- Categorize Predictions and Count ---
        abs_errors = np.abs(predicted_abs_price_for_plot - y_test_abs)
        excellent_count = (abs_errors <= 5).sum()
        good_count = ((abs_errors > 5) & (abs_errors <= 20)).sum()
        terrible_count = (abs_errors > 20).sum()
        total_predictions = len(abs_errors)

        prediction_category_counts.append({
            'Model': model_name,
            'Excellent (<=$5)': excellent_count,
            'Good (>$5 to <=$20)': good_count,
            'Considerable Deviation (>$20)': terrible_count,
            'Total': total_predictions
        })

    # --- NEW: Calculate and add Ensemble Model Performance ---
    st.subheader("Ensemble Model Performance:")
    st.info("The Ensemble Model combines predictions from Linear Regression (Abs Price) and XGBoost (Returns - Tuned) by averaging their results.")

    lr_abs_model = models.get('Linear Regression (Abs Price)')
    xgb_returns_tuned_model = models.get('XGBoost (Returns - Tuned)')

    if lr_abs_model and xgb_returns_tuned_model:
        # Get LR Abs predictions
        lr_abs_test_predictions = lr_abs_model.predict(X_test_abs)

        # Get XGBoost Returns Tuned predictions (converted to absolute price)
        xgb_returns_tuned_test_predictions = xgb_returns_tuned_model.predict(X_test_returns)
        xgb_returns_tuned_abs_test_predictions = X_test_returns['close_lag1'] * (1 + pd.Series(xgb_returns_tuned_test_predictions, index=X_test_returns.index))

        # Calculate ensemble predictions
        ensemble_test_predictions = (lr_abs_test_predictions + xgb_returns_tuned_abs_test_predictions) / 2

        # Calculate metrics for ensemble
        ensemble_mae = mean_absolute_error(y_test_abs, ensemble_test_predictions)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test_abs, ensemble_test_predictions))
        ensemble_r2 = r2_score(y_test_abs, ensemble_test_predictions)

        performance_results.append({
            'Model': 'Ensemble (LR Abs + XGBoost Returns)',
            'MAE': ensemble_mae,
            'RMSE': ensemble_rmse,
            'R-squared': ensemble_r2
        })
        
        # Add ensemble predictions to plot_data_comparison
        plot_data_comparison['Ensemble (LR Abs + XGBoost Returns)'] = ensemble_test_predictions

        # Add ensemble to category counts
        ensemble_abs_errors = np.abs(ensemble_test_predictions - y_test_abs)
        ensemble_excellent = (ensemble_abs_errors <= 5).sum()
        ensemble_good = ((ensemble_abs_errors > 5) & (ensemble_abs_errors <= 20)).sum()
        ensemble_terrible = (ensemble_abs_errors > 20).sum()

        prediction_category_counts.append({
            'Model': 'Ensemble (LR Abs + XGBoost Returns)',
            'Excellent (<=$5)': ensemble_excellent,
            'Good (>$5 to <=$20)': ensemble_good,
            'Considerable Deviation (>$20)': ensemble_terrible,
            'Total': len(ensemble_abs_errors)
        })
    else:
        st.warning("Ensemble model components not fully loaded, cannot display its performance.")

    performance_df_display = pd.DataFrame(performance_results)
    st.dataframe(performance_df_display.round(4), use_container_width=True)

    st.subheader("Prediction Accuracy Categories (on Test Set):")
    st.info("""
    This table breaks down how many predictions for each model fall into different accuracy categories:
    * **Excellent:** Absolute error is $5 or less.
    * **Good:** Absolute error is between $5 and $20.
    * **Considerable Deviation:** Absolute error is greater than $20.
    """)
    category_df_display = pd.DataFrame(prediction_category_counts)
    st.dataframe(category_df_display, use_container_width=True)


    st.subheader("Performance Visualization on Test Data:")
    st.markdown("""
    This chart visually compares the actual gold prices in the test set (blue line) against the predictions
    of all loaded models. This helps to quickly identify which models track the true price movements most accurately.
    """)

    fig_comparison = go.Figure()

    fig_comparison.add_trace(go.Scatter(x=plot_data_comparison.index, y=plot_data_comparison['Actual'],
                                        mode='lines', name='Actual Price', line=dict(color='blue', width=2)))

    model_plot_colors = {
        'Linear Regression (Abs Price)': 'red',
        'XGBoost (Abs Price - Tuned)': 'green',
        'Linear Regression (Returns)': 'cyan',
        'XGBoost (Returns - Tuned)': 'purple',
        'Ensemble (LR Abs + XGBoost Returns)': 'orange'
    }

    model_plot_dash = {
        'Linear Regression (Abs Price)': 'dot',
        'XGBoost (Abs Price - Tuned)': 'dash',
        'Linear Regression (Returns)': 'longdash',
        'XGBoost (Returns - Tuned)': 'dashdot',
        'Ensemble (LR Abs + XGBoost Returns)': 'solid'
    }

    for col_name in plot_data_comparison.columns:
        if col_name != 'Actual':
            fig_comparison.add_trace(go.Scatter(x=plot_data_comparison.index, y=plot_data_comparison[col_name],
                                                mode='lines', name=col_name,
                                                line=dict(color=model_plot_colors.get(col_name, 'grey'),
                                                          dash=model_plot_dash.get(col_name, 'solid'),
                                                          width=1 if col_name != 'Ensemble (LR Abs + XGBoost Returns)' else 2))) # Make ensemble line thicker

    fig_comparison.update_layout(
        title='Actual vs. Predicted Gold Prices (Test Set)',
        xaxis_title='Date',
        yaxis_title='Gold Price ($)',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comparison, use_container_width=True)