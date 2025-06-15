# 💰 Gold Price Predictor: Unraveling Gold's Future Trends

This project showcases a machine learning-powered web application built with Streamlit, designed to predict gold prices. It employs various models, including XGBoost and Linear Regression, to forecast future prices and analyze historical model performance.

## ✨ Features

* **Robust Data Pipeline:** Handles historical gold price data from 2013-09-13 to 2024-11-22, including cleaning, combining, and managing time gaps.
* **Comprehensive Feature Engineering:** Utilizes key financial time-series features such as lagged prices (up to 2 days), Simple Moving Averages (5-day, 20-day), daily returns, lagged daily returns, and 10-day price volatility. Temporal features (day of week, month, year, day of year) are also included.
* **Multiple Prediction Models:**
    * **Linear Regression (Absolute Price):** A strong baseline model predicting absolute gold prices directly.
    * **XGBoost (Absolute Price - Tuned):** A powerful tree-based model attempting direct price prediction.
    * **Linear Regression (Returns):** Predicts daily percentage change in gold price.
    * **XGBoost (Returns - Tuned):** Optimized tree-based model predicting daily percentage change.
    * **Ensemble Model (LR Abs + XGBoost Returns):** A combined model averaging predictions from the Linear Regression (Abs Price) and the XGBoost (Returns - Tuned) models, leveraging their individual strengths.
* **Interactive Streamlit Web Application:**
    * **Predict Tomorrow's Price:** Input today's OHLC data to get a forecast for the next trading day.
    * **Analyze Historical Price:** Select a past date from the test set (2024-08-02 onwards) to see how accurately the model would have predicted its price, compared against the actual value.
    * **Model Performance Overview:** A dedicated tab displaying comprehensive performance metrics (MAE, RMSE, R-squared) and an interactive plot comparing all models on the test set. Includes prediction accuracy categories (Excellent, Good, Considerable Deviation).
    * **Dynamic Model Selection:** Allows users to choose and compare predictions from different models.
    * **Interactive Visualizations:** Uses Plotly for engaging and zoomable charts.

## 📊 Model Performance Highlights (on Test Set: 2024-08-02 to 2024-11-22)

The Ensemble Model demonstrated the best combined performance:

| Model                                     | MAE     | RMSE    | R-squared |
| :---------------------------------------- | :------ | :------ | :-------- |
| LR (Abs Price)                            | ~5.84   | ~7.62   | ~0.994    |
| XGBoost (Abs Price - Tuned)               | ~174.31 | ~199.39 | ~-2.88    |
| LR (Returns - Abs Price Derived)          | ~7.96   | ~9.99   | ~0.990    |
| XGBoost (Returns - Tuned, Abs Price Derived) | ~22.84  | ~28.68  | ~0.916    |
| **Ensemble (LR Abs + XGBoost Returns)** | **~13.79** | **~17.71** | **~0.969**|

*Note: R-squared values closer to 1.0 indicate a better fit. Negative R-squared means the model performs worse than simply predicting the mean.*

**Key Learnings:**
* **Linear Regression's Strength:** For highly autocorrelated time series like gold prices, simple Linear Regression excels at capturing strong linear trends and can achieve remarkably high accuracy.
* **XGBoost and Extrapolation:** Tree-based models like XGBoost struggle significantly with extrapolating far beyond their training data range when predicting absolute values, leading to poor performance.
* **Returns vs. Absolute Prices:** XGBoost performs much better when predicting daily returns (percentage changes) rather than absolute prices, as returns are more bounded and stationary.
* **Power of Ensembling:** Combining models (like LR for trend and XGBoost for changes) can yield superior results, as seen with the Ensemble model's improved R-squared.

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Git (for cloning the repository)
* A GitHub account (to create your repository)
* (Optional, but recommended) A Google Drive account for running the Colab notebook.

### Installation & Setup

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
    (Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details).

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **Windows (Command Prompt):** `.\venv\Scripts\activate.bat`
    * **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
        *(If you encounter an Execution Policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell, confirm `Y`, then retry activation.)*
    * **macOS/Linux:** `source venv/bin/activate`

4.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas numpy scikit-learn xgboost plotly
    ```

### Data Setup

1.  **Download Raw Data:** Obtain the CSV files from their original Kaggle source (you'll need to join them as in the Colab notebook):
    * `gold prices.csv`: [https://www.kaggle.com/datasets/kapturovalexander/gold-and-silver-prices-2013-2023](https://www.kaggle.com/datasets/kapturovalexander/gold-and-silver-prices-2013-2023)
    * `Gold Futures Historical Data (23.01.24-22.11.24).csv`: (This was likely from a different source as you mentioned, ensure you have this file as well).
2.  **Place Data:** Put these two `.csv` files into the root directory of your project (same folder as `app.py`).

### Model Training & Saving (Google Colab)

1.  **Open Google Colab Notebook:**
    * Go to [colab.research.google.com](https://colab.research.google.com/).
    * Create a new notebook and copy the entire content from the `Gold_Price_Prediction_FINAL_Training.ipynb` (the Python code from **Part 1, Step 2** of this guide) into your Colab notebook.
2.  **Upload CSVs to Google Drive:** Ensure your raw `.csv` files are uploaded to a folder (e.g., `gold_data`) in your Google Drive, and update the `DRIVE_PATH` variable in the Colab notebook accordingly.
3.  **Run All Cells:** Execute all cells in the Colab notebook sequentially. This will train all models and save them as `.joblib` files to your Google Drive.
4.  **Download Models:** After successful execution, download all the `.joblib` files (6 files) from your Google Drive `gold_data` folder and place them into the root directory of your local project (same folder as `app.py`).

### Running the Streamlit Application

1.  **Activate Virtual Environment:** Ensure your virtual environment is active in your terminal.
2.  **Run Streamlit:**
    ```bash
    streamlit run app.py
    ```
    Your default web browser will open to the Streamlit application.

## 🤝 Contributing

Feel free to fork this repository, open issues, or submit pull requests.

## ⚠️ Disclaimer

This tool is for educational and demonstrative purposes only. Gold price predictions are highly speculative and should not be used as financial advice or for real-world trading decisions.

## 📚 Data Source

* **Gold & Silver Prices (2013-2023):** [Kaggle](https://www.kaggle.com/datasets/kapturovalexander/gold-and-silver-prices-2013-2023) by Kapturov Alexander.
* **Gold Futures Historical Data (2024):** (Specify your source here if you have a public link, otherwise state "Obtained from [Your Source/Process]" or omit if sensitive).