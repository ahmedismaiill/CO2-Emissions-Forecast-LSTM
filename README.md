# 🌍 Global CO2 Emissions Forecasting: A Deep Learning Approach

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
The rapid accumulation of Carbon Dioxide (CO2) is a primary driver of global climate change. Understanding historical emission patterns and accurately forecasting future trends is crucial for environmental policymaking.

This project implements a **Long Short-Term Memory (LSTM)** neural network to predict daily CO2 emissions across **14 countries** and **6 economic sectors**. By transforming raw transactional data into a multivariate time-series, the model captures complex non-linear dependencies, seasonality, and the impact of global events (such as the COVID-19 pandemic).

The project further integrates **MLOps** practices using **MLflow** for experiment tracking and features a user-friendly **Streamlit Dashboard** for real-time forecasting.

---

## 📊 Dataset Description
The analysis utilizes a comprehensive dataset recording daily CO2 emissions in **MtCO₂/day** (Million Tonnes of Carbon Dioxide per day).

*   **Scope:** 14 Countries (including Brazil, China, US, India, Germany, etc.)
*   **Sectors:** Power, Industry, Ground Transport, Residential, Domestic Aviation, International Aviation.
*   **Original Size:** 135,408 records (Long Format).
*   **Processed Size:** 22,568 records (Wide Format, Pivot Table).

**Data Transformation:**
The data was pivoted from a "long" format (one row per sector) to a "wide" format. This allows the LSTM model to learn the correlations between all 6 sectors simultaneously for a specific country and date.

---

## 🎥 Project Demos

### 1. MLflow Experiment Tracking
A walkthrough of the training pipeline, metric logging (R², MSE), and artifact management (visualizations/models) using MLflow.

https://github.com/user-attachments/assets/d28a8c97-d54e-4b77-ab35-4dadaca1c160

### 2. Streamlit Forecasting Dashboard
A demonstration of the interactive GUI where users can select a country and date to generate real-time emission predictions.

https://github.com/user-attachments/assets/3143b04d-ac54-4ec4-92f1-10be3989e50f

---

## 🛠 Methodology

### 1. Data Preparation
*   **Datetime Conversion:** Standardizing dates and sorting chronologically.
*   **Pivoting:** Reshaping data so columns represent sectors (`Domestic Aviation`, `Power`, etc.).
*   **Feature Engineering:**
    *   **Country Encoding:** Using `LabelEncoder` to transform country names into numerical features.
    *   **Scaling:** Applying `MinMaxScaler` to normalize features and targets between 0 and 1.
    *   **Sliding Window:** Creating a time-series window of **12 days** (Time Steps) to predict the next day's emissions.

### 2. LSTM Model Architecture
We constructed a Deep Learning model using **TensorFlow/Keras**:
*   **Input Layer:** `(12, 7)` shape (12 days history, 6 sectors + 1 country code).
*   **LSTM Layers:** Two LSTM layers (128 units and 64 units) to capture temporal dependencies.
*   **Regularization:** Dropout layers (20%) to prevent overfitting.
*   **Output Layer:** A Dense layer with 6 units (one for each sector) using ReLU activation.

### 3. MLOps & Deployment
*   **MLflow:** Used to log experiments, save the model with an inferred signature, and track metrics (MAE, MSE, RMSE, R²).
*   **Streamlit:** A Python-based web app that loads the trained `.keras` model and scalers to provide an interface for non-technical users.

---

## 📈 Results
The model was evaluated on a held-out test set (last 15% of the timeline), achieving exceptional accuracy.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | **0.9889** | The model explains ~99% of the variance in the data. |
| **MSE** | 0.3954 | Low squared error indicating a tight fit. |
| **MAE** | 0.2318 | Minimal average absolute deviation per prediction. |

**Key Visual Insights:**
*   **Seasonality:** Accurately captured winter heating spikes in the *Residential* sector.
*   **Impact of Events:** Modeled the sharp decline and recovery in *Ground Transport* during the 2020 pandemic.

---

## 💻 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/ahmedismaiill/CO2-Emissions-Forecast-LSTM.git
    cd CO2-Emissions-Forecast-LSTM
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Training Script (with MLflow)**
    ```bash
    python train_model.py
    # To view the MLflow UI:
    mlflow ui
    ```

4.  **Launch the Streamlit App**
    ```bash
    streamlit run app.py
    ```

---
## 📂 Project Structure

```text
├── .vscode/
│   └── settings.json
│
├── dataset/
│   └── dataset.csv                 # Original CO₂ emissions dataset
│
├── docs/
│   └── CO2 Emissions LSTM.pdf       # Detailed project report
│
├── flags/                          # Country flags used in dashboard
│   ├── Flag-China.webp
│   ├── Flag_of_Brazil.svg.png
│   ├── Flag_of_France.svg
│   ├── Flag_of_Germany.svg.png
│   ├── Flag_of_India.svg.webp
│   ├── Flag_of_Italy.svg.webp
│   ├── Flag_of_Japan.svg.webp
│   ├── Flag_of_Russia.svg
│   ├── Flag_of_Spain.svg
│   ├── Flag_of_the_United_Kingdom.svg
│   ├── Flag_of_the_United_States.svg
│   └── Flag-map_of_the_world_(2018).png
│
├── mlruns/                         # MLflow experiment tracking
│   ├── 0/
│   ├── <experiment_id>/
│   │   ├── <run_id>/
│   │   │   ├── artifacts/
│   │   │   │   ├── pred_vs_true.png
│   │   │   │   └── model/
│   │   │   │       └── model.keras
│   │   │   ├── metrics/
│   │   │   ├── params/
│   │   │   └── tags/
│   │   └── ...
│   └── .trash/
│
├── models/                         # Saved trained models & scalers
│   ├── lstm_model.keras
│   ├── scaler_X.pkl
│   ├── scaler_y.pkl
│   ├── label_encoder_country.pkl
│   ├── X_test.npy
│   └── y_test.npy
│
├── notebooks/
│   └── Co2Emission.ipynb            # EDA & model experimentation
│
├── app.py                          # Streamlit application
├── Mlflow.py                       # MLflow training & logging script
├── pivot_df.csv                    # Processed pivot dataset
├── requirements.txt                # Python dependencies
├── carbon-dioxide-emissions-featured.jpg
├── pred_vs_true.png
├── pred_vs_true_sectors_subplot.png
└── README.md                       # Project documentation
