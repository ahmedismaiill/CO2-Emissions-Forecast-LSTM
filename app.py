import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st

# Load Models and Scalers
pivot_df = pd.read_csv('pivot_df.csv')
pivot_df["date"] = pd.to_datetime(pivot_df["date"]) 

model = load_model('models/lstm_model.keras')
x_scaler = joblib.load('models/scaler_X.pkl')
y_scaler = joblib.load('models/scaler_y.pkl')
le_country = joblib.load("models/label_encoder_country.pkl")
TIME_STEPS = 12
MAX_FORECAST_DAYS = 7

feature_cols = ['Domestic Aviation', 'Ground Transport', 'Industry', 'International Aviation', 'Power', 'Residential','country_enc']
emojis = ['✈️', '🚗', '🏭', '✈️', '⚡', '🏠']

country_list = countries = {
    "Brazil": "flags/Flag_of_Brazil.svg.png",
    "China": "flags/Flag-China.webp",
    "EU27 & UK": "flags/Flag_of_Europe.svg.png",       
    "France": "flags/Flag_of_France.svg",
    "Germany": "flags/Flag_of_Germany.svg.png",
    "India": "flags/Flag_of_India.svg.webp",
    "Italy": "flags/Flag_of_Italy.svg.webp",
    "Japan": "flags/Flag_of_Japan.svg.webp",
    "ROW": "flags/istockphoto-93436189-612x612.jpg",           # Rest of the World
    "Russia": "flags/Flag_of_Russia.svg",
    "Spain": "flags/Flag_of_Spain.svg",
    "UK": "flags/Flag_of_the_United_Kingdom.svg",
    "US": "flags/Flag_of_the_United_States.svg",
    "WORLD": "flags/Flag-map_of_the_world_(2018).png"          
}

def predict_for_country_date(model,country,date,pivot_df,x_scaler,y_scaler,TIME_STEPS=TIME_STEPS,MAX_FORECAST_DAYS=MAX_FORECAST_DAYS):
    date = pd.Timestamp(date)
    country_enc = le_country.transform([country])[0]
    temp = pivot_df[pivot_df["country"] == country].sort_values("date")

    temp = temp[temp["date"] < date]

    if len(temp) < TIME_STEPS:
        raise ValueError(f"Not enough data for {country}. "f"Need {TIME_STEPS}, got {len(temp)}")

    last_known_date = temp["date"].max()
    Error = False
    if date > last_known_date + pd.Timedelta(days=MAX_FORECAST_DAYS):
        Error = True
        st.error(
              f"Prediction date {date.date()} Exceeds the Maximum "
              f"Allowed Horizon of {MAX_FORECAST_DAYS} Days "
              f"After {last_known_date.date()}."
        )

    last_window = temp.iloc[-TIME_STEPS:].copy()
    last_window["country_enc"] = country_enc

    last_window_features = last_window[feature_cols].values
    last_window_scaled = x_scaler.transform(last_window_features)
    last_window_scaled = last_window_scaled.reshape(1, TIME_STEPS, len(feature_cols))
    
    pred_scaled = model.predict(last_window_scaled, verbose=0)
    pred = y_scaler.inverse_transform(pred_scaled)[0]

    new_record = {
        "date": date,
        "country": country,
        "Domestic Aviation": pred[0],
        "Ground Transport": pred[1],
        "Industry": pred[2],
        "International Aviation": pred[3],
        "Power": pred[4],
        "Residential": pred[5],
        "country_enc": country_enc
    }

    pivot_df = pd.concat(
        [pivot_df, pd.DataFrame([new_record])],
        ignore_index=True
    )
    pivot_df.sort_values(["country", "date"], inplace=True)
    pivot_df.to_csv("pivot_df.csv", index=False)
    return pred, last_window, pivot_df,Error

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Build Streamlit
st.cache_data.clear()
st.set_page_config(page_title="Co2 Emission Prediction",page_icon='💨',layout="wide")

col1, col2, col3 = st.columns([1, 2, 1])  

with col2:
    st.title('CO2 Emissions Prediction App 🌎📈')
    st.image('carbon-dioxide-emissions-featured.jpg', use_container_width=True)
    
st.markdown(
        """
        <div style="padding:10px; border-radius:5px; background-color:#0e1117;">
            <h4 style="color:#faf7f7;">🌍 CO2 Emissions Prediction App predicts daily CO2 emissions for a selected country and date across six sectors: ✈️ Domestic Aviation, 🚗 Ground Transport, 🏭 Industry, ✈️ International Aviation, ⚡ Power, and 🏠 Residential. The app uses an LSTM model 🤖 to forecast sector-wise emissions.</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("## Enter The Date and Country to Predict CO2 Emissions")

col1_input, col2_input = st.columns([1, 1])

with col1_input:
    selected_country = st.selectbox("Select Country", list(countries.keys()))
    st.image(countries[selected_country], width=500,use_container_width=True)
with col2_input:
    user_date = st.date_input("Select prediction date",value=pd.to_datetime("2023-06-01"))
  
    if st.button("Predict CO2 Emissions"):
       st.info("Predicting...")
       pred_values, last_window, pivot_df ,Error = predict_for_country_date(model, selected_country, user_date, pivot_df, x_scaler, y_scaler)
       if not Error:
         st.success(f"Prediction for {selected_country} on {user_date}:")
         sectors_with_emoji = [f"{e} {s}" for e, s in zip(emojis, feature_cols[:-1])]
         prediction_data = {
              "Sector": sectors_with_emoji,
               "Predicted Emission": [f"{pred:.2f} MtCO₂/day" for pred in pred_values]}
         df_pred = pd.DataFrame(prediction_data)
         st.table(df_pred)

with st.expander("𝄜 See The Hole DataFrame"):
    st.dataframe(pivot_df.iloc[:,:-1])