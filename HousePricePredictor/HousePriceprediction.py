import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = "https://raw.githubusercontent.com/sonarsushant/California-House-Price-Prediction/refs/heads/master/housing.csv"
data = pd.read_csv(url)

# Preprocess the data
# Convert categorical features to numerical
data['ocean_proximity'] = LabelEncoder().fit_transform(data['ocean_proximity'])

# Select features and target variable
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit interface
st.title("California House Price Prediction")

# Input features
longitude = st.slider("Longitude", min_value=float(X['longitude'].min()), max_value=float(X['longitude'].max()), value=float(X['longitude'].mean()))
latitude = st.slider("Latitude", min_value=float(X['latitude'].min()), max_value=float(X['latitude'].max()), value=float(X['latitude'].mean()))
housing_median_age = st.slider("Housing Median Age", min_value=int(X['housing_median_age'].min()), max_value=int(X['housing_median_age'].max()), value=int(X['housing_median_age'].mean()))
total_rooms = st.slider("Total Rooms", min_value=int(X['total_rooms'].min()), max_value=int(X['total_rooms'].max()), value=int(X['total_rooms'].mean()))
total_bedrooms = st.slider("Total Bedrooms", min_value=int(X['total_bedrooms'].min()), max_value=int(X['total_bedrooms'].max()), value=int(X['total_bedrooms'].mean()))
population = st.slider("Population", min_value=int(X['population'].min()), max_value=int(X['population'].max()), value=int(X['population'].mean()))
households = st.slider("Households", min_value=int(X['households'].min()), max_value=int(X['households'].max()), value=int(X['households'].mean()))
median_income = st.slider("Median Income", min_value=float(X['median_income'].min()), max_value=float(X['median_income'].max()), value=float(X['median_income'].mean()))
ocean_proximity = st.selectbox("Ocean Proximity", options=data['ocean_proximity'].unique())

# Predict button
if st.button("Predict Price"):
    # Prepare input data for prediction
    input_data = [[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]]
    
    # Make prediction
    predicted_price = model.predict(input_data)
    
    # Display the prediction
    st.success(f"Predicted House Price: ${predicted_price[0]:,.2f}")

# Run the Streamlit app using the command:
# streamlit run your_script_name.py
