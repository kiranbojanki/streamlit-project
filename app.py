import streamlit as st
import pandas as pd
import joblib

# Load the trained models
rf_model = joblib.load('random_forest_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')

# Function to predict heart disease using both models
def predict_heart_disease(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Streamlit App
st.title('Heart Disease Prediction App')

# Sidebar for user input
st.sidebar.header('Input Patient Data')

def user_input_features():
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.sidebar.selectbox('Sex', (1, 0))  # 1 = male, 0 = female
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', (0, 1, 2, 3))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol (chol)', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
    restecg = st.sidebar.selectbox('Resting ECG (restecg)', (0, 1, 2))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', (0, 1))
    oldpeak = st.sidebar.slider('ST Depression (oldpeak)', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment (slope)', (0, 1, 2))
    ca = st.sidebar.selectbox('Number of Major Vessels (ca)', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thalassemia (thal)', (0, 1, 2, 3))

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Store user input data
user_data = user_input_features()

# Display user input
st.subheader('Patient Input Data')
st.write(user_data)

# Predict using Random Forest
if st.button('Predict using Random Forest'):
    rf_prediction = predict_heart_disease(rf_model, user_data.iloc[0])
    st.subheader('Random Forest Prediction')
    st.write('Heart Disease: Yes' if rf_prediction == 1 else 'Heart Disease: No')

# Predict using Decision Tree
if st.button('Predict using Decision Tree'):
    dt_prediction = predict_heart_disease(dt_model, user_data.iloc[0])
    st.subheader('Decision Tree Prediction')
    st.write('Heart Disease: Yes' if dt_prediction == 1 else 'Heart Disease: No')
