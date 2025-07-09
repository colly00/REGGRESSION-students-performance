import joblib
import numpy as np 
import streamlit as st 

# Load the trained regression model
model = joblib.load('precapst_1.joblib')

# Title of the app
st.title('🎓 Student Performance Score Predictor (Regression)')

# Input fields 
study_hours_per_day = st.number_input('📘 Study hours per day', min_value=0, max_value=10)
social_media_hours = st.number_input('📱 Social media hours per day', min_value=0, max_value=10)
netflix_hours = st.number_input('🎬 Netflix hours per day', min_value=0, max_value=10)
part_time_job = st.selectbox('💼 Has part-time job?', ['yes', 'no'])
attendance_percentage = st.number_input('📊 Attendance percentage', min_value=0, max_value=100)
sleep_hours = st.number_input('🛏️ Sleep hours per day', min_value=0, max_value=10)
extracurricular_participation = st.selectbox('🎭 Participates in extracurricular activities?', ['yes', 'no'])

# Encode categorical inputs
part_encoded = 1 if part_time_job == 'yes' else 0
extra_encoded = 1 if extracurricular_participation == 'yes' else 0

# Prediction button
if st.button('🔍 Predict Score'):
    input_data = np.array([[study_hours_per_day, social_media_hours, netflix_hours,
                            part_encoded, attendance_percentage, sleep_hours, extra_encoded]])
    
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))

    # Output
    st.subheader('📈 Predicted Performance Score')
    st.success(f"Estimated Score: **{round(prediction, 2)} / 100**")
