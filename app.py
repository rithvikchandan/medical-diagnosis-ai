import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import os 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Define the display_input function
def display_input(label, tooltip, key, input_type="number", min_value=None, max_value=None, step=None, format="%.6f"):
    """
    A helper function to display input fields with precise decimal handling.
    """
    if input_type == "number":
        # Use text_input and convert to float
        value = st.text_input(label, help=tooltip, key=key)
        
        # Try converting input to float and applying the format
        try:
            float_value = float(value)
            formatted_value = format % float_value
            return float(formatted_value)  # Ensure the value is returned as a float
        except ValueError:
            return None  # Return None if the conversion fails (invalid input)
        
    elif input_type == "text":
        return st.text_input(label, help=tooltip, key=key)
    elif input_type == "selectbox":
        return st.selectbox(label, options=tooltip, key=key)

import streamlit as st
import os

# Page Config
st.set_page_config(
    page_title="Disease Prediction",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
/* Background Image */
[data-testid="stAppViewContainer"] {
    background-image: url("background.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Overlay to make text readable */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5); /* Adjust opacity here */
}

/* Card Styling */
.css-1d391kg {
    background: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* Heading Styling */
h1, h2, h3 {
    color: white;
    font-family: 'Arial', sans-serif;
}

/* Input Field Styling */
.stTextInput>div>div>input {
    border-radius: 5px;
    border: 1px solid #6a11cb;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# Load Brain Tumor Model from Models folder
brain_tumor_model = load_model("Models/Brain_Tumour.keras")

# Load Models
models = {
    'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
    'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb')),
    'breast_cancer': pickle.load(open('Models/breast_cancer_model.sav', 'rb')),
    'brain_tumor': brain_tumor_model

}

# Sidebar Navigation
with st.sidebar:
    st.title("🚀 Navigation")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction", "Lung Cancer Prediction", "Hypo-Thyroid Prediction", "Breast Cancer Prediction",  "Brain Tumor Prediction"],
        icons=["house", "activity", "heart", "person", "lungs", "thermometer", "dna", "brain"],
        default_index=0,
        styles={
            "container": {"padding": "10px", "background": "white", "border-radius": "10px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "color": "black"},
            "nav-link-selected": {"background": "#6a11cb", "color": "white"}
        }
    )

# Home Page
if selected == "Home":
    st.title("🌟 Welcome to DISEASE PREDICTION APP 🌟")
    st.write("""
    This app predicts the likelihood of various diseases based on user inputs.
    Select a disease from the sidebar to get started.
    """)
# Disease Prediction Pages
# The disease prediction pages would have their inputs styled the same way as in the example you already provided.

# Example - Diabetes Prediction
elif selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    st.write("Enter the following details to predict diabetes:")

    # Input Fields
    Pregnancies = display_input('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies', 'number', min_value=0, max_value=20, step=1)
    Glucose = display_input('Glucose Level', 'Enter glucose level', 'Glucose', 'number', min_value=0, max_value=300, step=1)
    BloodPressure = display_input('Blood Pressure', 'Enter blood pressure value', 'BloodPressure', 'number', min_value=0, max_value=200, step=1)
    SkinThickness = display_input('Skin Thickness', 'Enter skin thickness value', 'SkinThickness', 'number', min_value=0, max_value=100, step=1)
    Insulin = display_input('Insulin Level', 'Enter insulin level', 'Insulin', 'number', min_value=0, max_value=1000, step=1)
    BMI = display_input('BMI', 'Enter Body Mass Index value', 'BMI', 'number', min_value=0.0, max_value=100.0, step=0.1)
    DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function', 'Enter diabetes pedigree function value', 'DiabetesPedigreeFunction', 'number', min_value=0.0, max_value=2.5, step=0.01)
    Age = display_input('Age', 'Enter age of the person', 'Age', 'number', min_value=0, max_value=120, step=1)

    if st.button('Predict Diabetes'):
        with st.spinner("Predicting..."):
            diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            st.success('The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic')

# Heart Disease Prediction Page
elif selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    st.write("Enter the following details to predict heart disease:")

    # Input Fields
    age = display_input('Age', 'Enter age of the person', 'age', 'number', min_value=0, max_value=120, step=1)
    sex = display_input('Sex (1 = male; 0 = female)', 'Enter sex of the person', 'sex', 'number', min_value=0, max_value=1, step=1)
    cp = display_input('Chest Pain types (0, 1, 2, 3)', 'Enter chest pain type', 'cp', 'number', min_value=0, max_value=3, step=1)
    trestbps = display_input('Resting Blood Pressure', 'Enter resting blood pressure', 'trestbps', 'number', min_value=0, max_value=200, step=1)
    chol = display_input('Serum Cholesterol in mg/dl', 'Enter serum cholesterol', 'chol', 'number', min_value=0, max_value=600, step=1)
    fbs = display_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', 'Enter fasting blood sugar', 'fbs', 'number', min_value=0, max_value=1, step=1)
    restecg = display_input('Resting Electrocardiographic results (0, 1, 2)', 'Enter resting ECG results', 'restecg', 'number', min_value=0, max_value=2, step=1)
    thalach = display_input('Maximum Heart Rate achieved', 'Enter maximum heart rate', 'thalach', 'number', min_value=0, max_value=300, step=1)
    exang = display_input('Exercise Induced Angina (1 = yes; 0 = no)', 'Enter exercise induced angina', 'exang', 'number', min_value=0, max_value=1, step=1)
    oldpeak = display_input('ST depression induced by exercise', 'Enter ST depression value', 'oldpeak', 'number', min_value=0.0, max_value=10.0, step=0.1)
    slope = display_input('Slope of the peak exercise ST segment (0, 1, 2)', 'Enter slope value', 'slope', 'number', min_value=0, max_value=2, step=1)
    ca = display_input('Major vessels colored by fluoroscopy (0-3)', 'Enter number of major vessels', 'ca', 'number', min_value=0, max_value=3, step=1)
    thal = display_input('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)', 'Enter thal value', 'thal', 'number', min_value=0, max_value=2, step=1)

    if st.button('Predict Heart Disease'):
        with st.spinner("Predicting..."):
            # Get prediction and probability
            probabilities = models['heart_disease'].predict_proba([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])[0]

            # Flip prediction logic (since 0 & 1 are reversed)
            if probabilities[0] > probabilities[1]:  # If No Disease Probability is Higher
                st.error(f"🚨 The person has heart disease. Confidence: {probabilities[0] * 100:.2f}%")
            else:
                st.success(f"✅ The person does NOT have heart disease. Confidence: {probabilities[1] * 100:.2f}%")

# Parkinson's Disease Prediction Page
elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")
    st.write("Enter the following details to predict Parkinson's disease:")

    # Input Fields with Fixed Precision
    MDVP_Fo = display_input('MDVP:Fo(Hz)', 'Enter MDVP:Fo(Hz) value', 'MDVP_Fo', 'number', min_value=0.0, max_value=1000.0, step=0.1, format="%.1f")
    MDVP_Fhi = display_input('MDVP:Fhi(Hz)', 'Enter MDVP:Fhi(Hz) value', 'MDVP_Fhi', 'number', min_value=0.0, max_value=1000.0, step=0.1, format="%.1f")
    MDVP_Flo = display_input('MDVP:Flo(Hz)', 'Enter MDVP:Flo(Hz) value', 'MDVP_Flo', 'number', min_value=0.0, max_value=1000.0, step=0.1, format="%.1f")
    Jitter_percent = display_input('MDVP:Jitter(%)', 'Enter MDVP:Jitter(%) value', 'Jitter_percent', 'number', min_value=0.00001, max_value=1.0, step=0.00001, format="%.5f")
    Jitter_Abs = display_input('MDVP:Jitter(Abs)', 'Enter MDVP:Jitter(Abs) value', 'Jitter_Abs', 'number', min_value=0.00001, max_value=0.1, step=0.00001, format="%.6f")
    RAP = display_input('MDVP:RAP', 'Enter MDVP:RAP value', 'RAP', 'number', min_value=0.00001, max_value=0.1, step=0.00001, format="%.5f")
    PPQ = display_input('MDVP:PPQ', 'Enter MDVP:PPQ value', 'PPQ', 'number', min_value=0.00001, max_value=0.1, step=0.00001, format="%.5f")
    DDP = display_input('Jitter:DDP', 'Enter Jitter:DDP value', 'DDP', 'number', min_value=0.00001, max_value=0.1, step=0.00001, format="%.5f")
    Shimmer = display_input('MDVP:Shimmer', 'Enter MDVP:Shimmer value', 'Shimmer', 'number', min_value=0.0001, max_value=1.0, step=0.0001, format="%.4f")
    Shimmer_dB = display_input('MDVP:Shimmer(dB)', 'Enter MDVP:Shimmer(dB) value', 'Shimmer_dB', 'number', min_value=0.01, max_value=1.0, step=0.01, format="%.3f")
    APQ3 = display_input('Shimmer:APQ3', 'Enter Shimmer:APQ3 value', 'APQ3', 'number', min_value=0.0001, max_value=1.0, step=0.0001, format="%.4f")
    APQ5 = display_input('Shimmer:APQ5', 'Enter Shimmer:APQ5 value', 'APQ5', 'number', min_value=0.0001, max_value=1.0, step=0.0001, format="%.4f")
    APQ = display_input('MDVP:APQ', 'Enter MDVP:APQ value', 'APQ', 'number', min_value=0.0001, max_value=1.0, step=0.0001, format="%.4f")
    DDA = display_input('Shimmer:DDA', 'Enter Shimmer:DDA value', 'DDA', 'number', min_value=0.0001, max_value=1.0, step=0.0001, format="%.4f")
    NHR = display_input('NHR', 'Enter NHR value', 'NHR', 'number', min_value=0.0, max_value=1.0, step=0.0001, format="%.2f")
    HNR = display_input('HNR', 'Enter HNR value', 'HNR', 'number', min_value=0.0, max_value=40.0, step=0.1, format="%.1f")
    RPDE = display_input('RPDE', 'Enter RPDE value', 'RPDE', 'number', min_value=0.0, max_value=1.0, step=0.0001, format="%.2f")
    DFA = display_input('DFA', 'Enter DFA value', 'DFA', 'number', min_value=0.0, max_value=1.0, step=0.0001, format="%.2f")
    Spread1 = display_input('Spread1', 'Enter Spread1 value', 'Spread1', 'number', min_value=-10.0, max_value=10.0, step=0.1, format="%.1f")
    Spread2 = display_input('Spread2', 'Enter Spread2 value', 'Spread2', 'number', min_value=-10.0, max_value=10.0, step=0.1, format="%.1f")
    D2 = display_input('D2', 'Enter D2 value', 'D2', 'number', min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
    PPE = display_input('PPE', 'Enter PPE value', 'PPE', 'number', min_value=0.0, max_value=1.0, step=0.001, format="%.2f")

    if st.button("Predict Parkinson's Disease"):
        with st.spinner("Predicting..."):
            # Get decision function values
            parkinsons_prediction = models['parkinsons'].decision_function([
                [MDVP_Fo, MDVP_Fhi, MDVP_Flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, Spread1, Spread2, D2, PPE]
            ])

            # Assuming positive class is on the right side of the hyperplane
            if parkinsons_prediction[0] >= 0:
                st.success("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")

# Lung Cancer Prediction Page
elif selected == "Lung Cancer Prediction":
    st.title("Lung Cancer Prediction")
    st.write("Enter the following details to predict lung cancer:")

    # Input Fields
    GENDER = display_input('Gender (1 = Male; 0 = Female)', 'Enter gender of the person', 'GENDER', 'number', min_value=0, max_value=1, step=1)
    AGE = display_input('Age', 'Enter age of the person', 'AGE', 'number', min_value=0, max_value=120, step=1)
    SMOKING = display_input('Smoking (1 = Yes; 0 = No)', 'Enter if the person smokes', 'SMOKING', 'number', min_value=0, max_value=1, step=1)
    YELLOW_FINGERS = display_input('Yellow Fingers (1 = Yes; 0 = No)', 'Enter if the person has yellow fingers', 'YELLOW_FINGERS', 'number', min_value=0, max_value=1, step=1)
    ANXIETY = display_input('Anxiety (1 = Yes; 0 = No)', 'Enter if the person has anxiety', 'ANXIETY', 'number', min_value=0, max_value=1, step=1)
    PEER_PRESSURE = display_input('Peer Pressure (1 = Yes; 0 = No)', 'Enter if the person is under peer pressure', 'PEER_PRESSURE', 'number', min_value=0, max_value=1, step=1)
    CHRONIC_DISEASE = display_input('Chronic Disease (1 = Yes; 0 = No)', 'Enter if the person has a chronic disease', 'CHRONIC_DISEASE', 'number', min_value=0, max_value=1, step=1)
    FATIGUE = display_input('Fatigue (1 = Yes; 0 = No)', 'Enter if the person experiences fatigue', 'FATIGUE', 'number', min_value=0, max_value=1, step=1)
    ALLERGY = display_input('Allergy (1 = Yes; 0 = No)', 'Enter if the person has allergies', 'ALLERGY', 'number', min_value=0, max_value=1, step=1)
    WHEEZING = display_input('Wheezing (1 = Yes; 0 = No)', 'Enter if the person experiences wheezing', 'WHEEZING', 'number', min_value=0, max_value=1, step=1)
    ALCOHOL_CONSUMING = display_input('Alcohol Consuming (1 = Yes; 0 = No)', 'Enter if the person consumes alcohol', 'ALCOHOL_CONSUMING', 'number', min_value=0, max_value=1, step=1)
    COUGHING = display_input('Coughing (1 = Yes; 0 = No)', 'Enter if the person experiences coughing', 'COUGHING', 'number', min_value=0, max_value=1, step=1)
    SHORTNESS_OF_BREATH = display_input('Shortness Of Breath (1 = Yes; 0 = No)', 'Enter if the person experiences shortness of breath', 'SHORTNESS_OF_BREATH', 'number', min_value=0, max_value=1, step=1)
    SWALLOWING_DIFFICULTY = display_input('Swallowing Difficulty (1 = Yes; 0 = No)', 'Enter if the person has difficulty swallowing', 'SWALLOWING_DIFFICULTY', 'number', min_value=0, max_value=1, step=1)
    CHEST_PAIN = display_input('Chest Pain (1 = Yes; 0 = No)', 'Enter if the person experiences chest pain', 'CHEST_PAIN', 'number', min_value=0, max_value=1, step=1)

    if st.button("Predict Lung Cancer"):
        # Collecting inputs
        inputs = [GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]

        # Check if any input is None (invalid input)
        if None in inputs:
            st.error("Please fill in all fields correctly.")
        else:
            with st.spinner("Predicting..."):
                # Use predict_proba() if available
                if hasattr(models['lung_cancer'], 'predict_proba'):
                    lungs_prediction_proba = models['lung_cancer'].predict_proba([inputs])
                    print(f"Prediction Probabilities: {lungs_prediction_proba[0]}")  # This prints both class probabilities

                    # Extract the probability for lung cancer (class 1)
                    probability = lungs_prediction_proba[0][1]
                    
                    # Apply the threshold (0.02 for 2% confidence)
                    if probability >= 0.005:
                        st.success(f"The person has lung cancer. Confidence: {probability * 100:.2f}%")
                    else:
                        st.success(f"The person does not have lung cancer. Confidence: {probability * 100:.2f}%")
                else:
                    # If the model does not support predict_proba(), use predict() directly
                    lungs_prediction = models['lung_cancer'].predict([inputs])
                    st.success("The person has lung cancer" if lungs_prediction[0] == 1 else "The person does not have lung cancer")

# Hypo-Thyroid Prediction Page
elif selected == "Hypo-Thyroid Prediction":
    st.title("Hypo-Thyroid Prediction")
    st.write("Enter the following details to predict hypo-thyroid disease:")

    # Input Fields for Hypo-Thyroid Prediction
    age = display_input('Age', 'Enter age of the person', 'age', 'number', min_value=0, max_value=120, step=1)
    sex = display_input('Sex (1 = Male; 0 = Female)', 'Enter sex of the person', 'sex', 'number', min_value=0, max_value=1, step=1)
    on_thyroxine = display_input('On Thyroxine (1 = Yes; 0 = No)', 'Enter if the person is on thyroxine', 'on_thyroxine', 'number', min_value=0, max_value=1, step=1)
    tsh = display_input('TSH Level', 'Enter TSH level', 'tsh', 'number', min_value=0.0, max_value=200.0, step=0.1)
    t3_measured = display_input('T3 Measured (1 = Yes; 0 = No)', 'Enter if T3 was measured', 't3_measured', 'number', min_value=0, max_value=1, step=1)
    t3 = display_input('T3 Level', 'Enter T3 level', 't3', 'number', min_value=0.0, max_value=10.0, step=0.1)
    tt4 = display_input('TT4 Level', 'Enter TT4 level', 'tt4', 'number', min_value=0.0, max_value=500.0, step=0.1)

    if st.button("Predict Hypo-Thyroid"):
        # Collecting inputs for Hypo-Thyroid Prediction
        inputs = [age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]

        # Check if any input is None (invalid input)
        if None in inputs:
            st.error("Please fill in all fields correctly.")
        else:
            with st.spinner("Predicting..."):
                # Use predict_proba() if available
                if hasattr(models['thyroid'], 'predict_proba'):
                    thyroid_prediction_proba = models['thyroid'].predict_proba([inputs])
                    print(f"Prediction Probabilities: {thyroid_prediction_proba[0]}")  # This prints both class probabilities

                    # Extract the probability for hypo-thyroidism (class 1)
                    probability = thyroid_prediction_proba[0][1]

                    # Apply the threshold (0.02 for 2% confidence)
                    if probability >= 0.5:
                        st.success(f"The person has hypo-thyroidism. Confidence: {probability * 100:.2f}%")
                    else:
                        st.success(f"The person does not have hypo-thyroidism. Confidence: {probability * 100:.2f}%")
                else:
                    # If the model does not support predict_proba(), use predict() directly
                    thyroid_prediction = models['thyroid'].predict([inputs])
                    st.success("The person has hypo-thyroidism" if thyroid_prediction[0] == 1 else "The person does not have hypo-thyroidism")

# breast cancer prediction page

elif selected == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")
    st.write("Enter the following details to predict breast cancer:")

    # Input Fields (Using Correct Feature Names)
    mean_radius = st.number_input("Mean Radius", format="%.3f")
    mean_texture = st.number_input("Mean Texture", format="%.3f")
    mean_perimeter = st.number_input("Mean Perimeter", format="%.2f")
    mean_area = st.number_input("Mean Area", format="%.1f")
    mean_smoothness = st.number_input("Mean Smoothness", format="%.4f")
    mean_compactness = st.number_input("Mean Compactness", format="%.4f")
    mean_concavity = st.number_input("Mean Concavity", format="%.4f")
    mean_concave_points = st.number_input("Mean Concave Points", format="%.4f")
    mean_symmetry = st.number_input("Mean Symmetry", format="%.4f")
    mean_fractal_dimension = st.number_input("Mean Fractal Dimension", format="%.4f")

    radius_error = st.number_input("Radius Error", format="%.3f")
    texture_error = st.number_input("Texture Error", format="%.3f")
    perimeter_error = st.number_input("Perimeter Error", format="%.2f")
    area_error = st.number_input("Area Error", format="%.1f")
    smoothness_error = st.number_input("Smoothness Error", format="%.4f")
    compactness_error = st.number_input("Compactness Error", format="%.4f")
    concavity_error = st.number_input("Concavity Error", format="%.4f")
    concave_points_error = st.number_input("Concave Points Error", format="%.4f")
    symmetry_error = st.number_input("Symmetry Error", format="%.4f")
    fractal_dimension_error = st.number_input("Fractal Dimension Error", format="%.4f")

    worst_radius = st.number_input("Worst Radius", format="%.3f")
    worst_texture = st.number_input("Worst Texture", format="%.3f")
    worst_perimeter = st.number_input("Worst Perimeter", format="%.2f")
    worst_area = st.number_input("Worst Area", format="%.1f")
    worst_smoothness = st.number_input("Worst Smoothness", format="%.4f")
    worst_compactness = st.number_input("Worst Compactness", format="%.4f")
    worst_concavity = st.number_input("Worst Concavity", format="%.4f")
    worst_concave_points = st.number_input("Worst Concave Points", format="%.4f")
    worst_symmetry = st.number_input("Worst Symmetry", format="%.4f")
    worst_fractal_dimension = st.number_input("Worst Fractal Dimension", format="%.4f")

    # Button for Prediction
    if st.button("Predict Breast Cancer"):
        with st.spinner("Predicting..."):
            try:
                # Ensure the input order matches model training
                input_data = [[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                               mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
                               radius_error, texture_error, perimeter_error, area_error, smoothness_error,
                               compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                               worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
                               worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]]
                
                # Scale the input using the same scaler used in training
                user_input_scaled = scaler.transform(input_data)

                # Get the prediction
                prediction = models['breast_cancer'].predict(user_input_scaled)[0]

                # Swap the labels (0 ↔ 1) if the model predicts the opposite
                corrected_prediction = 1 if prediction == 0 else 0  

                # Display Corrected Output
                if corrected_prediction == 1:
                    st.success("⚠️ The tumor is **Malignant (Cancerous)**. Consult a doctor.")
                else:
                    st.success("✅ The tumor is **Benign (Non-Cancerous)**. No cancer detected.")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

elif selected == "Brain Tumor Prediction":
    st.title(" Brain Tumor Detection Using AI")
    st.write("Upload an MRI image to check for brain tumor presence.")

    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded MRI Image", use_container_width=True)


        # Preprocess Image
        def preprocess_image(img):
            img = img.resize((64, 64))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0
            return img

        processed_image = preprocess_image(image_pil)
        result = brain_tumor_model.predict(processed_image)

        # Show Prediction
        prediction = "Tumor Detected (Positive)" if result[0][0] > 0.5 else "No Tumor Detected (Negative)"
        st.subheader(f"🔍 **Prediction:** {prediction}")

