import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# load trained model
model = joblib.load("breast_cancer_model.pkl")

# load dataset
df = pd.read_csv("Breast_cancer_dataset.csv")

# Remove unwanted columns
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Feature columns
features = df.drop("diagnosis", axis=1)

# Streamlit page config
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom dark theme
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3, h4, h5, h6 {
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #161A23;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("🏥 Breast Cancer Predictor")

st.write("""
This application predicts whether a breast tumor is:

- **Benign (Non-Cancerous)**
- **Malignant (Cancerous)**

using Machine Learning and symptom analysis.
""")

# Sidebar
st.sidebar.header("Cell Nuclei Measurements")

# User inputs
input_data = {}

# First 10 features as sliders
for column in features.columns[:10]:
    min_value = float(features[column].min())
    max_value = float(features[column].max())
    mean_value = float(features[column].mean())
    
    input_data[column] = st.sidebar.slider(
        label=column,
        min_value=min_value,
        max_value=max_value,
        value=mean_value
    )

# Remaining features use mean values
for column in features.columns[10:]:
    input_data[column] = float(features[column].mean())

# Symptom Section
st.sidebar.header("Patient Symptoms")

lump = st.sidebar.checkbox("Breast Lump")
pain = st.sidebar.checkbox("Breast Pain")
swelling = st.sidebar.checkbox("Swelling")
skin_irritation = st.sidebar.checkbox("Skin Irritation")
nipple_discharge = st.sidebar.checkbox("Nipple Discharge")
redness = st.sidebar.checkbox("Redness")
fatigue = st.sidebar.checkbox("Fatigue")
weight_loss = st.sidebar.checkbox("Weight Loss")
family_history = st.sidebar.checkbox("Family History")
lymph_nodes = st.sidebar.checkbox("Swollen Lymph Nodes")

# Symptom score
symptom_score = sum([
    lump,
    pain,
    swelling,
    skin_irritation,
    nipple_discharge,
    redness,
    fatigue,
    weight_loss,
    family_history,
    lymph_nodes
])

# Convert input into Dataframe
input_df = pd.DataFrame([input_data])

# Layout
col1, col2 = st.columns([3, 1])

# Radar chart section
with col1:
    st.subheader("📊 Cell Measurements Radar Chart")
    
    categories = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension"
    ]
    
    values = [
        input_data[features.columns[0]],
        input_data[features.columns[1]],
        input_data[features.columns[2]],
        input_data[features.columns[3]],
        input_data[features.columns[4]],
        input_data[features.columns[5]],
        input_data[features.columns[6]],
        input_data[features.columns[7]],
        input_data[features.columns[8]],
        input_data[features.columns[9]]
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Measurements'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        showlegend=False,
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Prediction section
with col2:
    st.subheader("🧠 Prediction Result")
    
    # Analyze button
    if st.button("🔍 Analyze System"):
        
        # Prediction
        prediction = model.predict(input_df)[0]
        
        # Prediction probability
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Tumor prediction
        if prediction == 0:
            st.success("✅ Benign Tumor")
        else:
            st.error("⚠️ Malignant Tumor")
            
        # Probability
        benign_prob = prediction_proba[0]
        malignant_prob = prediction_proba[1]
        
        st.write("### Prediction Probability")
        
        st.write(f"**Benign:** {benign_prob:.2%}")
        st.progress(float(benign_prob))
        
        st.write(f"**Malignant:** {malignant_prob:.2%}")
        st.progress(float(malignant_prob))
        
        # Symptom analysis
        st.write("### Symptom Analysis")
        
        if symptom_score <= 2:
            st.success("🟢 Low Risk Condition")
            st.write("""
            Current condition appears less harmful.
            
            ### Suggestions:
            - Maintain healthy lifestyle
            - Regular medical checkups
            - Monitor symptoms carefully
            """)
        elif symptom_score <= 5:
            st.warning("🟠 Moderate Risk Condition")
            st.write("""
            Symptoms indicate moderate risk.
            
            ### Suggestions:
            - Consult doctor
            - Take proper treatment
            - Perform screening tests
            """)
        else:
            st.error("🔴 High Risk Condition")
            st.write("""
            Symptoms may indicate serious condition.
            
            ### Immediate Actions:
            - Visit oncologist immediately
            - Take diagnostic tests
            - Start treatment quickly
            """)
            
        # Risk percentage
        risk_percent = (symptom_score / 10) * 100
        
        st.write("### Risk Percentage")
        st.progress(int(risk_percent))
        st.write(f"**Risk Level:** {risk_percent:.0f}%")
        
        st.info("""
        This app can assist medical professionals, but should not replace clinical diagnosis.
        """)

# Footer
st.markdown("""
<hr style="margin-top:40px;">
<p style='text-align: center; font-size: 12px; color: gray;'>
Developed by Shaik mohammed ali
</p>
""", unsafe_allow_html=True)