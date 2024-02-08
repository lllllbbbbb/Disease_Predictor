import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import RobustScaler
from streamlit_extras.let_it_rain import rain

with st.sidebar:
  st.image("logo_t.png")

page_bg_img = """
<style>
[data-testid=stAppViewContainer]{
 background: rgb(255,255,255);
background: linear-gradient(90deg, rgba(255,255,255,1) 0%, rgba(232, 236, 241,1) 100%);
}
[data-testid=stHeader]{
background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #31333f;
    color:#FFFFFF;
}
div.stButton > button:hover {
    background-color: #FFFFFF;
    color:#000000;
    }
</style>""", unsafe_allow_html=True)
# imports
@st.cache_data
def load_data():
    voting = load('voting_diabete.joblib')
    df = pd.read_parquet('diabete.parquet')
    scaler = load('scaler.joblib')
    return(voting,df, scaler)
voting, df, scaler = load_data()
#
st.title('Diabetes Predictor')
st.header('Please enter your medical data')
col0, col1, col2 = st.columns([1,1,3])
with col0:
    pg = st.number_input(f"**Number of Pregnancies** ", value=0, step=1, min_value=0)
    gl = st.number_input(f"**Glucose Levels (mg/dl)**", value=100, min_value=0)

    bp = st.number_input(f"**Blood Pressure (mmHg)**", value=70, min_value=0)
    il = st.number_input(f"**Insulin levels (µIU/mL)**", value=100, min_value=0)
with col1:
    age = st.number_input(f"**Age (years)**", value=35, step=1, min_value=0)

    bmi = st.number_input(f"**BMI** ", value=20, min_value=0)
    dpf = st.number_input(f"**Diabetes Pedigree Function** ", value=0.1, min_value=0.0)

def create_dataframe(pg,gl,bp,il,bmi,dpf,age):
    data=[pg,gl,bp,il,bmi,dpf,age]
    udf = pd.DataFrame([data], columns=["Number of Pregnancies ","Glucose Levels ","Blood Pressure ","Insulin levels ","BMI ","Diabetes Pedigree Function ","Age "])
    return udf
st.divider()
if st.button(r"$\textsf{\Huge Discover the risk}$"):
    udf = create_dataframe(pg,gl,bp,il,bmi,dpf,age)
    scaled_udf = scaler.transform(np.array([[pg,gl,bp,il,bmi,dpf,age]]))
    outcome = voting.predict(scaled_udf)
    certainity = voting.predict_proba(scaled_udf)
    if outcome[0] == 0:
        st.success("You might not be diabetic.")
        st.write("Certainity :",round(certainity[0][0]*100,2),"%")
    else:
        st.error("You might be diabetic. Please contact your general practitioner.")
        st.write("Certainity :",round(certainity[0][1]*100,2),"%")


st.info("The predictions on this site are provided for information purposes only. The content is in no way intended as a substitute for a medical examination, consultation, diagnosis or treatment")
