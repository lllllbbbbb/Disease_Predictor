import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import Normalizer

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
    voting_l = load('voting_liver.joblib')
    df_liver = pd.read_parquet('ckd_total.parquet')
    scaler_liver = load('scaler_liver.joblib')
    return(voting_l,df_liver, scaler_liver)
voting_l, df_liver, scaler_liver = load_data()

if "sex" not in st.session_state:
    st.session_state.sex = "Male"
#
st.title('Liver Disease Predictor')
st.header('Please enter your medical data')
col0, col1, col2 = st.columns(3)
with col0:
    ag = st.number_input(f"**Age (years)**", value=45, min_value=0)
    sx = st.radio(f"**Sex** ",["Male","Female"],key="sex")
    tb = st.number_input(f"**Total Bilirubin (mg/dl)**", value=1.0, min_value=0.0)
    #db = st.number_input(f"**Direct_Bilirubin ", value=0.3, min_value=0.0)


with col1:
    ap = st.number_input(f"**Alkaline Phosphatase (UI/l)**", value=208.0, min_value=0.0)
    ala = st.number_input(f"**Alamine Aminotransferase (UI/l)**",value=35.0, min_value=0.0)
    #asa = st.number_input(f"**Aspartate_Aminotransferase ",value=41, min_value=0)
    #tp = st.number_input(f"**Total_Protiens ",value=6.6, min_value=0.0)
    #al = st.number_input(f"**Albumin ",value=3.1, min_value=0.0)
    agr = st.number_input(f"**Albumin and Globulin Ratio** ",value=0.95, min_value=0.0)
st.divider()
if st.button(r"$\textsf{\Huge Discover the risk}$"):
    if st.session_state.sex == "Male":
        sx = 1
    else:
        sx = 0
    scaled_udf = scaler_liver.transform(np.array([[ag,sx,tb,ap,ala,agr]]))
    outcome = voting_l.predict(scaled_udf)
    certainity = voting_l.predict_proba(scaled_udf)
    if outcome[0] == 0:
        st.success("You might not have the liver disease.")
        st.write("Certainity :",round(certainity[0][0]*100,2),"%")

    else:
        st.error("You might have the liver disease. Please contact your general practitioner.")
        st.write("Certainity :",round(certainity[0][1]*100,2),"%")
st.info("The predictions on this site are provided for information purposes only. The content is in no way intended as a substitute for a medical examination, consultation, diagnosis or treatment")
