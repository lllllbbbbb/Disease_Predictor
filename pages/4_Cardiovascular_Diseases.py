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
    voting_hd = load('voting_hd.joblib')
    df_hd = pd.read_parquet('df_hd.parquet')
    scaler_hd = load('scaler_hd.joblib')
    return(voting_hd,df_hd, scaler_hd)
voting_hd, df_hd, scaler_hd = load_data()

if "sex" not in st.session_state:
    st.session_state.sex = "Male"
#
st.title('Cardiovascular Diseases Predictor')
st.header('Please enter your medical data')
col0, col1, col2 = st.columns(3)

with col0:
    ag = st.number_input(f"**Age (years)**", value=55, min_value=0)
    sx = st.radio(f"**Sex** ",["Male","Female"],key="sex")
    #cpt = st.number_input(f"**Chest Pain Type ", 1, min_value=0, max_value=3)
    cpt = st.radio(f"**Chest Pain Type** ",[0,1,2,3], index=1)
    rbp = st.number_input(f"**Resting Blood Pressure (mmHg)**", value=130, min_value=0)
    sc = st.number_input(f"**Serum Cholesterol (mg/dL)**", value=240.5, min_value=0.0)
    #fbs = st.number_input(f"**Glycémie à Jeun ", 0)
    fbs = st.radio(f"**Glycémie à Jeun** ",[0,1])
    #rer = st.number_input(f"**Resting Electrocardiographic Results ", 1)
    rer = st.radio(f"**Resting Electrocardiographic Results** ", [0,1,2], index=1)


with col1:
    mhr = st.number_input(f"**Maximum Heart Rate Achieved (bpm)**", value=152.5, min_value=0.0)
    #eia = st.number_input(f"**Exercise Induced Angina ",0)
    eia = st.radio(f"**Exercise Induced Angina**",[0,1])
    stdie = st.number_input(f"**ST Depression Induced by Exercise (mm)**",value=0.8, min_value=0.0)
    #spest = st.number_input(f"**Slope of the Peak Exercise ST Segment ",1)
    spest = st.radio(f"**Slope of the Peak Exercise ST Segment** ",[0,1,2],index=1)
    #mvcf = st.number_input(f"**Number of Major Vessels Colored by Fluoroscopy ",0)
    mvcf = st.radio(f"**Number of Major Vessels Colored by Fluoroscopy** ",[0,1,2,3,4])
    #th = st.number_input(f"**Thalassemia ",2)
    th = st.radio(f"**Thalassemia** ",[0,1,2,3],index=2)

st.divider()
if st.button(r"$\textsf{\Huge Discover the risk}$"):
    if st.session_state.sex == "Male":
        sx = 1
    else:
        sx = 0
    scaled_udf = scaler_hd.transform(np.array([[ag,sx,cpt,rbp,sc,fbs,rer,mhr,eia,stdie,spest,mvcf,th]]))
    outcome = voting_hd.predict(scaled_udf)
    certainity = voting_hd.predict_proba(scaled_udf)
    if outcome[0] == 0:
        st.success("You might not have a cardiovascular disease.")
        st.write("Certainity :",round(certainity[0][0]*100,2),"%")

    else:
        st.error("You might have a cardiovascular disease. Please contact your general practitioner.")
        st.write("Certainity :",round(certainity[0][1]*100,2),"%")


st.info("The predictions on this site are provided for information purposes only. The content is in no way intended as a substitute for a medical examination, consultation, diagnosis or treatment.")
