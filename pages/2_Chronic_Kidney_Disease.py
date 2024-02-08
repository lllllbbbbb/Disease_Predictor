import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import RobustScaler

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
    voting_d = load('voting_ckd.joblib')
    df_ckd = pd.read_parquet('ckd_total.parquet')
    scaler_ckd = load('scaler_ckd.joblib')
    return(voting_d,df_ckd, scaler_ckd)
voting_d, df_ckd, scaler_ckd = load_data()

#
st.title('Chronic Kidney Disease Predictor')
st.header('Please enter your medical data')
c = st.container()
col0, col1, col2, col3 = st.columns(4)
with col0:
    al = st.radio(f"**Albumine Level**", [0,1,2,3,4,5], 0,horizontal=True)
#    su = st.radio(f"**Sugar Levels**", [0,1,2,3,4,5], 0)
    rbc = st.radio(f"**Red Blood Cells Count**",["normal", "abnormal"], index=0)
    if rbc == "normal":
        rbc = 0
    else :
        rbc = 1
#    pc = st.radio(f"**Puss Cells in Blood**",["normal", "abnormal"], index=0)
#    if pc == "normal":
#        pc = 0
#    else :
#        pc = 1
#    pcc = st.radio(f"**Puss Cells in Urine**",["not present", "present"],index=0)
#    if pcc == "present":
#        pcc = 1
#    else :
#        pcc = 0

#    ba = st.radio(f"**Presence of Bacteria**",["not present", "present"],index=0)
#    if ba == "present":
#        ba = 1
#    else :
#        ba = 0
    htn = st.radio(f"**Hypertension**", ["no","yes"],index=0)
    if htn == "no":
        htn = 0
    else :
        htn = 1
    dm = st.radio(f"**Diabetes Mellitus**",["no","yes"],index=0)
    if dm == "no":
        dm = 0
    else :
        dm = 1

with col1:

#    cad = st.radio(f"**Coronary Arterial Disease**",["no","yes"],index=0)
#    if cad == "no":
#        cad = 0
#    else :
#        cad = 1
#    appet = st.radio(f"**Appetite**",["good","poor"],index=0)
#    if appet == "good":
#        appet = 0
#    else:
#        appet = 1
#    pe = st.radio(f"**Peripheral Edema**",["no","yes"],index=0)
#    if pe == "no":
#        pe = 0
#    else:
#        pe = 1
#    ane = st.radio(f"**Anemia**",["no","yes"],index=0)
#    if ane == "no":
#        ane = 0
#    else:
#        ane = 1
#    age = st.number_input(f"**Age**",value=54, min_value=0)
    bp = st.number_input(f"**Systolic Blood Pressure**", value=80, min_value=0)
    sg = st.number_input(f"**Specific Gravity**",value=1.02, min_value=0.0)
    bgr = st.number_input(f"**Blood Glucose Random**",value=119, min_value=0)
    bu = st.number_input(f"**Blood Urea**",value=41, min_value=0)


with col2:

#    sc = st.number_input(f"**Seric Creatinine**",value=1.2, min_value=0.0)
    sod = st.number_input(f"**Sodium Levels**",value=139, min_value=0)
#    pot = st.number_input(f"**Potassium Levels**",value=4.4, min_value=0.0)
    hemo = st.number_input(f"**Hemoglobin Levels**",value=13.5, min_value=0.0)
    pcv = st.number_input(f"**Packed Cells Volume**",value=41, min_value=0)
    wc = st.number_input(f"**White Blood Cells Count**",value=8000, min_value=0)
#    rc = st.number_input(f"**Red Blood Cells Count**",value= 4.76, min_value=0.0)

#st.write(np.array([[al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane,age,bp,sg,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc]]))
st.divider()
if st.button(r"$\textsf{\Huge Discover the risk}$"):
    scaled_udf = scaler_ckd.transform(np.array([[al,rbc,htn,dm,bp,sg,bgr,sod,hemo,pcv,wc]]))
    outcome = voting_d.predict(scaled_udf)
    certainity = voting_d.predict_proba(scaled_udf)
    if outcome[0] == 0:
        st.success("You might not have the chronic kidney disease.")
        st.write("Certainity :",round(certainity[0][0]*100,2),"%")

    else:
        st.error("You might have the chronic kidney disease. Please contact your general practitioner.")
        st.write("Certainity :",round(certainity[0][1]*100,2),"%")


st.info("The predictions on this site are provided for information purposes only. The content is in no way intended as a substitute for a medical examination, consultation, diagnosis or treatment")
