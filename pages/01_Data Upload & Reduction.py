import streamlit as st
import pandas as pd
from utils import dataUpload, importMoessStdFile, prepareDataset, subsetsOfDatasets, preProcessingData, calcFullOutputFile, calcRegressionsAndProduceResults

# initilising parameters
st.session_state.dfRaw = None
st.session_state.dfRaw_input = None
st.session_state.dfMoess = None
st.session_state.dfMoessNames = None
st.session_state.dfSampleNames = None
st.session_state.dfFitData = None
st.session_state.dfMain = None
# st.session_state.stdSelection = None
st.session_state.fitParametersTAP2 = None
st.session_state.fitParametersTAP4 = None


with st.sidebar:
    with st.expander("Instructions for data upload"):
        st.write("""
            The uploaded file needs to have specific structure. For instructions how to prepare your initial file and avoid pitfals while doing so, you will
            find the required documentation [here](https://hezel2000.quarto.pub/flank-method-documentation/).""")

with st.sidebar:
    with st.expander("Instructions for data reduction"):
        st.write("""
            Choose wheter all or only your inspected data shall be reduced. Then select the standards used to claculte the fit parameters.
            Note that you can only choose those also present in the Moessbauer standard data file. Click 'Calculate Results'.
            You can change your selection of standards or whether to use all or only the inspected data anytime.
            However, after each change you need to click 'Calculate Results' again.
        """)


#st.session_state.dfMoess = importMoessStdFile(toggle_own_Moess_file)
dataUpload()
importMoessStdFile()

if st.session_state.dfRaw is not None:
    st.session_state.AllInsp = st.radio(
        'Choose whether all data or only inspected data will be used', ('All', 'Inspected'), horizontal=True)

    # The following commands need to be run (although these need to run again below), as these produce the allSmpNames variable, which is requried in the following multiselect
    prepareDataset(st.session_state.AllInsp)
    subsetsOfDatasets()

    allSmpNames = st.session_state.dfMoessNames   # st.session_state.dfSampleNames
    st.session_state.stdSelection = st.multiselect(
        'Select standards used to calculate the Fit Parameters', allSmpNames, allSmpNames[:4])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.drMonitorName = st.selectbox(
            'Select the standard used as drift monitor', st.session_state.dfMoess['Name'])
    with col2:
        moessCategories = st.session_state.dfMoess.columns.tolist()
        moessSelOpt = [moessCategories[i] for i in (2, 4, 6)]
        st.session_state.selMoessData = st.selectbox(
            'Select the Moessbauer values to be used', moessSelOpt)
    with col3:
        st.session_state.nr_of_samples = st.number_input(
            'Number of random samples used for calculations (0 = all)', value=0)

    if st.button('Calculate Results'):
        prepareDataset(st.session_state.AllInsp)
        subsetsOfDatasets()
        preProcessingData(st.session_state.nr_of_samples)
        calcRegressionsAndProduceResults(st.session_state.selMoessData)
        calcFullOutputFile(st.session_state.nr_of_samples)
        st.markdown(
            '<p style="color:green"><b>Flank data successfully reduced!</b></p>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:green'><b>Continue on the sidebar with 'Data Inspection' or directly go to 'Output'</b></p>", unsafe_allow_html=True)

    if st.session_state.dfFitData is not None:
        st.markdown('<h4 style="color:blue"><b>Your Selected Standards Used for Fitting</b> </h3>',
                    unsafe_allow_html=True)
        st.write(st.session_state.dfFitData.round(3))

        st.markdown(
            '<h4 style="color:blue"><b>Calculated Fit Parameters for 2TAPL & 4TAPL</b> </h4>', unsafe_allow_html=True)
        st.write(pd.DataFrame({'Parameter': ['A', 'B', 'C', 'D'],
                               'TAP2': st.session_state.fitParametersTAP2,
                               'TAP4': st.session_state.fitParametersTAP4}))

        st.markdown('<h4 style="color:green"><b>Formula to calculate Fe3+</b> </h4>',
                    unsafe_allow_html=True)
        st.write(f'$Fe^{{3+}}$ = -{round(st.session_state.fitParametersTAP2[0],2)} - {round(st.session_state.fitParametersTAP2[1],2)} $\\times$ $\\frac{{L\\beta}}{{L\\alpha}}$ - {round(st.session_state.fitParametersTAP2[2],2)} $\\times$ $\Sigma$Fe - {round(st.session_state.fitParametersTAP2[3],2)} $\\times$ $\Sigma$Fe $\\times$ $\\frac{{L\\beta}}{{L\\alpha}}$ + $\Sigma$Fe')
        st.write(f'$Fe^{{3+}}$ = -{round(st.session_state.fitParametersTAP4[0],2)} - {round(st.session_state.fitParametersTAP4[1],2)} $\\times$ $\\frac{{L\\beta}}{{L\\alpha}}$ - {round(st.session_state.fitParametersTAP4[2],2)} $\\times$ $\Sigma$Fe - {round(st.session_state.fitParametersTAP4[3],2)} $\\times$ $\Sigma$Fe $\\times$ $\\frac{{L\\beta}}{{L\\alpha}}$ + $\Sigma$Fe')
        st.write('The result is $Fe^{3+}$ in wt%.')