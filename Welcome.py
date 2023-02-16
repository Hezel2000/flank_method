#!/usr/bin/env python
# coding: utf-8

import streamlit as st

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

# initilising parameters
st.session_state.dfRaw = None
st.session_state.dfMoessNames = None
st.session_state.dfSampleNames = None
st.session_state.dfFitData = None
st.session_state.dfMain = None
# st.session_state.stdSelection = None
st.session_state.fitParametersTAP2 = None
st.session_state.fitParametersTAP4 = None

st.subheader("Welcome to Flank Data Reduction")

col1, col2 = st.columns([4, 1])
with col1:
    st.write('''This online tool allows to calculate Fe3+/FeT abundances from an electron microprobe (EPMA) file. 
    Check out the Flank Method Documentation [website](https://hezel2000.quarto.pub/flank-method-documentation/) 
    for in-depth information about the method, how to prepare the EPMA file, video tutorials how to use this program, the code documentation, literature, a method 
    section for publications, sample files to test this tool, ... – and everything else related to this tool or the flank method.''')
    st.write(
        '''Each page has boxes on the sidebar with information and instructions for the according page – see the example box in the sidebar of this page''')
    st.write(
        '''If you have any question or found a bug, contact me: dominik.hezel-at-em.uni-frankfurt.de''')
    st.write(
        ''':blue[**Known problem** The app occasionally drops the data (This might happen in short or long intervals). This requires a start from the beginning.
        This can be annoying and will be resolved, as soon as the program, documentation webpage and publication are done. So it might take a little.]''')

with col2:
    st.image('flank method documentation/images/flank-method-logo.png')

with st.sidebar:
    with st.expander("Example Info Box"):
        st.write("""
        Such boxes on each page contain information about how to use the according page.
        """)

st.markdown(':red[**This is an entirely new (16.02.2023) version, with a lot of technical changes! Please let me know asap if something is not working properly**]')
