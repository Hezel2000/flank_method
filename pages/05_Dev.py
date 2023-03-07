import streamlit as st

st.write('tet')


def dev():
    import streamlit as st
    import pandas as pd

    st.dataframe(dfFitData)

    st.write('resultsFe3Smp')
    st.session_state.resultsFe3Smp

    st.write('resultsFe3Std')
    st.session_state.resultsFe3Std

    # st.write('smpAndstdList')
    # st.session_state.smpAndstdList

    st.write('Fe3SmpAndFe3Std_2')
    fil = st.session_state.resultsFe3Smp['Name'].isin(
        st.session_state.dfMoessNames)
    st.session_state.Fe3SmpAndFe3Std_2 = pd.concat([
        st.session_state.resultsFe3Smp[~fil], st.session_state.resultsFe3Std])
    st.session_state.Fe3SmpAndFe3Std_2

    st.write('dfMain')
    st.session_state.dfMain


dev
