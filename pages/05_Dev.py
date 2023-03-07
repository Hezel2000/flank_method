def s_states():
    import streamlit as st

    st.dataframe(st.session_state.dfFitData)


s_states()
