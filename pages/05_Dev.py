import streamlit as st
st.session_state.dfMoess[st.session_state.dfMoess['Name'].str.contains(
        'AlmO')]
st.session_state.dfMoess[st.session_state.dfMoess['Name'].str.contains(
        'AND1')]
st.session_state.dfMoess
st.session_state

# st.dataframe(st.session_state.dfMain)
# st.write(st.session_state.seltmp)