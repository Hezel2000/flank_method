import streamlit as st
f1=st.session_state.dfMoess[st.session_state.dfMoess['Name'].str.contains(
        'AlmO')]
st.write(f1['Fe2+/SumFe'].tolist()[0])
f2=st.session_state.dfMoess[st.session_state.dfMoess['Name'].str.contains(
        'AND1')]
st.write(f2['Fe2+/SumFe'].tolist()[0])
st.session_state.dfMoess
st.session_state

# st.dataframe(st.session_state.dfMain)
# st.write(st.session_state.seltmp)