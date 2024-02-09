import streamlit as st
#st.session_state

# st.dataframe(st.session_state.dfMain)
# st.write(st.session_state.seltmp)

st.session_state.tmp
st.session_state.dfFitData

st.session_state.fitParametersTAP2


st.session_state.dfMain

st.session_state.dfMeasStdDataTAP2
st.session_state.dfMeasSmpDataTAP2

st.session_state.fitParametersTAP2




#st.session_state.rfp

# df

fil1 = st.session_state.dfMain['Name'].str.startswith('dr')


st.session_state.dfSampleNames = st.session_state.dfMain[~fil1].loc[:, 'Name'].drop_duplicates()
st.session_state.dfSampleNames

st.session_state.dfMeasStdDataTAP2
st.session_state.dfMeasSmpDataTAP2

st.session_state.fitParametersTAP2

st.session_state.dfFitData

# for i in st.session_state.dfSampleNames.tolist():
#     res1 = list(map(lambda x: i if x in i else 0,
#                     st.session_state.dfMoess['Name']))
#     res2 = [n for n in res1 if n != 0]
#     if len(res2) == 1:
#         st.session_state.dfMoessNames.append(res2[0])

st.session_state.dfMoessNames
