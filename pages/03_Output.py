import streamlit as st


def outputForm():
    import streamlit as st
    import pandas as pd

    st.subheader('Samples Result Table')

    st.session_state.smp_output_file_download = st.session_state.smp_output_file.drop(
        columns=['Point Nr.', 'index', 'Current (nA)', 'Moessbauer', r'$\Delta$ Meas - Moess (2TAPL)',  r'$\Delta$ Meas - Moess (4TAPL)', r"L$\beta$ (TAP2)", r"L$\alpha$ (TAP2)", r"L$\beta$ (TAP4)", r"L$\alpha$ (TAP4)", r"Current (nA)", r"L$\beta$/L$\alpha$ (TAP2)", r"L$\beta$/L$\alpha$ (TAP4)"]).round(4)
    st.dataframe(st.session_state.smp_output_file_download)

    csv = st.session_state.smp_output_file_download.to_csv().encode('utf-8')
    st.download_button(
        label="Download flank measurement sample output file as .csv",
        data=csv,
        file_name='flank measurement sample output file.csv',
        mime='text/csv',
    )

    st.subheader('Standards Result Table')

    st.session_state.std_output_file_download = st.session_state.std_output_file.round(
        4)
    st.dataframe(st.session_state.std_output_file_download)

    csv = st.session_state.std_output_file_download.to_csv().encode('utf-8')
    st.download_button(
        label="Download flank measurement standard output file as .csv",
        data=csv,
        file_name='flank measurementstandard output file.csv',
        mime='text/csv',
    )

# ----------------------------------
# ------------ Side Bar ------------
# ----------------------------------


with st.sidebar:
    with st.expander("Instructions for this site"):
        st.write("""
            See previews of output files and download these with one click on the respective buttons.
        """)

with st.sidebar:
    with st.expander("Interactive Tables"):
        st.write("""
            Clicking on a column header will sort the table according to this column. All tables are searchable, using cmd+F (Mac) or ctrl+F (Windows).
        """)


if st.session_state.dfFitData is not None:
    outputForm()
else:
    st.write("Please upload data in 'Data Upload and Reduction'")
