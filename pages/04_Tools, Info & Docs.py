def tools_info():
    import streamlit as st

    toolSel = st.sidebar.radio(
        'Select Tool/Info', ('Tutorials & Documentations', 'Crystal Positioning', 'Result Tables', 'Check Data Integrity', 'Calculate individual Fe2+ & Fe3+', 'Downloads'))

    if toolSel == 'Tutorials & Documentations':
        tutorials_instructions()
    elif toolSel == 'Crystal Positioning':
        crystalPositioning()
    elif toolSel == 'Result Tables':
        resultTables()
    elif toolSel == 'Check Data Integrity':
        checkDataIntegrityPage()
    elif toolSel == 'Calculate individual Fe2+ & Fe3+':
        individualFe3Fe2Calculation()
    elif toolSel == 'Downloads':
        downloads()


# ------------ Start Crystal Positioning

def crystalPositioning():
    import streamlit as st
    import pandas as pd
    from bokeh.plotting import figure
    from bokeh.models import Span
    # from bokeh.models import Range1d

    st.subheader('Determining the flank positions from difference spectra')

    st.session_state.FeSpectra = 0
    uploaded_FeSpectra = st.file_uploader('')
    if uploaded_FeSpectra is not None:
        #st.session_state.FeSpectra = pd.read_csv(uploaded_FeSpectra)
        st.session_state.FeSpectra = pd.read_csv(uploaded_FeSpectra, sep=";|,", engine="python")
    

    if st.session_state.FeSpectra is 0:
        st.write('No file loaded.')
    else:
        if st.session_state.FeSpectra.columns.tolist()[0] == 'Unnamed: 0':
            st.session_state.FeSpectra.drop(st.session_state.FeSpectra.columns[0], axis=1, inplace=True)
        crystal = st.selectbox('Select crystal', ['2TAPL', '4TAPL'])
        col1, col2 = st.columns(2)
        with col1:
            Lb_flank_pos = st.number_input(
                'L-value for Lb', value=187.631)
        with col2:
            La_flank_pos = st.number_input(
                'L-value for La', value=191.218)

        df_closest_Lb_Fe3poor = st.session_state.FeSpectra.iloc[(
            st.session_state.FeSpectra['L-value'] - Lb_flank_pos).abs().argsort()[:1]]['Fe3+ poor - ' + crystal].values[0]
        df_closest_La_Fe3poor = st.session_state.FeSpectra.iloc[(
            st.session_state.FeSpectra['L-value'] - La_flank_pos).abs().argsort()[:1]]['Fe3+ poor - ' + crystal].values[0]

        df_closest_Lb_Fe3rich = st.session_state.FeSpectra.iloc[(
            st.session_state.FeSpectra['L-value'] - Lb_flank_pos).abs().argsort()[:1]]['Fe3+ rich - ' + crystal].values[0]
        df_closest_La_Fe3rich = st.session_state.FeSpectra.iloc[(
            st.session_state.FeSpectra['L-value'] - La_flank_pos).abs().argsort()[:1]]['Fe3+ rich - ' + crystal].values[0]

        st.write('Fe3+ poor Lb/La ratio: ',
                 round(df_closest_Lb_Fe3poor / df_closest_La_Fe3poor, 2))
        st.write('Fe3+ rich Lb/La ratio: ',
                 round(df_closest_Lb_Fe3rich / df_closest_La_Fe3rich, 2))

        fig = figure(width=600, height=400)
        fig.line(st.session_state.FeSpectra['L-value'], st.session_state.FeSpectra['Fe3+ poor - ' + crystal],
                 color='green', legend_label='Fe3+ poor (' + crystal + ')')
        fig.line(st.session_state.FeSpectra['L-value'], st.session_state.FeSpectra['Fe3+ rich - ' + crystal],
                 color='blue', legend_label='Fe3+ rich (' + crystal + ')')
        fig.line(st.session_state.FeSpectra['L-value'], st.session_state.FeSpectra['Fe3+ rich - ' + crystal] -
                 st.session_state.FeSpectra['Fe3+ poor - ' + crystal], color='orange', legend_label='difference spectrum')
        vline_Lb = Span(location=Lb_flank_pos, dimension='height',
                        line_color='grey', line_dash='dashed', line_width=2)
        vline_La = Span(location=La_flank_pos, dimension='height',
                        line_color='grey', line_dash='dashed', line_width=2)
        fig.renderers.extend([vline_Lb, vline_La])
        fig.xaxis.axis_label = 'L-value (mm)'
        fig.yaxis.axis_label = 'counts'
        # fig.y_range = Range1d(-500, 1300)
        fig.add_layout(fig.legend[0], 'below')

        st.bokeh_chart(fig)

# ------------ End Crystal Positioning

# ------------ Start Check Data Integrity

def checkDataIntegrityPage():
    import streamlit as st
    import pandas as pd

    def checkDataIntegrity(df_test):
        import re

        value_is_zero = []
        value_is_negative = []
        string_to_value = []
        value_is_string = []
        value_is_empty = []
        value_is_other = []

        for i1 in range(3, df_test.shape[1]):
            for i2 in range(df_test.shape[0]):
                val = df_test.iloc[i2, i1]

                if pd.isna(val):
                    value_is_empty.append([df_test.columns[i1], i2 + 1])
                elif isinstance(val, (float, int)):
                    if val == 0:
                        value_is_zero.append([df_test.columns[i1], i2 + 1])
                    elif val < 0:
                        value_is_negative.append(
                            [df_test.columns[i1], i2 + 1])
                elif isinstance(val, str):
                    if bool(re.match(r'^[\d.+-]+$', val)):
                        val_tmp = float(val)
                        df.iloc[i2, i1] = val_tmp
                        string_to_value.append(
                            [df_test.columns[i1], i2 + 1])
                        if val_tmp == 0:
                            value_is_zero.append(
                                [df_test.columns[i1], i2 + 1])
                        elif val_tmp < 0:
                            value_is_negative.append(
                                [df_test.columns[i1], i2 + 1])
                    else:
                        # res = 'string'
                        value_is_string.append(
                            [df_test.columns[i1], i2 + 1])
                else:
                    value_is_other.append([df_test.columns[i1], i2 + 1])

        return [['The following entries have the value 0 – you might want to delete these in the original file:', value_is_zero],
                ['The following entries have a negative value – you might want to delete these in the original file:', value_is_negative],
                ['The following entries were transformed from strings into numbers:',
                    string_to_value],
                ['The following entries are strings that do not contain a numeric value – you need to delete these in the original file:', value_is_string],
                ['The follwoing entries are not a number, most likely empty entries:', value_is_empty],
                ['The following entries contain neither a string nor a numeric value  – you need to delete these in the original file:', value_is_other]]

    st.subheader(
        '2  Check Data Integrity')
    st.write('This is not required, but strongly recommended when using the dataset for the first time to identify any problems that some data might subsequently cause.')
    if st.button('Check Data Integrity'):
        dfImpCatList = st.session_state.dfRaw.columns.tolist()

        # testing for duplicates
        st.markdown(
            '<h5 style="color:rgb(105, 105, 105)">Duplicate test</h5>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:rgb(105, 105, 105)">This tests whether duplicate point names exist</p>', unsafe_allow_html=True)
        duplicate_point_names = st.session_state.dfRaw['Comment'][st.session_state.dfRaw['Comment'].duplicated(
            keep=False)]
        if len(duplicate_point_names) == 0:
            st.markdown(
                '<p style="color:green">no duplicate point names in the uploaded file</p>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:red">the following point names occure multiple times in the uploaded file:</p>', unsafe_allow_html=True)
            duplicate_point_names

        # required categoires test
        st.markdown(
            '<h5 style="color:rgb(105, 105, 105)">Checking Required Categories</h5>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:rgb(105, 105, 105)">This tests whether all required categories are present in the uploaded file</p>', unsafe_allow_html=True)

        req_cat_list = ['Point', 'Comment', 'Inspected',
                        'Bi(Net)', 'Ar(Net)', 'Br(Net)', 'As(Net)', 'Current']

        test_req_cat = []
        for i in req_cat_list:
            if i not in dfImpCatList:
                test_req_cat.append(i)
        if len(test_req_cat) == 0:
            st.markdown(
                '<p style="color:green">all required categories in input file</p>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:red">the following categories are missing in the input file</p>', unsafe_allow_html=True)
            st.write(test_req_cat)

        # check for unusual data entries
        st.markdown(
            '<h5 style="color:rgb(105, 105, 105)">Checking Data Integrity</h5>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:rgb(105, 105, 105)">The following are various checks for unususal data entries, which might be corrected before proceeding</p>', unsafe_allow_html=True)

        checkDataIntegrity(st.session_state.dfMain)

        data_integrity_results = checkDataIntegrity(st.session_state.dfMain)

        for i in data_integrity_results:
            res_string = i[0]
            res_data = i[1]

            st.markdown(
                f'<h5 style="color:rgb(105, 105, 105)">{res_string}</h5>', unsafe_allow_html=True)
            t = []
            if len(res_data) > 0:
                res_tmp = pd.DataFrame(res_data)
                tmp = res_tmp[0].drop_duplicates()

                for res_data in tmp:
                    t.append(
                        [res_data, res_tmp[res_tmp[0] == res_data][1].tolist()])

                st.write('1: for the following categories: ')
                st.write(str([row[0] for row in t]))
                st.write('')
                st.write('2: specifically: ')

                for i2 in t:
                    st.write('for ' + i2[0] +
                             ' the entires for the following points:')
                    st.write(i2[1])

            else:
                st.write('none')
            st.write('---------------------------------')
        st.markdown(
            '<h5 style="color:green">Check Finished!</h5>', unsafe_allow_html=True)
# ------------ End Check Data Integrity

# ------------- Start Result Tables

def resultTables():
    import streamlit as st

    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    with st.sidebar:
        with st.expander("Instructions for this site"):
            st.write("""
             The results of the Fe3+ abundances in the standards, the drift monitor, as well as the samples are displayed.
             Each table can be downloaded as a single .csv file if required. More complete output options are provided in 'Output'.
             Once, everything is checked, proceed to 'Data Inspection' – or use this site to come back and check individual values.
         """)

    with st.sidebar:
        with st.expander("Interactive Tables"):
            st.write("""
                Clicking on a column header will sort the table according to this column. All tables are searchable, using cmd+F (Mac) or ctrl+F (Windows).
            """)

    st.subheader('$$Fe^{3+}/ \Sigma Fe$$' + ' in the Standards')
    st.write(
        r'$Fe^{3+}/\Sigma Fe$ deviation from the Moessbauer data should be <0.01-0.015')
    st.dataframe(st.session_state.resultsFe3Std.round(4))
    csv = convert_df(st.session_state.resultsFe3Std)
    st.download_button(
        label="Download standard data as .csv",
        data=csv,
        file_name='Fe3+ of all measured standards.csv',
        mime='text/csv',
    )

    st.subheader('$$Fe^{3+}/ \Sigma Fe$$' + ' in the Drift Monitor')
    st.write(
        r'$Fe^{3+}/\Sigma Fe$ deviation from the Moessbauer data should be <0.01-0.015')
    st.dataframe(st.session_state.resultsFe3Drift.round(4))
    csv = convert_df(st.session_state.resultsFe3Drift)
    st.download_button(
        label="Download drift data as .csv",
        data=csv,
        file_name='Fe3+ of all drift measurements.csv',
        mime='text/csv',
    )

    st.subheader('$$Fe^{3+}/ \Sigma Fe$$' + ' in the Samples')
    st.write(r'The error on $Fe^{3+}/\Sigma Fe$ in the Smp is 0.02')
    st.dataframe(st.session_state.resultsFe3Smp.round(4))
    csv = convert_df(st.session_state.resultsFe3Smp)
    st.download_button(
        label="Download sample data as .csv",
        data=csv,
        file_name='Fe3+ of all measured samples.csv',
        mime='text/csv',
    )

# ------------- Start Result Tables

# ------------ Start Individual Fe3+ & Fe2+ calculation

def individualFe3Fe2Calculation():
    import streamlit as st
    import pandas as pd

    A = st.number_input('Insert parameter A',
                        value=st.session_state.fitParametersTAP2[0])
    B = st.number_input('Insert parameter B',
                        value=st.session_state.fitParametersTAP2[1])
    C = st.number_input('Insert parameter C',
                        value=st.session_state.fitParametersTAP2[2])
    D = st.number_input('Insert parameter D',
                        value=st.session_state.fitParametersTAP2[3])
    fetot = st.number_input('Insert the Fetot of the sample', value=20.6)
    Lb_La = st.number_input('Insert the Lb/La of the sample', value=1.4)

    if st.button('Calculate Fe2+ & Fe3+'):
        st.write('$$Fe2+$$: ' + str(round(A + B * Lb_La +
                                          C * fetot + D * fetot * Lb_La, 3)) + ' wt%')
        st.write('$$Fe3+$$: ' + str(round(-A - B * Lb_La - C *
                                          fetot - D * fetot * Lb_La + fetot, 3)) + ' wt%')

    st.markdown('<h4 style="color:blue"><b>Calculated Fit Parameters for 2TAPL & 4TAPL</b> </h4>',
                unsafe_allow_html=True)
    st.write(pd.DataFrame({'Parameter': ['A', 'B', 'C', 'D'],
                           'TAP2': st.session_state.fitParametersTAP2,
                           'TAP4': st.session_state.fitParametersTAP4}))

    st.markdown('<h4 style="color:green"><b>Formulas to calculate Fe2+ and Fe3+</b> </h4>',
                unsafe_allow_html=True)
    st.latex(r'''Fe^{2+} = A + B \times \frac{L\beta}{L\alpha} + C \times \Sigma Fe + D \times \Sigma Fe \times \frac{L\beta}{L\alpha}''')
    st.latex(r'''Fe^{3+} = -A - B \times \frac{L\beta}{L\alpha} - C \times \Sigma Fe - D \times \Sigma Fe \times \frac{L\beta}{L\alpha} + Fe_{tot}''')
    st.latex(
        r'''\textrm{The result is } Fe^{2+} \textrm{ or } Fe^{3+} \textrm{, respectively, in wt\%} ''')

    with st.sidebar:
        with st.expander("Instructions for this site"):
            st.write("""
        Input/change the parameters to test how the result depends on the respective, individual parameters.
         """)

# ------------ End Individual Fe3+ & Fe2+ calculation

# --------- Start Tutorials & Instructions

def tutorials_instructions():
    import streamlit as st
    st.subheader('Tutorials & Documentations')

    st.markdown(
        '**Comprehensive Tutorials, Documentation, etc. about this flank method reduction program, and everything else regarding the flank method is available here:**')
    documentation_link = '[Flank Method World](https://hezel2000.quarto.pub/flank-method-documentation/)'
    st.markdown(documentation_link, unsafe_allow_html=True)

# --------- End Tutorials & Instructions


# --------- Start Downloads

def downloads():
    import streamlit as st
    st.markdown(
        '''**The Moessbauer standard data file**''')
    st.download_button(
        label="Download Moessbauer data table as .csv",
        data=st.session_state.dfMoess.to_csv().encode('utf-8'),
        file_name='Moessbauer Standard Data.csv',
        mime='text/csv',
    )

# --------- End Downloads


tools_info()
