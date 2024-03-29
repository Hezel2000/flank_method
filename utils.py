import streamlit as st
import pandas as pd
import numpy as np

def dataUpload():    
    toggle_flank_demo_data = st.toggle("Use demo dataset")
    st.write("... and continue with 'Calculate Results' at the bottom")

    if toggle_flank_demo_data:
        st.session_state.dfRaw_input = st.session_state.FeSpectra = pd.read_csv('data/test data/flank method test dataset.csv')
    else:
        uploaded_file = st.file_uploader('Data file')
        if uploaded_file is not None:
            st.session_state.dfRaw_input = pd.read_csv(
                uploaded_file, sep=";|,", engine="python")

    if st.session_state.dfRaw_input is None:
        st.write('No data file uploaded yet')
    else:
        if st.session_state.dfRaw_input.columns.tolist()[0] == 'Unnamed: 0':
            st.session_state.dfRaw_input.drop(
                st.session_state.dfRaw_input.columns[0], axis=1, inplace=True)

        fil = (st.session_state.dfRaw_input['Inspected'] == 'ignore') | (
            st.session_state.dfRaw_input['Inspected'] == 'Ignore')
        st.session_state.dfRaw = st.session_state.dfRaw_input[~fil]

        st.session_state.sel_ignore = st.radio("Select whether to see the uploaded file with or without 'ignore' in rows", (
            "see without 'ignore'", "see with 'ignore'"), horizontal=True)
        with st.expander('You uploaded the following data for flank reduction'):
            if st.session_state.sel_ignore == "see without 'ignore'":
                st.dataframe(st.session_state.dfRaw)
            else:
                st.dataframe(st.session_state.dfRaw_input)


def importMoessStdFile():
    toggle_own_Moess_file = st.toggle('Upload and use own Moessbauer file', False)

    if toggle_own_Moess_file:
        uploaded_moess_file = st.file_uploader('Optional Moessbauer file')
        if uploaded_moess_file is not None:
            st.session_state.dfMoess = pd.read_csv(
                uploaded_moess_file, sep=";|,", engine="python")
            
        if st.session_state.dfMoess is None:
            st.write('No Moessbauer file uploaded yet.')
        else:
            with st.expander('You uploaded the following Meossbauer standards for flank reduction'):
                st.dataframe(st.session_state.dfMoess)
    else:
        #st.session_state.dfMoess = pd.read_csv('data/test data/moessbauer standard test dataset.csv')
        st.session_state.dfMoess = pd.read_csv('data/moessbauer standard dataset.csv')
        with st.expander('Othwerwise the following standard Moessbauer file on record will be used.'):
                st.dataframe(st.session_state.dfMoess)



# -----------------------------------------#
# ------------ Start Data Reduction -------#
# -----------------------------------------#


# ------------ Start Prepare Dataset
def prepareDataset(sel):
    if sel == 'All':
        dfComplete = st.session_state.dfRaw_input
        sel_column = 'Comment'
        first_3_cat_renamings = {
            'Point': 'Point Nr.', 'Comment': 'Name', 'Inspected': 'Name Inspected'}
    else:
        dfComplete = st.session_state.dfRaw
        sel_column = 'Inspected'
        first_3_cat_renamings = {'Point': 'Point Nr.',
                                 'Comment': 'Name of All', 'Inspected': 'Name'}

    for ind in dfComplete.index:
        measurementPointName = dfComplete[sel_column][ind]
        if 'Grid' in measurementPointName:
            dfComplete[sel_column] = dfComplete[sel_column].replace(
                [measurementPointName], measurementPointName.split('Grid')[0])
        elif 'Line' in measurementPointName:
            dfComplete[sel_column] = dfComplete[sel_column].replace(
                [measurementPointName], measurementPointName.split('Line')[0])

    oxide_with_unit_list = []
    oxide_list = []
    for i in dfComplete.columns.tolist():
        if 'Mass%' in i:
            oxide_with_unit_list.append(i)
            oxide_list.append(i.split('(')[0])
    extracted_categories = ['Point', 'Comment', 'Inspected'] + oxide_with_unit_list + [
        'Bi(Net)', 'Ar(Net)', 'Br(Net)', 'As(Net)', 'Current']
    rename_columns_dict = {**first_3_cat_renamings,
                           **dict(zip(oxide_with_unit_list, oxide_list)),
                           **{'Bi(Net)': r'L$\beta$ (TAP2)', 'Ar(Net)': r'L$\alpha$ (TAP2)',
                              'Br(Net)': r'L$\beta$ (TAP4)', 'As(Net)': r'L$\alpha$ (TAP4)',
                              'Current': 'Current (nA)'}}

    df = dfComplete[extracted_categories].rename(
        columns=rename_columns_dict)

    df = pd.concat([df, df[r'L$\beta$ (TAP2)']/df[r'L$\alpha$ (TAP2)'],
                    df[r'L$\beta$ (TAP4)']/df[r'L$\alpha$ (TAP4)']], axis=1)

    st.session_state.dfMain = df.rename(
        columns={0: r'L$\beta$/L$\alpha$ (TAP2)', 1: r'L$\beta$/L$\alpha$ (TAP4)'})

# ------------ End Prepare Dataset
    
# ------------ Start produce dfdr and dfSampleNames
def subsetsOfDatasets():
    # import pandas as pd
    # import streamlit as st
    # a df with only drift measurements
    # drift measurements will be stored in the DataFrame: dfdr
    fil1 = st.session_state.dfMain['Name'].str.startswith('dr')
    drOnly = st.session_state.dfMain[fil1]

    selectedStandards = ['AlmO']  # ['AlmO', 'UA5', 'UA10', 'Damknolle']

    selStdNames = []
    for i in st.session_state.dfMain['Name']:
        for j in selectedStandards:
            if j in i:
                selStdNames.append(i)

    selStdNames = list(dict.fromkeys(selStdNames))

    fil2 = st.session_state.dfMain['Name'].isin(selStdNames)

    st.session_state.dfdr = pd.concat(
        [drOnly, st.session_state.dfMain[fil2]]).sort_values('Point Nr.')

    # a df with only standards & samples
    st.session_state.dfSampleNames = st.session_state.dfMain[~fil1].loc[:, 'Name'].drop_duplicates(
    )

    # Names of all the standards in the Moessbauer standard file
    st.session_state.dfMoessNames = []
    for i in st.session_state.dfSampleNames.tolist():
        res1 = list(map(lambda x: i if x in i else 0,
                        st.session_state.dfMoess['Name']))
        res2 = [n for n in res1 if n != 0]
        if len(res2) == 1:
            st.session_state.dfMoessNames.append(res2[0])

# ------------ End produce dfdr and dfSampleNames


# -- Start Extract data and calculate the average for a sample/standard from its multiple measurement points

def extractAndCalculateAverages(data, l, crystal, nr_of_smp):
    import pandas as pd
    if crystal == 'TAP2':
        Lb = r'L$\beta$ (TAP2)'
        La = r'L$\alpha$ (TAP2)'
    else:
        Lb = r'L$\beta$ (TAP4)'
        La = r'L$\alpha$ (TAP4)'

    res = []
    for i in l:
        fil = data['Name'] == i
        # if we would use data instead of d, data would be replaced by only the selected elements
        # This is to only use a subset of data from each measurement point (which in fact are multiple points)
        if nr_of_smp != 0:
            if nr_of_smp > len(data[fil]):
                nr_of_smp = len(data[fil])
            d = data[fil].sample(nr_of_smp)
        else:
            d = data[fil]
        # d = data[fil]
        resFeConcentrations = (
            d['FeO'] * 55.845 / (55.845 + 15.9994)).mean()
        resLBetaAlphaRatios = (d[Lb]/d[La]).mean()
        res.append([d['Point Nr.'].tolist()[0], i,
                    resFeConcentrations, resLBetaAlphaRatios])

    if crystal == 'TAP2':
        ret = pd.DataFrame(res).rename(columns={
            0: 'Point Nr.', 1: 'Name', 2: r'Fe$_{tot}$', 3: r'L$\beta$/L$\alpha$ (TAP2)'})
    else:
        ret = pd.DataFrame(res).rename(columns={
            0: 'Point Nr.', 1: 'Name', 2: r'Fe$_{tot}$', 3: r'L$\beta$/L$\alpha$ (TAP4)'})

    return ret

# -- End Extract data and calculate the average for a sample/standard from its multiple measurement points

# ------  Start Fit Parameter linear regression
def regressionFitParameters(inpData, crystal):
    data = inpData
    if crystal == 'TAP2':
        crystalName = ' (TAP2)'
    else:
        crystalName = ' (TAP4)'

    x = st.session_state.dfFitData[r'L$\beta$/L$\alpha$' + crystalName]
    y = st.session_state.dfFitData[r'Fe$_{tot}$']
    z = st.session_state.dfFitData[r'Fe$^{2+}$']

    A = [
        # length(x) sum(x) sum(y) sum(x.*y)
        [len(x), x.sum(), y.sum(), (x * y).sum()],
        # sum(x) sum(x.^2) sum(x.*y) sum(y.*x.^2)
        [x.sum(), (x ** 2).sum(), (x * y).sum(), (y * x ** 2).sum()],
        # sum(y) sum(x.*y) sum(y.^2) sum(x.*y.^2)
        [y.sum(), (x * y).sum(), (y ** 2).sum(), (x * y ** 2).sum()],
        # sum(x.*y) sum((x.^2).*y) sum(x.*y.^2) sum((x.^2).*(y.^2))]
        [(x * y).sum(), ((x ** 2) * y).sum(),
            (x * y ** 2).sum(), ((x ** 2) * (y ** 2)).sum()]
    ]

    v = [z.sum(), (z * x).sum(), (z * y).sum(), (x * y * z).sum()]

    rfp = np.linalg.inv(A) @ v     # regression parameters
    st.session_state.tmp = rfp
    if crystal == 'TAP2':
        st.session_state.fitParametersTAP2 = rfp
    else:
        st.session_state.fitParametersTAP4 = rfp

    res = rfp[0] + rfp[1] * (data[r'L$\beta$/L$\alpha$' + crystalName]) + rfp[2] * \
        data[r'Fe$_{tot}$'] + rfp[3] * \
        (data[r'Fe$_{tot}$'] * data[r'L$\beta$/L$\alpha$' + crystalName])

    resultsFe3FP = (data[r'Fe$_{tot}$'] - res)/data[r'Fe$_{tot}$']
    
    return resultsFe3FP

# ------  End Fit Parameter linear regression


# ------  Start Pre-processsing data
# Command for getting Fe2+ and Fetot values from the dfMoss dataset
def extractKnownFe2(stdNameForMatching):
    foundStd = st.session_state.dfMoess[st.session_state.dfMoess['Name'].str.contains(
        stdNameForMatching)]
    Fe2ModAbValue = foundStd['Fe2+/SumFe'].tolist()[0]
    return Fe2ModAbValue


def preProcessingData(nr_of_smp):
    # import pandas as pd
    # import streamlit as st
    st.session_state.smpList = list(
        set(st.session_state.dfSampleNames.tolist()) - set(st.session_state.stdSelection))
    st.session_state.dfMeasStdSelTAP2 = extractAndCalculateAverages(
        st.session_state.dfMain, st.session_state.stdSelection, 'TAP2', nr_of_smp)
    st.session_state.dfMeasStdDataTAP2 = extractAndCalculateAverages(
        st.session_state.dfMain, st.session_state.dfMoessNames, 'TAP2', nr_of_smp)
    st.session_state.dfMeasDriftTAP2 = extractAndCalculateAverages(
        st.session_state.dfMain, st.session_state.dfdr['Name'].drop_duplicates().tolist(), 'TAP2', nr_of_smp)
    st.session_state.dfMeasSmpDataTAP2 = extractAndCalculateAverages(
        st.session_state.dfMain, st.session_state.smpList, 'TAP2', nr_of_smp).sort_values(by='Point Nr.')

    st.session_state.dfMeasStdSelTAP4 = extractAndCalculateAverages(
        st.session_state.dfMain, st.session_state.stdSelection, 'TAP4', nr_of_smp)
    st.session_state.dfMeasStdDataTAP4 = extractAndCalculateAverages(
        st.session_state.dfMain, st.session_state.dfMoessNames, 'TAP4', nr_of_smp)
    st.session_state.dfMeasDriftTAP4 = extractAndCalculateAverages(
        st.session_state.dfMain, st.session_state.dfdr['Name'].drop_duplicates().tolist(), 'TAP4', nr_of_smp)
    st.session_state.dfMeasSmpDataTAP4 = extractAndCalculateAverages(
        st.session_state.dfMain, st.session_state.smpList, 'TAP4', nr_of_smp).sort_values(by='Point Nr.')

    # Combining measured standard data and required known Fe2+ and Fetot from standard data (-> Moessbauer data)
    combMoessAndMeasStdData = []
    for i in st.session_state.dfMeasStdSelTAP2['Name']:
        res = extractKnownFe2(i.split('_')[0].strip())
        combMoessAndMeasStdData.append(res)

    dfFitData = pd.concat([st.session_state.dfMeasStdSelTAP2,
                           st.session_state.dfMeasStdSelTAP4[r'L$\beta$/L$\alpha$ (TAP4)']], axis=1)
    dfFitData = pd.concat(
        [dfFitData, pd.DataFrame(combMoessAndMeasStdData)], axis=1)
    dfFitData = dfFitData.rename(
        columns={0: r'Fe$^{2+}$/$\Sigma$Fe (Moess)'})

    dfFitData = pd.concat([dfFitData, pd.DataFrame(
        dfFitData.loc[:, r'Fe$_{tot}$'] * dfFitData.loc[:, r'Fe$^{2+}$/$\Sigma$Fe (Moess)'])], axis=1)
    st.session_state.dfFitData = dfFitData.rename(
        columns={0: r'Fe$^{2+}$'})


def calcFullOutputFile(nOfAn):
    # Preparation of a file with averages, used for output, in parametrisation and results inspection
    st.session_state.dfMainColumns = st.session_state.dfMain.drop(
        'Point Nr.', axis=1).columns

    st.session_state.output1 = pd.DataFrame()
    st.session_state.output2 = pd.DataFrame()
    st.session_state.n_of_analyses = []
    st.session_state.smpAndstdList = pd.Series(st.session_state.dfSampleNames.tolist() +
                                               st.session_state.dfMoessNames).drop_duplicates()
    fil = st.session_state.resultsFe3Smp['Name'].isin(
        st.session_state.dfMoessNames)
    st.session_state.Fe3SmpAndFe3Std = pd.concat(
        [st.session_state.resultsFe3Smp[~fil], st.session_state.resultsFe3Std])

    for i in st.session_state.smpAndstdList:
        if (True in st.session_state.dfMain['Name'].isin([i]).drop_duplicates().tolist()) & (True in st.session_state.Fe3SmpAndFe3Std['Name'].isin([i]).drop_duplicates().tolist()):
            fil1 = st.session_state.dfMain['Name'] == i
            data1 = st.session_state.dfMain[fil1].loc[:,
                                                      st.session_state.dfMainColumns]
            fil2 = st.session_state.Fe3SmpAndFe3Std['Name'] == i
            data2 = st.session_state.Fe3SmpAndFe3Std[fil2]

            st.session_state.output1 = pd.concat(
                [st.session_state.output1, data1.mean(numeric_only=True)], axis=1)  # change: I had to add numeric_only=True
            st.session_state.output1.rename(columns={0: i}, inplace=True)
            st.session_state.output2 = pd.concat(
                [st.session_state.output2, data2])
            if nOfAn == 0:
                n_of_an = len(data1)
            elif len(data1) < nOfAn:
                n_of_an = len(data1)
            else:
                n_of_an = nOfAn
            st.session_state.n_of_analyses.append(n_of_an)

    st.session_state.output_file = pd.concat([st.session_state.output2.reset_index(
        drop=True), st.session_state.output1.T.reset_index()], axis=1)
    st.session_state.output_file.insert(
        1, 'n', st.session_state.n_of_analyses)

    fil2 = st.session_state.output_file['Name'].isin(
        st.session_state.dfMoessNames)
    st.session_state.smp_output_file = st.session_state.output_file[~fil2]
    st.session_state.std_output_file = st.session_state.output_file[fil2]

# ------  End Pre-processsing data
    
# ------ Start Calculate regressions & produce results
def calcRegressionsAndProduceResults(selMoessData):
    resultsFe3StdFPTAP2 = pd.DataFrame(regressionFitParameters(
        st.session_state.dfMeasStdDataTAP2, 'TAP2'))
    resultsFe3DriftFPTAP2 = pd.DataFrame(
        regressionFitParameters(st.session_state.dfMeasDriftTAP2, 'TAP2'))
    resultsFe3SmpFPTAP2 = pd.DataFrame(regressionFitParameters(
        st.session_state.dfMeasSmpDataTAP2, 'TAP2'))
    resultsFe3StdFPTAP4 = pd.DataFrame(regressionFitParameters(
        st.session_state.dfMeasStdDataTAP4, 'TAP4'))
    resultsFe3DriftFPTAP4 = pd.DataFrame(
        regressionFitParameters(st.session_state.dfMeasDriftTAP4, 'TAP4'))
    resultsFe3SmpFPTAP4 = pd.DataFrame(regressionFitParameters(
        st.session_state.dfMeasSmpDataTAP4, 'TAP4'))
    print('model linear regression results successfully produced')

    fe3StdMoessList = []
    for i in st.session_state.dfMeasStdDataTAP2['Name']:
        fil = list(map(lambda x: True if x in i else False,
                       st.session_state.dfMoess['Name']))
        # ['Fe3+/SumFe'].values[0])
        fe3StdMoessList.append(
            st.session_state.dfMoess[fil][selMoessData].values[0])
    fe3StdMoessList = pd.DataFrame(fe3StdMoessList)

    st.session_state.resultsFe3Std = pd.concat([st.session_state.dfMeasStdDataTAP2['Point Nr.'], st.session_state.dfMeasStdDataTAP2['Name'], st.session_state.dfMeasStdDataTAP2[r'Fe$_{tot}$'], fe3StdMoessList[0], resultsFe3StdFPTAP2[0], resultsFe3StdFPTAP2[0] - fe3StdMoessList[0], resultsFe3StdFPTAP4[0], resultsFe3StdFPTAP4[0] - fe3StdMoessList[0], resultsFe3StdFPTAP2[0]-resultsFe3StdFPTAP4[0]], axis=1, keys=['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)', 'Moessbauer', r'Fe$^{3+}$/$\Sigma$Fe (2TAPL)', r'$\Delta$ Meas - Moess (2TAPL)', r'Fe$^{3+}$/$\Sigma$Fe (4TAPL)', r'$\Delta$ Meas - Moess (4TAPL)', '2TAPL-4TAPL'
                                                                                                                                                                                                                                                                                                                                                                                                                            ])
    drMonitorFe3 = st.session_state.dfMoess[st.session_state.dfMoess['Name']
                                            == st.session_state.drMonitorName]['Fe3+/SumFe'].values[0]
    st.session_state.resultsFe3Drift = pd.concat([st.session_state.dfMeasDriftTAP2['Point Nr.'], st.session_state.dfMeasDriftTAP2['Name'], st.session_state.dfMeasDriftTAP2[r'Fe$_{tot}$'], pd.Series([drMonitorFe3]*len(st.session_state.dfMeasDriftTAP4))  # fe3StdMoessList[0]
                                                  , resultsFe3DriftFPTAP2[0], resultsFe3DriftFPTAP2[0] - drMonitorFe3, resultsFe3DriftFPTAP4[0], resultsFe3DriftFPTAP4[0] - drMonitorFe3], axis=1, keys=['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)', 'Moessbauer', r'Fe$^{3+}$/$\Sigma$Fe (2TAPL)', r'$\Delta$ Meas - Moess (2TAPL)', r'Fe$^{3+}$/$\Sigma$Fe (4TAPL)', r'$\Delta$ Meas - Moess (4TAPL)'
                                                                                                                                                                                                         ])
    st.session_state.resultsFe3Smp = pd.concat([st.session_state.dfMeasSmpDataTAP2['Point Nr.'], st.session_state.dfMeasSmpDataTAP2['Name'], st.session_state.dfMeasSmpDataTAP2[r'Fe$_{tot}$'], resultsFe3SmpFPTAP2[0], resultsFe3SmpFPTAP4[0], resultsFe3SmpFPTAP2[0]-resultsFe3SmpFPTAP4[0]], axis=1,
                                               keys=['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)', r'Fe$^{3+}$/$\Sigma$Fe (2TAPL)', r'Fe$^{3+}$/$\Sigma$Fe (4TAPL)', '2TAPL-4TAPL'])
# ------ End Calculate regressions & produce results