#!/usr/bin/env python
# coding: utf-8

import streamlit as st

#hide_st_style = """
#            <style>
#            #MainMenu {visibility: hidden;}
#            footer {visibility: hidden;}
#            header {visibility: hidden;}
#            </style>
#            """
#st.markdown(hide_st_style, unsafe_allow_html=True)


#-----------------------------------------#
#------------ Start ----------------------#
#-----------------------------------------#
def start():
    import streamlit as st
    import pandas as pd
    
#------------ Start Test for Duplicates
    
    def reportDuplicatesInList(l):
        import streamlit as st
        tmp = []
        tmp2 = []
        for i in l:
            if i not in tmp:
                tmp.append(i)
            else:
                tmp2.append(i)
        
        if len(tmp2) == 0:
            res = 'No duplicates for sample or standard names were found'
        else:
            res = 'The following Duplicates were found:'
    
        return st.markdown(f'<p style="color:green"><b>{res}</b> </p>', unsafe_allow_html=True)
#------------ End Test for Duplicates


    st.write("# Welcome to Flank Data Reduction")
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             Upload your file, then proceed to 'Data Reduction'. For instructions how to prepare your initial file, you will
             find the required documentation in 'Tutorials & Instructions'.
         """)
         
    with st.sidebar:
        with st.expander("New here? Click here!"):
         st.write("""
             The best place to start for you is 'Tutorials & Instructions'.
             Alternatively, 'Method & References' is a valuable resource of basic information.'
         """)

    st.markdown(""" **Start your reduction journey by uploading your data file below** """)
    
    uploaded_file = st.file_uploader('')
    if uploaded_file is not None:
         st.session_state.dfRaw = pd.read_csv(uploaded_file)
         #st.write(reportDuplicatesInList(st.session_state.dfRaw.loc[:, 'Comment']))
         reportDuplicatesInList(st.session_state.dfRaw.loc[:, 'Comment'])
         st.markdown('<p style="color:green"><b>You uploaded the following data for flank reduction</b> </p>', unsafe_allow_html=True)
         st.session_state.dfSampleNames = None  # initilising these parameter for the next page, where this would otherwise produce an error message
         st.session_state.dfFitData = None      # initilising these parameter for the next page, where this would otherwise produce an error message
         st.write(st.session_state.dfRaw)

    st.markdown('''**The following Moessbauer standard data will be used for your data reduction**''')
    st.session_state.dfMoess = pd.read_csv('https://raw.githubusercontent.com/Hezel2000/GeoDataScience/main/data/moessbauer%20standard%20data.csv')
    st.write(st.session_state.dfMoess)


#-----------------------------------------#
#------------ Start Data Reduction -------#
#-----------------------------------------#
def dataReduction():
    import pandas as pd

#------------ Start Prepare Dataset
    def prepareDataset(sel):
        dfComplete = st.session_state.dfRaw
        if sel == 'all':
            for ind in dfComplete.index:
                measurementPointName = dfComplete['Comment'][ind]
        
                if 'Grid' in measurementPointName:
                    dfComplete['Comment'] = dfComplete['Comment'].replace([measurementPointName],measurementPointName.split('Grid')[0])
                elif 'Line' in measurementPointName:
                    dfComplete['Comment'] = dfComplete['Comment'].replace([measurementPointName],measurementPointName.split('Line')[0])
        
        
            df = dfComplete.loc[:, ['Point', 'Comment', 'Inspected', 'SiO2(Mass%)', 'TiO2(Mass%)', 'Al2O3(Mass%)', 'Cr2O3(Mass%)', 'FeO(Mass%)', 'MnO(Mass%)',
                            'NiO(Mass%)', 'MgO(Mass%)',  'CaO(Mass%)',  'Na2O(Mass%)', 'K2O(Mass%)', 'P2O5(Mass%)', 'Total(Mass%)',
                            'Bi(Net)', 'Ar(Net)', 'Br(Net)', 'As(Net)', 'Current']]
        

            df = df.rename(columns = {'Point':'Point Nr.', 'Comment':'Name', 'Inspected': 'Name Inspected', 'SiO2(Mass%)':'SiO2', 'TiO2(Mass%)':'TiO2', 'Al2O3(Mass%)':'Al2O3',
                                      'Cr2O3(Mass%)':'Cr2O3', 'FeO(Mass%)':'FeO', 'MnO(Mass%)':'MnO', 'NiO(Mass%)':'NiO',
                                      'MgO(Mass%)':'MgO', 'CaO(Mass%)':'CaO', 'Na2O(Mass%)':'Na2O', 'K2O(Mass%)':'K2O',
                                      'P2O5(Mass%)':'P2O5', 'Total(Mass%)':'Total',
                                      'Bi(Net)':r'L$\beta$ (TAP2)', 'Ar(Net)':r'L$\alpha$ (TAP2)',
                                      'Br(Net)':r'L$\beta$ (TAP4)', 'As(Net)':r'L$\alpha$ (TAP4)',
                                      'Current':'Current (nA)'})
        #                              'Bi':'Lbeta (TAP2)', 'Ar':'Lalpha (TAP2)',
        #                              'Br':'Lbeta (TAP4)', 'As':'Lalpha (TAP4)'})
        else:
            for ind in dfComplete.index:
                measurementPointName = dfComplete['Inspected'][ind]
        
                if 'Grid' in measurementPointName:
                    dfComplete['Inspected'] = dfComplete['Inspected'].replace([measurementPointName],measurementPointName.split('Grid')[0])
                elif 'Line' in measurementPointName:
                    dfComplete['Inspected'] = dfComplete['Inspected'].replace([measurementPointName],measurementPointName.split('Line')[0])
        
        
            df = dfComplete.loc[:, ['Point', 'Comment', 'Inspected', 'SiO2(Mass%)', 'TiO2(Mass%)', 'Al2O3(Mass%)', 'Cr2O3(Mass%)', 'FeO(Mass%)', 'MnO(Mass%)',
                            'NiO(Mass%)', 'MgO(Mass%)',  'CaO(Mass%)',  'Na2O(Mass%)', 'K2O(Mass%)', 'P2O5(Mass%)', 'Total(Mass%)',
                            'Bi(Net)', 'Ar(Net)', 'Br(Net)', 'As(Net)', 'Current']]
        

            df = df.rename(columns = {'Point':'Point Nr.', 'Comment':'Name of All', 'Inspected':'Name', 'SiO2(Mass%)':'SiO2', 'TiO2(Mass%)':'TiO2', 'Al2O3(Mass%)':'Al2O3',
                                      'Cr2O3(Mass%)':'Cr2O3', 'FeO(Mass%)':'FeO', 'MnO(Mass%)':'MnO', 'NiO(Mass%)':'NiO',
                                      'MgO(Mass%)':'MgO', 'CaO(Mass%)':'CaO', 'Na2O(Mass%)':'Na2O', 'K2O(Mass%)':'K2O',
                                      'P2O5(Mass%)':'P2O5', 'Total(Mass%)':'Total',
                                      'Bi(Net)':r'L$\beta$ (TAP2)', 'Ar(Net)':r'L$\alpha$ (TAP2)',
                                      'Br(Net)':r'L$\beta$ (TAP4)', 'As(Net)':r'L$\alpha$ (TAP4)',
                                      'Current':'Current (nA)'})
        #                              'Bi':'Lbeta (TAP2)', 'Ar':'Lalpha (TAP2)',
        #                              'Br':'Lbeta (TAP4)', 'As':'Lalpha (TAP4)'})
    
        df = pd.concat([df, df[r'L$\beta$ (TAP2)']/df[r'L$\alpha$ (TAP2)'], df[r'L$\beta$ (TAP4)']/df[r'L$\alpha$ (TAP4)']], axis = 1)
        
        st.session_state.dfMain = df.rename(columns = {0:r'L$\beta$/L$\alpha$ (TAP2)', 1:r'L$\beta$/L$\alpha$ (TAP4)'})
        
#------------ End Prepare Dataset

#------------ Start produce dfdr and dfSampleNames

    def subsetsOfDatasets():
        # a df with only drift measurements
        # drift measurements will be stored in the DataFrame: dfdr
        fil1 = st.session_state.dfMain['Name'].str.startswith('dr')
        drOnly = st.session_state.dfMain[fil1]
        
        selectedStandards = ['AlmO'] #['AlmO', 'UA5', 'UA10', 'Damknolle']
    
        selStdNames = []
        for i in st.session_state.dfMain['Name']:
            for j in selectedStandards:
                if j in i:
                    selStdNames.append(i)
    
        selStdNames = list(dict.fromkeys(selStdNames))
    
        fil2 = st.session_state.dfMain['Name'].isin(selStdNames)
    
        st.session_state.dfdr = pd.concat([drOnly, st.session_state.dfMain[fil2]]).sort_values('Point Nr.')
    
        # a df with only standards & samples
        st.session_state.dfSampleNames = st.session_state.dfMain[~fil1].loc[:, 'Name'].drop_duplicates()
        
        # Names of all the standards in the Moessbauer standard file
        st.session_state.dfMoessNames = []
        for i in st.session_state.dfSampleNames.tolist():
            res1 = list(map(lambda x : i if x in i else 0, st.session_state.dfMoess['Name']))
            res2 = [n for n in res1 if n != 0]
            if len(res2) == 1:
                st.session_state.dfMoessNames.append(res2[0])
    
#------------ End produce dfdr and dfSampleNames


##-----------------------------------------------##
##-- Extract data and calculate the average  ----##
##-- for a sample/standard from its multiple ----##
##-- measurement points                      ----##
##-----------------------------------------------##

    def extractAndCalculateAverages(data, l, crystal):    
        if crystal == 'TAP2':
            Lb = r'L$\beta$ (TAP2)'
            La = r'L$\alpha$ (TAP2)'
        else:
            Lb = r'L$\beta$ (TAP4)'
            La = r'L$\alpha$ (TAP4)'
        
        res = []
        for i in l:
            fil = data['Name'] == i
            d = data[fil] # if we would use data instead of d, data would be replaced by only the selected elements
            resFeConcentrations = (d['FeO'] * 55.845 / (55.845 + 15.9994)).mean()
            resLBetaAlphaRatios = (d[Lb]/d[La]).mean()
            res.append([d['Point Nr.'].tolist()[0], i, resFeConcentrations, resLBetaAlphaRatios])
    
        if crystal == 'TAP2':
            ret = pd.DataFrame(res).rename(columns = {0:'Point Nr.', 1:'Name', 2:r'Fe$_{tot}$', 3:r'L$\beta$/L$\alpha$ (TAP2)'})        
        else:
            ret = pd.DataFrame(res).rename(columns = {0:'Point Nr.', 1:'Name', 2:r'Fe$_{tot}$', 3:r'L$\beta$/L$\alpha$ (TAP4)'})
            
        return ret
    

##-----------------------------------------------##
##------  Fit Parameter linear regression  ------##
##-----------------------------------------------##
        
    def regressionFitParameters(inpData, crystal):
        import numpy as np
        
        data = inpData
        if crystal == 'TAP2':
            crystalName = ' (TAP2)'
        else:
            crystalName = ' (TAP4)'
        
        x = st.session_state.dfFitData[r'L$\beta$/L$\alpha$' + crystalName]
        y = st.session_state.dfFitData[r'Fe$_{tot}$']
        z = st.session_state.dfFitData[r'Fe$^{2+}$']
    
        A = [
            [len(x), x.sum(), y.sum(), (x * y).sum()],                         # length(x) sum(x) sum(y) sum(x.*y)
            [x.sum(), (x ** 2).sum(), (x * y).sum(), (y * x ** 2).sum()],          # sum(x) sum(x.^2) sum(x.*y) sum(y.*x.^2)
            [y.sum(), (x * y).sum(), (y ** 2).sum(), (x * y ** 2).sum()],          # sum(y) sum(x.*y) sum(y.^2) sum(x.*y.^2)
            [(x * y).sum(), ((x ** 2) * y).sum(), (x * y ** 2).sum(), ((x ** 2) * (y **2 )).sum()]  # sum(x.*y) sum((x.^2).*y) sum(x.*y.^2) sum((x.^2).*(y.^2))]
            ]
    
        v = [z.sum(), (z * x).sum(), (z * y).sum(), (x * y * z).sum()]
        
        rfp = np.linalg.inv(A) @ v     # regression parameters
    
        if crystal == 'TAP2':
            st.session_state.fitParametersTAP2 = rfp
        else:
            st.session_state.fitParametersTAP4 = rfp
        
        res = rfp[0] + rfp[1] * (data[r'L$\beta$/L$\alpha$' + crystalName]) + rfp[2] * data[r'Fe$_{tot}$'] + rfp[3] * (data[r'Fe$_{tot}$'] * data[r'L$\beta$/L$\alpha$' + crystalName])
    
        resultsFe3FP = (data[r'Fe$_{tot}$'] - res)/data[r'Fe$_{tot}$']
    
        return resultsFe3FP


##-----------------------------------------------##
##-----------  Pre-processsing data  ------------##
##-----------------------------------------------##
        
# Command for getting Fe2+ and Fetot values from the dfMoss dataset
    def extractKnownFe2(stdNameForMatching):
        foundStd = st.session_state.dfMoess[st.session_state.dfMoess['Name'].str.contains(stdNameForMatching)]
        #Fe2Value = foundStd['FeO (wt%)'].tolist()[0] * 55.845/(55.845 + 15.9994)
        Fe2ModAbValue = foundStd['Fe2+/SumFe'].tolist()[0]
        return Fe2ModAbValue


    def preProcessingData():        
        # Getting the indices of the samples and standards
        #samplesListReIndexed = pd.Series(st.session_state.dfSampleNames.tolist())

        #fil = samplesListReIndexed.str.contains('AlmO') | samplesListReIndexed.str.contains('UA5') | samplesListReIndexed.str.contains('UA10') | samplesListReIndexed.str.contains('Damknolle')
                
        #samples = samplesListReIndexed[~fil].index.values.tolist()
        #standards = samplesListReIndexed[fil].index.values.tolist()
        
        # Getting sample data
        #st.session_state.smpList = st.session_state.dfSampleNames.iloc[samples].tolist()

        # First, the indices of the standard measurements must be input from. These are found in the dfSampleNames above
        #st.session_state.stdList = st.session_state.dfSampleNames.iloc[standards].tolist()

        # Extracting FeO and Lalpha/Lbeta, the Lbeta/Lalpha ratios are calculated from the measured Lbeta and Lalpha cps, and the data are averaged
        #st.session_state.dfMeasStdDataTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdList, 'TAP2')
        #st.session_state.dfMeasStdDataTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdList, 'TAP4')
        
        st.session_state.smpList = list(set(st.session_state.dfSampleNames.tolist()) - set(st.session_state.stdSelection))
        
        st.session_state.dfMeasStdSelTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdSelection, 'TAP2')
        st.session_state.dfMeasStdDataTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.dfMoessNames, 'TAP2')
        st.session_state.dfMeasDriftTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.dfdr['Name'].drop_duplicates().tolist(), 'TAP2')
        st.session_state.dfMeasSmpDataTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.smpList, 'TAP2').sort_values(by='Point Nr.')

        st.session_state.dfMeasStdSelTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdSelection, 'TAP4')
        st.session_state.dfMeasStdDataTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.dfMoessNames, 'TAP4')
        st.session_state.dfMeasDriftTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.dfdr['Name'].drop_duplicates().tolist(), 'TAP4')
        st.session_state.dfMeasSmpDataTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.smpList, 'TAP4').sort_values(by='Point Nr.')
        
        # Combining measured standard data and required known Fe2+ and Fetot from standard data (-> Moessbauer data)
        combMoessAndMeasStdData = []
        for i in st.session_state.dfMeasStdSelTAP2['Name']:
            res = extractKnownFe2(i.split('_')[0])
            combMoessAndMeasStdData.append(res)


        dfFitData = pd.concat([st.session_state.dfMeasStdSelTAP2, st.session_state.dfMeasStdSelTAP4[r'L$\beta$/L$\alpha$ (TAP4)']], axis = 1)
#        dfFitData = dfFitData.rename(columns = {0 : r'Fe$^{2+}$/$\Sigma$Fe', 1 : r'Fe$^{2+}$'})
        dfFitData = pd.concat([dfFitData, pd.DataFrame(combMoessAndMeasStdData)], axis = 1)
        dfFitData = dfFitData.rename(columns = {0 : r'Fe$^{2+}$/$\Sigma$Fe (Moess)'})

        dfFitData = pd.concat([dfFitData, pd.DataFrame(dfFitData.loc[:, r'Fe$_{tot}$'] * dfFitData.loc[:, r'Fe$^{2+}$/$\Sigma$Fe (Moess)'])], axis = 1)
        st.session_state.dfFitData = dfFitData.rename(columns = {0 : r'Fe$^{2+}$'})

    
##-----------------------------------------------##
##--  Calculate regressions & produce results  --##
##-----------------------------------------------##

    def calcRegressionsAndProduceResults():    
        resultsFe3StdFPTAP2 = pd.DataFrame(regressionFitParameters(st.session_state.dfMeasStdDataTAP2, 'TAP2'))
        resultsFe3DriftFPTAP2 = pd.DataFrame(regressionFitParameters(st.session_state.dfMeasDriftTAP2, 'TAP2'))
        resultsFe3SmpFPTAP2 = pd.DataFrame(regressionFitParameters(st.session_state.dfMeasSmpDataTAP2, 'TAP2'))
        resultsFe3StdFPTAP4 = pd.DataFrame(regressionFitParameters(st.session_state.dfMeasStdDataTAP4, 'TAP4'))
        resultsFe3DriftFPTAP4 = pd.DataFrame(regressionFitParameters(st.session_state.dfMeasDriftTAP4, 'TAP4'))
        resultsFe3SmpFPTAP4 = pd.DataFrame(regressionFitParameters(st.session_state.dfMeasSmpDataTAP4, 'TAP4'))
        print('model linear regression results successfully produced')
    
        fe3StdMoessList = []
        for i in st.session_state.dfMeasStdDataTAP2['Name']:
            fil = list(map(lambda x : True if x in i else False, st.session_state.dfMoess['Name']))
            fe3StdMoessList.append(st.session_state.dfMoess[fil]['Fe3+/SumFe'].values[0])
        fe3StdMoessList = pd.DataFrame(fe3StdMoessList)
    
        st.session_state.resultsFe3Std = pd.concat([st.session_state.dfMeasStdDataTAP2['Point Nr.'], st.session_state.dfMeasStdDataTAP2['Name']
                        , st.session_state.dfMeasStdDataTAP2[r'Fe$_{tot}$'], fe3StdMoessList[0]
                        ,resultsFe3StdFPTAP2[0], resultsFe3StdFPTAP2[0] - fe3StdMoessList[0]
                        ,resultsFe3StdFPTAP4[0], resultsFe3StdFPTAP4[0] - fe3StdMoessList[0]], axis = 1
                        ,keys = ['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)', 'Moessbauer'
                                 ,r'Fe$^{3+}$/$\Sigma$Fe (2TAPL)', r'$\Delta$ Meas - Moess (2TAPL)'
                                 ,r'Fe$^{3+}$/$\Sigma$Fe (4TAPL)', r'$\Delta$ Meas - Moess (4TAPL)'
                                ])
        drMonitorFe3 = st.session_state.dfMoess[st.session_state.dfMoess['Name'] == st.session_state.drMonitorName]['Fe3+/SumFe'].values[0]
        st.session_state.resultsFe3Drift = pd.concat([st.session_state.dfMeasDriftTAP2['Point Nr.'], st.session_state.dfMeasDriftTAP2['Name']
                        , st.session_state.dfMeasDriftTAP2[r'Fe$_{tot}$'], pd.Series([drMonitorFe3]*len(st.session_state.dfMeasDriftTAP4)) #fe3StdMoessList[0]
                        ,resultsFe3DriftFPTAP2[0], resultsFe3DriftFPTAP2[0] - drMonitorFe3
                        ,resultsFe3DriftFPTAP4[0], resultsFe3DriftFPTAP4[0] - drMonitorFe3], axis = 1
                        ,keys = ['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)', 'Moessbauer'
                                 ,r'Fe$^{3+}$/$\Sigma$Fe (2TAPL)', r'$\Delta$ Meas - Moess (2TAPL)'
                                 ,r'Fe$^{3+}$/$\Sigma$Fe (4TAPL)', r'$\Delta$ Meas - Moess (4TAPL)'
                                ])
        st.session_state.resultsFe3Smp = pd.concat([st.session_state.dfMeasSmpDataTAP2['Point Nr.'], st.session_state.dfMeasSmpDataTAP2['Name']
                        , st.session_state.dfMeasSmpDataTAP2[r'Fe$_{tot}$']
                        ,resultsFe3SmpFPTAP2[0], resultsFe3SmpFPTAP4[0]
                        ,resultsFe3SmpFPTAP2[0]-resultsFe3SmpFPTAP4[0]], axis = 1,
                         keys = ['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)'
                                 ,r'Fe$^{3+}$/$\Sigma$Fe (2TAPL)', r'Fe$^{3+}$/$\Sigma$Fe (4TAPL)', '2TAPL-4TAPL'])
   

#-----------------------
#--- Website Definitions
#-----------------------   
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             1- Choose which data shall be reduced. 2- Select the standards used to claculte the fit parameters. 
             Note that you can only choose those also present in the Moessbauer Standard Data file. 3- Click on 'Calculate Results'. 
             4- You can change your selection of standards or whether to use all or only the inspected data anytime.
             However, after each change you need to click 'Calculate Results' again.
             5- Proceed to 'Result Tables'.
         """)
         

    st.subheader('1  Choose whether all data or only inspected data will be used')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('all'):
            prepareDataset('all')
            subsetsOfDatasets()
            st.markdown('<p style="color:green">Data succesfully pre-processed</p>', unsafe_allow_html=True)
    with col2:
        if st.button('inspected only'):
            prepareDataset('inspected only')
            subsetsOfDatasets()
            st.markdown('<p style="color:green"><b>Data succesfully pre-processed</b> </p>', unsafe_allow_html=True)
        

    st.subheader('2  Select standards used to calculate the Fit Parameters')
    st.write("Click 'Calculate Results' after you selected the standards – **and click again, should you have changed your selection!**")
    
    if st.session_state.dfSampleNames is not None:
        allSmpNames = st.session_state.dfMoessNames   # st.session_state.dfSampleNames
        st.session_state.stdSelection = st.multiselect('', allSmpNames, allSmpNames[:4])
        st.session_state.drMonitorName = st.selectbox('Furhter select the standard used as drift monitor.', st.session_state.dfMoess['Name'])
        #st.write('You selected: ', st.session_state.stdSelection)
            
    if st.button('Calculate Results'):
        preProcessingData()
        calcRegressionsAndProduceResults()
        st.markdown('<p style="color:green"><b>Flank data successfully reduced!</b></p>', unsafe_allow_html=True)

    st.markdown('<h4 style="color:blue"><b>Your Selected Standards Used for Fitting</b> </h3>', unsafe_allow_html=True)
    if st.session_state.dfFitData is not None:
        st.write(st.session_state.dfFitData.round(3))        
        
        st.markdown('<h4 style="color:blue"><b>Calculated Fit Parameters for 2TAPL & 4TAPL</b> </h4>', unsafe_allow_html=True)
        st.write(pd.DataFrame({'Parameter':['A', 'B', 'C', 'D'],
                               'TAP2':st.session_state.fitParametersTAP2,
                               'TAP4':st.session_state.fitParametersTAP4}))
        
        st.markdown('<h4 style="color:black"><b>Regression formulas, in which the parameters are used</b> </h4>', unsafe_allow_html=True)
        st.latex(r'''Fe^{2+} = A + B \times \frac{L\beta}{L\alpha} + C \times \Sigma Fe + D \times \Sigma Fe \times \frac{L\beta}{L\alpha}''')
        st.latex(r'''Fe^{3+} = -A - B \times \frac{L\beta}{L\alpha} - C \times \Sigma Fe - D \times \Sigma Fe \times \frac{L\beta}{L\alpha} + Fe_{tot}''')
        st.latex(r'''\textrm{The result is } Fe^{2+} \textrm{ or } Fe^{3+} \textrm{, respectively, in wt\%} ''')


#-----------------------------------------#
#------------ Start Result Tables --------#
#-----------------------------------------#
def resultTables():
    
    @st.cache
    def convert_df(df):
         return df.to_csv().encode('utf-8')
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             The results of the Fe3+ abundances in the standards, the drift monitor, as well as the samples are displayed.
             Each table can be downloaded as a single .csv file if required. More complete output options are provided in 'Output'.
             Once, everything is checked, proceed to 'Visualisations' – or use this site to come back and check individual values.
         """)


    st.subheader('$$Fe^{3+}/ \Sigma Fe$$' + ' in the Standards')
    st.write(r'$Fe^{3+}/\Sigma Fe$ deviation from the Moessbauer data should be <0.01-0.015')
    st.write(st.session_state.resultsFe3Std.round(3))
    csv = convert_df(st.session_state.resultsFe3Std)
    st.download_button(
         label="Download standard data as .csv",
         data=csv,
         file_name='Fe3+ of all measured standards.csv',
         mime='text/csv',
     )
        

    st.subheader('$$Fe^{3+}/ \Sigma Fe$$' + ' in the Drift Monitor')
    st.write(r'$Fe^{3+}/\Sigma Fe$ deviation from the Moessbauer data should be <0.01-0.015')
    st.write(st.session_state.resultsFe3Drift.round(3))
    csv = convert_df(st.session_state.resultsFe3Drift)
    st.download_button(
         label="Download drift data as .csv",
         data=csv,
         file_name='Fe3+ of all drift measurements.csv',
         mime='text/csv',
     )
    
    st.subheader('$$Fe^{3+}/ \Sigma Fe$$' + ' in the Samples')
    st.write(r'The error on $Fe^{3+}/\Sigma Fe$ in the Smp is 0.02') 
    st.write(st.session_state.resultsFe3Smp.round(3))
    csv = convert_df(st.session_state.resultsFe3Smp)
    st.download_button(
         label="Download sample data as .csv",
         data=csv,
         file_name='Fe3+ of all measured samples.csv',
         mime='text/csv',
     )


#-----------------------------------------#
#------------ Start Visualisations--------#
#-----------------------------------------#
def visualisations():
    import streamlit as st
    import pandas as pd
    #from bokeh.plotting import figure, output_file, show
#    from bokeh.models import Panel, Tabs


#--------  Start Linear Regression with Fit Parameters
        
    def regressionFitParameters(inpData, crystal):
        import numpy as np
        
        data = inpData
        if crystal == 'TAP2':
            crystalName = ' (TAP2)'
        else:
            crystalName = ' (TAP4)'
        
        x = st.session_state.dfFitData[r'L$\beta$/L$\alpha$' + crystalName]
        y = st.session_state.dfFitData[r'Fe$_{tot}$']
        z = st.session_state.dfFitData[r'Fe$^{2+}$']
    
        A = [
            [len(x), x.sum(), y.sum(), (x * y).sum()],                         # length(x) sum(x) sum(y) sum(x.*y)
            [x.sum(), (x ** 2).sum(), (x * y).sum(), (y * x ** 2).sum()],          # sum(x) sum(x.^2) sum(x.*y) sum(y.*x.^2)
            [y.sum(), (x * y).sum(), (y ** 2).sum(), (x * y ** 2).sum()],          # sum(y) sum(x.*y) sum(y.^2) sum(x.*y.^2)
            [(x * y).sum(), ((x ** 2) * y).sum(), (x * y ** 2).sum(), ((x ** 2) * (y **2 )).sum()]  # sum(x.*y) sum((x.^2).*y) sum(x.*y.^2) sum((x.^2).*(y.^2))]
            ]
    
        v = [z.sum(), (z * x).sum(), (z * y).sum(), (x * y * z).sum()]
        
        rfp = np.linalg.inv(A) @ v     # regression parameters
    
        if crystal == 'TAP2':
            st.session_state.fitParametersTAP2 = rfp
        else:
            st.session_state.fitParametersTAP4 = rfp
        
        res = rfp[0] + rfp[1] * (data[r'L$\beta$/L$\alpha$' + crystalName]) + rfp[2] * data[r'Fe$_{tot}$'] + rfp[3] * (data[r'Fe$_{tot}$'] * data[r'L$\beta$/L$\alpha$' + crystalName])
    
        resultsFe3FP = (data[r'Fe$_{tot}$'] - res)/data[r'Fe$_{tot}$']
    
        return resultsFe3FP

#--------  End Linear Regression with Fit Parameters    
    
#--------  Start Drift Inspection
        
    def driftplots(sel):
        from bokeh.plotting import figure, output_file, ColumnDataSource
        from bokeh.models import Span, BoxAnnotation
        import numpy as np
        
        elements = st.session_state.dfMain.columns.tolist()[3:]
        
        if sel == 'Composition of drift standards':
            el = st.selectbox('Select', elements)
    
            av = np.average(st.session_state.dfdr[el])
            std = np.std(st.session_state.dfdr[el])
            
            reldev = 100 * np.std(st.session_state.dfdr[el])/np.average(st.session_state.dfdr[el])
        #                elif sel == 'elements  ':
    
              
            col1, col2 = st.columns([3, 1])
            col1.subheader('Drift Monitor')
            #st.write(st.session_state.dfdr.to_dict())
            
            TOOLTIPS = [('Name', '@Name'),
                        ('Point Nr.', '@{Point Nr.}'),
                        (el, '@'+el)]
            
            fig = figure(width=500, height=300, tooltips = TOOLTIPS)
                        
            fig.line(st.session_state.dfdr.loc[:, 'Point Nr.'], st.session_state.dfdr.loc[:, el])
            #fig.circle(st.session_state.dfdr.loc[:, 'Point Nr.'], st.session_state.dfdr.loc[:, el])
            output_file("toolbar.html")  
            source = ColumnDataSource(st.session_state.dfdr.to_dict('list'))
            
            fig.circle('Point Nr.', el, size=4, source=source)
        
            fig.xaxis.axis_label='Point Nr.'
            fig.yaxis.axis_label=el + ' (wt%)'
            
            av_hline = Span(location=av, dimension='width', line_dash='dashed', line_color='brown', line_width=2)
            std_add_hline = Span(location=av+std, dimension='width', line_dash='dashed', line_color='brown', line_width=1)
            std_sub_hline = Span(location=av-std, dimension='width', line_dash='dashed', line_color='brown', line_width=1)
            fig.renderers.extend([av_hline, std_add_hline, std_sub_hline])
            std_box = BoxAnnotation(bottom=av-std, top=av+std, fill_alpha=0.2, fill_color='yellow')
            fig.add_layout(std_box)

            col1.bokeh_chart(fig)
            
            col2.subheader('Statistics')
            resAv = 'average: ' + str(round(av, 2)) + '±' + str(round(std, 2))
            resRelStd ='rel. std.: ' + str(round(reldev, 2)) + '%'
            col2.write(resAv)
            col2.write(resRelStd)
            if reldev < 1:
                col2.color_picker('good data', '#39EC39')
            elif 1 <= reldev < 5:
                col2.color_picker('check data', '#F7CF0F')
            else:
                col2.color_picker('worrisome data', '#FF0000')
        
        else:
            fig = figure(width=600, height=400)
            fig.scatter(st.session_state.resultsFe3Drift['Fe$^{3+}$/$\Sigma$Fe (2TAPL)'],
                        st.session_state.resultsFe3Drift['Fe$^{3+}$/$\Sigma$Fe (4TAPL)'])
            fig.xaxis.axis_label=r'''Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)'''
            fig.yaxis.axis_label=r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)'
            
            x = np.linspace(0,1.1 * st.session_state.resultsFe3Drift['Fe$^{3+}$/$\Sigma$Fe (2TAPL)'].max(),10)
            fig.line(x, x)
            fig.line(x, x + .01, line_dash='dashed', line_color='orange')
            fig.line(x, x - .01, line_dash='dashed', line_color='orange')
            
            st.bokeh_chart(fig)
            
            
#--------  End Drift Inspection

#--------  Start Comparing Lalpha & Lbeta

    def comparinglalphalbeta():
        from bokeh.plotting import figure
        #from bokeh.models import Span, BoxAnnotation, Label
        import numpy as np
        
        plwidth=400
        plheight=250
        
        col1, col2, col3 = st.columns(3)
        
        col1.subheader('Sample measurements only')
        fig = figure(width=plwidth, height=plheight)
        tapl2Fe3 = st.session_state.resultsFe3Smp[r'Fe$^{3+}$/$\Sigma$Fe (2TAPL)']
        tapl4Fe3 = st.session_state.resultsFe3Smp[r'Fe$^{3+}$/$\Sigma$Fe (4TAPL)']
        fig.scatter(tapl2Fe3, tapl4Fe3, color='teal')
        
        x = np.linspace(0,1,10)
        fig.line(x, x)
        fig.line(x, x + .02, line_dash='dashed', line_color='orange')
        fig.line(x, x - .02, line_dash='dashed', line_color='orange')
        
        fig.xaxis.axis_label=r"$$Fe^{3+} / \Sigma Fe \textrm{ (2TAPL)}$$"
        fig.yaxis.axis_label=r'$$Fe^{3+} / \Sigma Fe \textrm{ (4TAPL)}$$'
        col1.bokeh_chart(fig)

        
        col1.subheader('All individual measurements')
        fig = figure(width=plwidth, height=plheight)
        tapl2Betacps = st.session_state.dfMain[r'L$\beta$ (TAP2)']
        tapl2Alphacps = st.session_state.dfMain[r'L$\alpha$ (TAP2)']
        tapl4Betacps = st.session_state.dfMain[r'L$\beta$ (TAP4)']
        tapl4Alphacps = st.session_state.dfMain[r'L$\alpha$ (TAP4)']
        
        fig.scatter(tapl2Betacps, tapl2Alphacps, legend_label='2TAPL')
        fig.scatter(tapl4Betacps, tapl4Alphacps, color='olive', legend_label='4TAPL')
        fig.xaxis.axis_label=r'$$L\beta \textrm{ (net intensities)}$$'
        fig.yaxis.axis_label=r'$$L\alpha \textrm{ (net intensities)}$$'
        #ax2.legend()
        col1.bokeh_chart(fig)
        
        
        col3.subheader('All individual measurements')
        fig = figure(width=plwidth, height=plheight)
        fig.line(st.session_state.dfMain['Point Nr.'], (tapl2Betacps/tapl2Alphacps)/(tapl4Betacps/tapl4Alphacps), color='teal')
        fig.xaxis.axis_label='Point Nr.'
        fig.yaxis.axis_label=r'$$L\beta/L\alpha \textrm{ (2TAPL)} /L\beta/L\alpha \textrm{ (4TAPL)}$$'
        col3.bokeh_chart(fig)
        

        col3.subheader('All individual measurements')
        fig = figure(width=plwidth, height=plheight)
        fig.scatter(st.session_state.dfMain['Point Nr.'], tapl2Betacps/tapl2Alphacps, legend_label='2TAPL')
        fig.scatter(st.session_state.dfMain['Point Nr.'], tapl4Betacps/tapl4Alphacps, color='olive', legend_label='4TAPL')
        fig.xaxis.axis_label='Point Nr.'
        fig.yaxis.axis_label=r'$$L\beta/L\alpha \textrm{ (counts-ratio)}$$'
        col3.bokeh_chart(fig)


#--------  End Comparing Lalpha & Lbeta
        
#-------- Start Parametrisation

    def parametrisationplot():
        from bokeh.plotting import figure
        import numpy as np
        
        Fetot = np.linspace(0, 60, 100)
        ATAP2, BTAP2, CTAP2, DTAP2 = st.session_state.fitParametersTAP2
        ATAP4, BTAP4, CTAP4, DTAP4 = st.session_state.fitParametersTAP4
        
        figParam = figure(plot_width=600, plot_height=400)
        
        for i in range(10):
            Fe3 = .1 * i
            figParam.line(Fetot, (-ATAP2 - CTAP2 * Fetot + Fetot - Fetot * Fe3) / 
                          (BTAP2 + DTAP2 * Fetot), line_color = 'blue', line_alpha=.3)
            figParam.line(Fetot, (-ATAP4 - CTAP4 * Fetot + Fetot - Fetot * Fe3) / 
                          (BTAP4 + DTAP4 * Fetot), line_color = 'orange', line_alpha=.3)
        
        figParam.circle(st.session_state.dfMeasSmpDataTAP2[r'Fe$_{tot}$'],st.session_state.dfMeasSmpDataTAP2[r'L$\beta$/L$\alpha$ (TAP2)'],
                         size = 5, legend_label = 'TAP2')
        figParam.circle(st.session_state.dfMeasSmpDataTAP4[r'Fe$_{tot}$'],st.session_state.dfMeasSmpDataTAP4[r'L$\beta$/L$\alpha$ (TAP4)'],
                         size = 5, fill_color='orange', line_color='orange', legend_label = 'TAP4')
        figParam.scatter(st.session_state.dfFitData[r'Fe$_{tot}$'],st.session_state.dfFitData[r'L$\beta$/L$\alpha$ (TAP2)'],
                         size = 8, line_color='black')
        figParam.scatter(st.session_state.dfFitData[r'Fe$_{tot}$'],st.session_state.dfFitData[r'L$\beta$/L$\alpha$ (TAP4)'],
                         size = 8, fill_color='orange', line_color='black')
        
        figParam.xaxis.axis_label = r'$\Sigma$Fe (wt%)'
        figParam.yaxis.axis_label = r'L$\beta$/L$\alpha$ (net cps-ratio)'
        figParam.axis.minor_tick_in = -3
        figParam.axis.minor_tick_out = 6
        
        st.bokeh_chart(figParam)
        
        st.markdown('<h4 style="color:black"><b>Regression formulas</b> </h4>', unsafe_allow_html=True)
        st.latex(r'''Fe^{2+} = A + B \times \frac{L\beta}{L\alpha} + C \times \Sigma Fe + D \times \Sigma Fe \times \frac{L\beta}{L\alpha}''')
        st.latex(r'''Fe^{3+} = -A - B \times \frac{L\beta}{L\alpha} - C \times \Sigma Fe - D \times \Sigma Fe \times \frac{L\beta}{L\alpha} + Fe_{tot}''')
        st.latex(r'''\textrm{The result is } Fe^{2+} \textrm{ or } Fe^{3+} \textrm{, respectively, in wt\%} ''')
        
        st.markdown('<h4 style="color:black"><b>Calculated Fit Parameters</b> </h4>', unsafe_allow_html=True)
        st.write(pd.DataFrame({'Parameter':['A', 'B', 'C', 'D'],
                               'TAP2':st.session_state.fitParametersTAP2,
                               'TAP4':st.session_state.fitParametersTAP4}))

    
#-------- End Parametrisation

#-------- Start Sample Inspection

    def sampleInspection(sel):
        from bokeh.plotting import figure, output_file, ColumnDataSource
        from bokeh.models import Span, BoxAnnotation, Label
        from bokeh.layouts import gridplot
        import numpy as np
        
        def plotStyle(data):
            av = np.average(data[2])
            std = np.std(data[2])
            reldev = 100 * std/av
            if reldev < 1:
                fcColor = 'green'
            elif 1 <= reldev < 5:
                fcColor = 'orange'
            else:
                fcColor = 'red'
            
            fig = figure(width=300, height=150)
            fig.scatter(data[1], data[2])

            av_hline = Span(location=av, dimension='width', line_dash='dashed', line_color='brown', line_width=2)
            std_add_hline = Span(location=av+std, dimension='width', line_dash='dashed', line_color='brown', line_width=1)
            std_sub_hline = Span(location=av-std, dimension='width', line_dash='dashed', line_color='brown', line_width=1)
            fig.renderers.extend([av_hline, std_add_hline, std_sub_hline])
            std_box = BoxAnnotation(bottom=av-std, top=av+std, fill_alpha=0.2, fill_color='yellow')
            fig.add_layout(std_box)
            
            statistics = Label(x=130, y=90, x_units='screen', y_units='screen',
                 text= str(data[0]) +'\n' + str(round(av, 2)) + '±' + str(round(std, 2)) + ' wt%  –  ' + 'rel. std.:' + str(round(reldev, 2)) + '%',
                 text_font_size='8pt', text_align='center',
                 render_mode='css', border_line_color=fcColor, border_line_alpha=.2,
                 background_fill_color=fcColor, background_fill_alpha=.3)
            fig.add_layout(statistics)
            
            return fig
        
        if sel == 'Select one element, display all samples':       
            elements = st.session_state.dfMain.columns.tolist()[3:]
            el = st.selectbox('Select an Element', elements)
            noc = st.number_input('Insert the Number of Plot Columns', value=3)
            
            plotList = []
            for i in st.session_state.dfSampleNames:
                 fil = st.session_state.dfMain['Name'].str.contains(i)
                 xdata = st.session_state.dfMain[fil].loc[:, 'Point Nr.']
                 data = st.session_state.dfMain[fil].loc[:, el]
                 dat = (i, xdata, data)
                 plotList.append(dat)
                 
            grid_layout = gridplot([plotStyle(i) for i in plotList], ncols=int(noc))
            st.bokeh_chart(grid_layout)
            
            
        elif sel == 'Select one sample, display all elements':
            elements = st.session_state.dfMain.columns.tolist()[3:]
            smpNames=st.session_state.dfSampleNames
            smp = st.selectbox('Select a Sample', smpNames)
            noc = st.number_input('Insert the Number of Plot Columns', value=3)
            
            plotList = []
            for i in elements:
                 fil = (st.session_state.dfMain['Name'] == smp) & (st.session_state.dfMain[i])
                 xdata = st.session_state.dfMain[fil].loc[:, 'Point Nr.']
                 data = st.session_state.dfMain[fil].loc[:, i]
                 dat = (i, xdata, data)
                 plotList.append(dat)
            
            grid_layout = gridplot([plotStyle(i) for i in plotList], ncols=int(noc))
            st.bokeh_chart(grid_layout)
            
            
        elif sel == 'Select one sample and one element':
            elements = st.session_state.dfMain.columns.tolist()[3:]
            smpNames=st.session_state.dfSampleNames
            smp = st.selectbox('Select a Sample', smpNames)
            el = st.selectbox('Select an Element', elements)
                                   
            col1, col2 = st.columns([3, 1])
            col1.subheader('Drift Monitor')
            
            dfMainFil = st.session_state.dfMain['Name'] == smp
            smpSel = st.session_state.dfMain[dfMainFil]
            
            av = np.average(smpSel[el])
            std = np.std(smpSel[el])
            reldev = 100 * np.std(smpSel[el])/np.average(smpSel[el])

            TOOLTIPS = [('Name', '@Name'),
                        ('Point Nr.', '@{Point Nr.}'),
                        (el, '@'+el)]
            
            fig = figure(width=500, height=300, tooltips = TOOLTIPS)
                        
            fig.line(smpSel.loc[:, 'Point Nr.'], smpSel.loc[:, el])
            #fig.circle(st.session_state.dfdr.loc[:, 'Point Nr.'], st.session_state.dfdr.loc[:, el])
            output_file("toolbar.html")
            source = ColumnDataSource(smpSel.to_dict('list'))
            
            fig.circle('Point Nr.', el, size=4, source=source)
        
            fig.xaxis.axis_label='Point Nr.'
            fig.yaxis.axis_label=el + ' (wt%)'

            av_hline = Span(location=av, dimension='width', line_dash='dashed', line_color='brown', line_width=2)
            std_add_hline = Span(location=av+std, dimension='width', line_dash='dashed', line_color='brown', line_width=1)
            std_sub_hline = Span(location=av-std, dimension='width', line_dash='dashed', line_color='brown', line_width=1)
            fig.renderers.extend([av_hline, std_add_hline, std_sub_hline])
            std_box = BoxAnnotation(bottom=av-std, top=av+std, fill_alpha=0.2, fill_color='yellow')
            fig.add_layout(std_box)

            col1.bokeh_chart(fig)
            
            
            col2.subheader('Statistics')
            resAv = 'average: ' + str(round(av, 2)) + '±' + str(round(std, 2))
            resRelStd ='rel. std.: ' + str(round(reldev, 2)) + '%'
            col2.write(resAv)
            col2.write(resRelStd)
            if reldev < 1:
                col2.color_picker('good data', '#39EC39')
            elif 1 <= reldev < 5:
                col2.color_picker('check data', '#F7CF0F')
            else:
                col2.color_picker('worrisome data', '#FF0000')
                

#-------- End Sample Inspection

#--------  Start Error Considerations

    def errorConsiderations():
        from bokeh.plotting import figure
        from bokeh.models import Range1d
        from bokeh.layouts import gridplot
                    
    ##-----------------------------------------------##
    ##-------------  result variations --------------##
    ##-----------------------------------------------##
        
        def errorPercentDeviations():
            colorList = ['olive', 'orange']
            
            fig1 = figure(title='2TAPL', width=400, height=250)
            tmp=1
            for i in [1, 2]:
                del tmp
                tmp = st.session_state.dfMeasSmpDataTAP2.copy()
                tmp[r'Fe$_{tot}$'] = tmp[r'Fe$_{tot}$'] * .01 * (100 - i)
                yData1 = regressionFitParameters(st.session_state.dfMeasSmpDataTAP2, 'TAP2') - regressionFitParameters(tmp, 'TAP2')
                fig1.line(range(len(yData1)), yData1, color=colorList[i-1], legend_label = str(i) + '%')
        
            fig1.xaxis.axis_label='sample'
            fig1.yaxis.axis_label=r'$$\textrm{absolute deviation of } Fe^{3+} / \Sigma Fe$$'
            fig1.y_range = Range1d(0, 1.3 * yData1.max())
    
            
            fig2 = figure(title='4TAPL', width=400, height=250)
            for i in [1, 2]:
                del tmp
                tmp = st.session_state.dfMeasSmpDataTAP4.copy()
                tmp[r'Fe$_{tot}$'] = tmp[r'Fe$_{tot}$'] * .01 * (100 - i)
                yData2 = regressionFitParameters(st.session_state.dfMeasSmpDataTAP4, 'TAP4') - regressionFitParameters(tmp, 'TAP4')
                fig2.line(range(len(yData2)), yData2, color=colorList[i-1], legend_label = str(i) + '%')
        
            fig2.xaxis.axis_label='sample'
            fig2.yaxis.axis_label=r'$$\textrm{absolute deviation of } Fe^{3+} / \Sigma Fe$$'
            fig2.y_range = Range1d(0, 1.3 * yData1.max())
            
            
            fig3 = figure(title='2TAPL', width=400, height=250)
            for i in [1, 2]:
                del tmp
                tmp = st.session_state.dfMeasSmpDataTAP2.copy()
                tmp[r'L$\beta$/L$\alpha$ (TAP2)'] = tmp[r'L$\beta$/L$\alpha$ (TAP2)'] * .01 * (100 - i)
                yData3 = regressionFitParameters(st.session_state.dfMeasSmpDataTAP2, 'TAP2') + regressionFitParameters(tmp, 'TAP2')
                fig3.line(range(len(yData3)), yData3, color=colorList[i-1], legend_label = str(i) + '%')
        
            fig3.xaxis.axis_label='sample'
            fig3.yaxis.axis_label=r'$$\textrm{absolute deviation of } Fe^{3+} / \Sigma Fe$$'
            fig3.y_range = Range1d(0, 1.3 * yData3.max())
            
            
            fig4 = figure(title='4TAPL', width=400, height=250)
            for i in [1, 2]:
                del tmp
                tmp = st.session_state.dfMeasSmpDataTAP4.copy()
                tmp[r'L$\beta$/L$\alpha$ (TAP4)'] = tmp[r'L$\beta$/L$\alpha$ (TAP4)'] * .01 * (100 - i)
                yData4 = regressionFitParameters(st.session_state.dfMeasSmpDataTAP4, 'TAP4') + regressionFitParameters(tmp, 'TAP4')
                fig4.line(range(len(yData4)), yData4, color=colorList[i-1], legend_label = str(i) + '%')
        
            fig4.xaxis.axis_label='sample'
            fig4.yaxis.axis_label=r'$$\textrm{absolute deviation of } Fe^{3+} / \Sigma Fe$$'
            fig4.y_range = Range1d(0, 1.3 * yData3.max())
    
    
            grid_layout = gridplot([fig1, fig2, fig3, fig4], ncols=2)
            st.bokeh_chart(grid_layout)
 
        
        
    ##-----------------------------------------------##
    ##---------------  sample s.d.  ----------------##
    ##-----------------------------------------------##
    
        def errorSmpFe3Dev():
            from bokeh.plotting import figure
            import numpy as np
            
            fig = figure(title='Samples', width=400, height=250)

            LRatioSmp = []
            for smpname in st.session_state.smpList:
                fil = st.session_state.dfMain['Name'] == smpname
                r = st.session_state.dfMain[fil][r'L$\beta$ (TAP2)']/st.session_state.dfMain[fil][r'L$\alpha$ (TAP2)']
                LRatioSmp.append(np.std(r))
        
            fig.line(range(len(LRatioSmp)), LRatioSmp)
            fig.xaxis.axis_label='sample'
            fig.yaxis.axis_label=r'abs. 1 s.d. of Lb/Ls of a single sample'
            #fig.set_ylim(0,.025)
            st.bokeh_chart(fig)
        
        
            drList = st.session_state.dfdr['Name'].drop_duplicates().tolist()
            
            fig = figure(title='Drift Monitor', width=400, height=250)
            LRatioDrift = []
            for smpname in drList:
                fil = st.session_state.dfMain['Name'] == smpname
                r = st.session_state.dfMain[fil][r'L$\beta$ (TAP2)']/st.session_state.dfMain[fil][r'L$\alpha$ (TAP2)']
                LRatioDrift.append(np.std(r))
        
            fig.line(range(len(LRatioDrift)), LRatioDrift)
            fig.xaxis.axis_label='sample'
            fig.yaxis.axis_label=r'abs. 1 s.d. of Lb/La of a single sample'
            st.bokeh_chart(fig)
            

#------------------
#---- Start Website
#------------------

        sel = st.radio('', ('How Fe3+ changes when Fetot and Lb/La change', 
                            'abs. 1 s.d. of Lb/La of Samples and Drfit Monitor'), horizontal = True)
        
        if sel == 'How Fe3+ changes when Fetot and Lb/La change':
            errorPercentDeviations()
        else:
            errorSmpFe3Dev()
            
#--------  End Error Considerations



#----------------------------------
#--------- Visualisations Side Bar
#----------------------------------

    plotSel = st.sidebar.radio('Select your Detail', ('Drift Inspection', 'Comparing La & Lb', 'Parametrisation', 'Sample Inspection', 'Error Considerations'))
    
    if plotSel == 'Drift Inspection':
        st.subheader('Drift Inspection')
        sel = st.radio('Choose what to inspect', ('Composition of drift standards', 'Fe3+ using 2TAPL vs. Fe3+ using 4TAPL'), horizontal=True)
        driftplots(sel)
    elif plotSel == 'Comparing La & Lb':
        st.subheader('Comparing La & Lb')
        comparinglalphalbeta()
    elif plotSel == 'Parametrisation':
        st.subheader('Parametrisation')
        parametrisationplot()
    elif plotSel == 'Sample Inspection':
        st.subheader('Sample Inspection')
        sel = st.selectbox('Select', ('Select one element, display all samples', 'Select one sample, display all elements', 'Select one sample and one element'))
        sampleInspection(sel)
    elif plotSel == 'Error Considerations':
        st.subheader('Error Considerations')
        errorConsiderations()
        
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             Use the various visualisation tools to analyse your data and optimise your upload file. 
             Check the 'Tutorials & Instructions' resource on how to do this.
         """)
         
    with st.sidebar:
        with st.expander("Info Drift Inspection"):
         st.write("""
             Check the composition of the dirft monintor measurements and identify the stability of/variations duringduring the measurement campaign.
             Also check how the Fe3+ abundances of the two analyser crystals compare.
         """)
                  
    with st.sidebar:
        with st.expander("Info Comparing La & Lb"):
         st.write("""
             The plots provide insights to potential issues during the measurements.
         """)
                  
    with st.sidebar:
        with st.expander("Info Parametrisation"):
         st.write("""
             The plot visualises the formula, with which the Fe3+ in the samples are calculated.
         """)
         
    with st.sidebar:
        with st.expander('Info Sample Inspection'):
            st.write("""
                This provides comprehensive possibilities to check the composition of all samples in various overviews to high granularity. 
            """)
            
    with st.sidebar:
        with st.expander('Info Error Considerations'):
            st.write("""
        How Fe3+ changes when Fetot and Lb/La change-- Individual samples are plotted along the x-axis.\
        For each sample, the Fetot (top 2 plots) and\
        Lbeta/Lalpha (bottom 2 plots) are changed by the percentage\
        given in the legend. The Fe3+/SumFe is then calculated with\
        the new Fetot or Lbeta/Lalpha. The result is then subtracted\
        from the true Fe3+/SumFe and plotted on the y-axis.\
          ---  sample s.d.--
        Individual samples/drift monitors are plotted along the x-axis.\
        The 1 s.d. of Lbeta/Lalpha of a single sample is calculated and\
        plotted on the y-axis.
            """)
           

#-----------------------------------------#
#------------ Start Individual Fe3+ & Fe2+ calculation
#-----------------------------------------#
def individualFe3Fe2Calculation():
    import pandas as pd
    
    st.markdown(f"# {list(page_names_to_funcs.keys())[4]}")

    
    A = st.number_input('Insert parameter A', value=st.session_state.fitParametersTAP2[0])
    B = st.number_input('Insert parameter B', value=st.session_state.fitParametersTAP2[1])
    C = st.number_input('Insert parameter C', value=st.session_state.fitParametersTAP2[2])
    D = st.number_input('Insert parameter D', value=st.session_state.fitParametersTAP2[3])
    fetot = st.number_input('Insert the Fetot of the sample', value=20.6)
    Lb_La = st.number_input('Insert the Lb/La of the sample', value=1.4)
    
    
    if st.button('Calculate Fe2+ & Fe3+'):
        st.write('$$Fe2+$$: ' + str(round(A + B * Lb_La + C * fetot + D * fetot * Lb_La, 3)) + ' wt%')
        st.write('$$Fe3+$$: ' + str(round(-A - B * Lb_La - C * fetot - D * fetot * Lb_La + fetot, 3)) + ' wt%')
        
    
    st.markdown('<h4 style="color:blue"><b>Calculated Fit Parameters for 2TAPL & 4TAPL</b> </h4>', unsafe_allow_html=True)
    st.write(pd.DataFrame({'Parameter':['A', 'B', 'C', 'D'],
                           'TAP2':st.session_state.fitParametersTAP2,
                           'TAP4':st.session_state.fitParametersTAP4}))
    
    st.markdown('<h4 style="color:black"><b>Formulas to calculate Fe2+ and Fe3+</b> </h4>', unsafe_allow_html=True)
    st.latex(r'''Fe^{2+} = A + B \times \frac{L\beta}{L\alpha} + C \times \Sigma Fe + D \times \Sigma Fe \times \frac{L\beta}{L\alpha}''')
    st.latex(r'''Fe^{3+} = -A - B \times \frac{L\beta}{L\alpha} - C \times \Sigma Fe - D \times \Sigma Fe \times \frac{L\beta}{L\alpha} + Fe_{tot}''')
    st.latex(r'''\textrm{The result is } Fe^{2+} \textrm{ or } Fe^{3+} \textrm{, respectively, in wt\%} ''')

        
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
        Input/change the parameters to test how the result depends on the respective, individual parameters.
         """)

#-----------------------------------------#
#------------ Start Output ---------------#
#-----------------------------------------#
def outputForm():
    
    st.markdown(f"# {list(page_names_to_funcs.keys())[5]}")
    st.subheader('This will come soon:')
    st.write('Output of the tables from Result Tables in one file')
    st.write('Output all results in a standard file')

    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             See previews of output files and download these with one click on the respective buttons.
         """)

#-----------------------------------------#
#--------- Start Tutorials & Instructions #
#-----------------------------------------#
def tutorials_instructions():
    
    tutorialSel = st.sidebar.radio('Select your tutorial:', 
                    ('Introduction', 'Videos', 'Text Material', 'Course', 'Documentation'))
    
    if tutorialSel == 'Introduction':
        st.header('Introduction')
        st.subheader('A start')
        st.write('''This site contains various resources. Check the panel on the left for e.g., 
                 video tutorials on how to use this flank data reduction online resource.''')


        st.subheader('Download your test dataset')
        st.download_button(
             label="Download Test Dataset",
             data='https://raw.githubusercontent.com/Hezel2000/microprobe/main/Flank%20Method%20Test%20Dataset.csv',
             file_name='Flank Method Test Dataset.csv',
             mime='text/csv',
         )
        st.write('Something is currenlty not working. Will be solved soon.')
        
    
    elif tutorialSel == 'Videos':
        st.header('Video Tutorials')
        
        videoList = ('Preparing the data file', 'Flank Data Reduction', 'The Flank Method',
                     'Microprobe Peak Position')
        videoLinkList = ('https://youtu.be/y-nVEMzy2gU', 'https://youtu.be/ub1wg-abesc', 'https://youtu.be/I0pjV1mEfSc', 'https://youtu.be/JB1zyITkBtM')
        selChapter = st.selectbox('Choose your tutorial', videoList)  
        if selChapter == videoList[0]:
            st.subheader(videoList[0])
            st.video(videoLinkList[0])    
        elif selChapter == videoList[1]:
            st.subheader(videoList[1])
            st.video(videoLinkList[1])    
        elif selChapter == videoList[2]:
            st.subheader(videoList[2])
            st.write('coming soon')
            #st.video(videoLinkList[2])    
        elif selChapter == videoList[3]:
            st.subheader(videoList[3])
            st.video(videoLinkList[3])  
        
    elif tutorialSel == 'Text Material':
        st.header('in preparation')
                
    elif tutorialSel == 'Course':
        st.header('in preparation')
        st.write('This will become a comprehensive course teaching the flank method.')
                
    elif tutorialSel == 'Documentation':
        st.header('The readme.md file from GitHub')
        st.write('https://github.com/Hezel2000/microprobe/blob/0498d9233fff2e2659e24028ad1beefc764c1cbb/readme.md')
                
        st.header('The GitHub repository for this resource')
        st.write('https://github.com/Hezel2000/microprobe/tree/0498d9233fff2e2659e24028ad1beefc764c1cbb')
                
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             Here you will find everything from how to use this site to reduce your data, up to
             how does the flank method work.
         """)


#-----------------------------------------#
#------------ Start Method & References --#
#-----------------------------------------#
def method_references():
    
    st.markdown(f"# {list(page_names_to_funcs.keys())[7]}")

    st.header('Method')
    st.write('In preparation.')
    
    st.header('References')
    st.write('Höfer H. E. and Brey G. P. (2007) The iron oxidation state of garnet by electron microprobe: Its determination with the flank method combined with major-element analysis. Am Mineral 92, 873–885.')
    st.write('Höfer H. E. (2002) Quantification of Fe2+/Fe3+ by Electron Microprobe Analysis – New Developments. Hyperfine Interact 144–145, 239–248.')
    st.write('Höfer H. E., Brey, G. P., and Hibberson, W. O. (2004) Iron oxidation state determination in synthetic pyroxenes by electron microprobe. Lithos, 73, 551.')
    st.write('Höfer H. E., Weinbruch S., Mccammon C. A. and Brey G. P. (2000) Comparison of two electron probe microanalysis techniques to determine ferric iron in synthetic wüstite samples. Eur J Mineral 12, 63–71.')
    st.write('Höfer, H. E., Brey, G. P., and Oberh‰nsli, R. (1996) The determination of the oxidation state of iron in synthetic garnets by X-ray spectroscopy with the electron microprobe. Physics and Chemistry of Minerals, 23, 241.')
    st.write('Höfer, H. E., Brey, G. P., Schulz-Dobrick, B., and Oberh‰nsli, R. (1994) The determination of the oxidation state of iron by the electron microprobe. European Journal of Mineralogy, 6, 407-418.')
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             vid
         """)

#------------ End Method & References


#-----------------------------------------#
#------------ Start Tools ----------------#
#-----------------------------------------#
def tools():
    import pandas as pd
    from bokeh.plotting import figure
    from bokeh.models import Span
    
    #st.markdown(f"# {list(page_names_to_funcs.keys())[8]}")

    st.header('Determining the flank positions from difference spectra')
    
    dfFeLSpectra = pd.read_csv('https://raw.githubusercontent.com/Hezel2000/microprobe/main/Fe%20Spektren%2020220620.csv')


    fig = figure(width=600, height=400)
    
    crystal = st.selectbox('Spectrometer & Crystal', ('2TAPL', '4TAPL'))
    pha = st.selectbox('PHA', ('int', 'diff'))
    acm = st.selectbox('Accumulations', ('1 Acm', '4 Acm'))
    lower_flank_pos, upper_flank_pos = st.slider('Adjust the lower (Lb) and upper (La) flank measurement positions', 187.0, 192.0, (188.0, 191.0), key=0)
    
    if (pha == 'diff') & (acm == '4 Acm'):
        st.write('not available')
    else:    
        fig.line(dfFeLSpectra['L-value'], dfFeLSpectra['AlmO, ' + pha + ' (' + crystal + '), ' + acm], color='green', legend_label='AlmO, int (' + crystal + ')')
        fig.line(dfFeLSpectra['L-value'], dfFeLSpectra['And, ' + pha + ' (' + crystal + '), ' + acm], color='blue', legend_label='And, int (' + crystal + ')')
        fig.line(dfFeLSpectra['L-value'], dfFeLSpectra['AlmO, ' + pha + ' (' + crystal + '), ' + acm] - dfFeLSpectra['And, ' + pha + ' (' + crystal + '), ' + acm], color='orange', legend_label='difference spectra')
        vline_lower = Span(location= lower_flank_pos, dimension='height', line_color='grey', line_dash='dashed', line_width=2)
        vline_upper = Span(location= upper_flank_pos, dimension='height', line_color='grey', line_dash='dashed', line_width=2)
        fig.renderers.extend([vline_lower, vline_upper])
        fig.xaxis.axis_label='L-value (mm)'
        fig.yaxis.axis_label='counts'
        fig.add_layout(fig.legend[0], 'below')
        
        st.bokeh_chart(fig)
        
        st.write('Hi Jie, you are on the internet now!')
    
    
# =============================================================================
#     fig = figure(width=600, height=400)
#     
#     selSpectra = st.multiselect('Crystal', dfFeLSpectra.columns, dfFeLSpectra.columns[1])
#     st.write(dfFeLSpectra.columns[:4])
#     st.write(selSpectra)
#     
#     plotList=[]
#     for i in selSpectra:
#         fig.line(dfFeLSpectra['L-value'], dfFeLSpectra[selSpectra[0]], color='olive')
#         pl=plotList.append(fig)
#     
#     st.bokeh_chart(pl)
# =============================================================================
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             vid
         """)

#------------ End Method & References


#-----------------------------------------#
#------------ Start Main Page Definitions #
#-----------------------------------------#

page_names_to_funcs = {
    'Start & upload Data': start,
    'Data Reduction': dataReduction,
    'Result Tables': resultTables,
    'Visualisations': visualisations,
    'Calculate individual Fe2+ & Fe3+': individualFe3Fe2Calculation,
    'Output': outputForm,
    'Tutorials & Instructions': tutorials_instructions,
    'Method & References': method_references,
    'Tools': tools
}

demo_name = st.sidebar.radio("Start your flank method analysis journey here", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

