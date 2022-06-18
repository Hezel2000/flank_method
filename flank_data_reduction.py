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
        st.session_state.smpList = list(set(st.session_state.dfSampleNames.tolist()) - set(st.session_state.stdSelection))

        # First, the indices of the standard measurements must be input from. These are found in the dfSampleNames above
        #st.session_state.stdList = st.session_state.dfSampleNames.iloc[standards].tolist()

        # Extracting FeO and Lalpha/Lbeta, the Lbeta/Lalpha ratios are calculated from the measured Lbeta and Lalpha cps, and the data are averaged
        #st.session_state.dfMeasStdDataTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdList, 'TAP2')
        st.session_state.dfMeasStdDataTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdSelection, 'TAP2')
        st.session_state.dfMeasSmpDataTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.smpList, 'TAP2')
        #st.session_state.dfMeasStdDataTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdList, 'TAP4')
        st.session_state.dfMeasStdDataTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdSelection, 'TAP4')
        st.session_state.dfMeasSmpDataTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.smpList, 'TAP4')
        
        # Combining measured standard data and required known Fe2+ and Fetot from standard data (-> Moessbauer data)
        combMoessAndMeasStdData = []
        for i in st.session_state.dfMeasStdDataTAP2['Name']:
            res = extractKnownFe2(i.split('_')[0])
            combMoessAndMeasStdData.append(res)


        dfFitData = pd.concat([st.session_state.dfMeasStdDataTAP2, st.session_state.dfMeasStdDataTAP4[r'L$\beta$/L$\alpha$ (TAP4)']], axis = 1)
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
        resultsFe3SmpFPTAP2 = pd.DataFrame(regressionFitParameters(st.session_state.dfMeasSmpDataTAP2, 'TAP2'))
        resultsFe3StdFPTAP4 = pd.DataFrame(regressionFitParameters(st.session_state.dfMeasStdDataTAP4, 'TAP4'))
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
                                 ,r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)', r'$\Delta$ Meas - Moess (TAP2)'
                                 ,r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)', r'$\Delta$ Meas - Moess (TAP4)'
                                ])
        st.session_state.resultsFe3Smp = pd.concat([st.session_state.dfMeasSmpDataTAP2['Point Nr.'], st.session_state.dfMeasSmpDataTAP2['Name']
                        , st.session_state.dfMeasSmpDataTAP2[r'Fe$_{tot}$']
                        ,resultsFe3SmpFPTAP2[0]
                        ,resultsFe3SmpFPTAP4[0]
                        ,resultsFe3SmpFPTAP2[0]-resultsFe3SmpFPTAP4[0]], axis = 1,
                         keys = ['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)'
                                 ,r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)', r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)', 'TAP2-TAP4'])
   

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
    import pandas as pd
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             1- Choose which data shall be reduced. 2- Select the standards used to claculte the fit parameters. 
             Note that you can only choose those also present in the Moessbauer Standard Data file. 3- Click on 'Calculate Results'. 
             4- You can change your selection of standards or whether to use all or only the inspected data anytime.
             However, after each change you need to click 'Calculate Results' again.
             5- Proceed to 'Result Tables'.
         """)

         
    st.subheader('Results for ' + '$$Fe^{3+}/ \Sigma Fe$$' + ' in the Standards')
    st.write(r'$Fe^{3+}/\Sigma Fe$ deviation from the Moessbauer data should be <0.01-0.015')
    st.write(st.session_state.resultsFe3Std.round(3))
    
    st.subheader('Results for ' + '$$Fe^{3+}/ \Sigma Fe$$' + ' in the Samples')
    st.write(r'The error on $Fe^{3+}/\Sigma Fe$ in the Smp is 0.02') 
    st.write(st.session_state.resultsFe3Smp.round(3))


#-----------------------------------------#
#------------ Start Visualisations--------#
#-----------------------------------------#
def visualisations():
    import streamlit as st
    import pandas as pd
    #from bokeh.plotting import figure, output_file, show
#    from bokeh.models import Panel, Tabs

    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             1- Choose which data shall be reduced. 2- Select the standards used to claculte the fit parameters. 
             Note that you can only choose those also present in the Moessbauer Standard Data file. 3- Click on 'Calculate Results'. 
             4- You can change your selection of standards or whether to use all or only the inspected data anytime.
             However, after each change you need to click 'Calculate Results' again.
             5- Proceed to 'Result Tables'.
         """)
         

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
        
        if sel == 'elements':
            el = st.selectbox('Select', elements)
    
            av = np.average(st.session_state.dfdr[el])
            std = np.std(st.session_state.dfdr[el])
            
            reldev = 100 * np.std(st.session_state.dfdr[el])/np.average(st.session_state.dfdr[el])
        #                elif sel == 'elements  ':
            if reldev < 1:
                fcColor = (.5, 0.8, 0)
            elif 1 <= reldev < 5:
                fcColor = 'orange'
            else:
                fcColor = 'r'
              
            col1, col2 = st.columns([3, 1])
            col1.subheader('Drift Monitor Visualisation')
            #st.write(st.session_state.dfdr.to_dict())
            
            TOOLTIPS = [('Name', '@Name'),
                        ('Point Nr.', '@{Point Nr.}'),
                        (el, '@'+el)]
            
            fig = figure(width=500, height=300, tooltips = TOOLTIPS)
                        
            fig.line(st.session_state.dfdr.loc[:, 'Point Nr.'], st.session_state.dfdr.loc[:, el])
            #fig.circle(st.session_state.dfdr.loc[:, 'Point Nr.'], st.session_state.dfdr.loc[:, el])
            output_file("toolbar.html")  
            source = ColumnDataSource(data = st.session_state.dfdr.to_dict('list'))
            
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
            redDot = "https://static.streamlit.io/examples/dice.jpg"
            redCaption = 'worrisome data'
            greenDot = 'https://github.com/Hezel2000/microprobe/blob/2c4fa6b09e359b1abd7346d6c054a4d9d2d408f3/red_dot.svg'
            col2.image(redDot, caption = redCaption, width=64)
            st.color_picker('worst', '#FF0000')
        
        else:
            dfdrLRatioTAP2 = st.session_state.dfdr[r'L$\beta$ (TAP2)']/st.session_state.dfdr[r'L$\alpha$ (TAP2)']
            dfdrLRatioTAP4 = st.session_state.dfdr[r'L$\beta$ (TAP4)']/st.session_state.dfdr[r'L$\alpha$ (TAP4)']
            dfdrCalList = pd.concat([st.session_state.dfdr['FeO'], dfdrLRatioTAP2], axis = 1)
            dfdrCalList = dfdrCalList.rename(columns = {'FeO': r'Fe$_{tot}$', 0:r'L$\beta$/L$\alpha$ (TAP2)'})
            dfdrCalList4 = pd.concat([st.session_state.dfdr['FeO'], dfdrLRatioTAP4], axis = 1)
            dfdrCalList4 = dfdrCalList4.rename(columns = {'FeO': r'Fe$_{tot}$', 0:r'L$\beta$/L$\alpha$ (TAP4)'})
    
            fig = figure(width=600, height=400)
            fig.scatter(regressionFitParameters(dfdrCalList, 'TAP2'),
                        regressionFitParameters(dfdrCalList4, 'TAP4'))
            fig.xaxis.axis_label=r'''Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)'''
            fig.yaxis.axis_label=r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)'
            st.bokeh_chart(fig)
            
#--------  End Drift Inspection

#--------  Start Comparing Lalpha & Lbeta

    def comparinglalphalbeta():
        from bokeh.plotting import figure
        #from bokeh.models import Span, BoxAnnotation, Label
        import numpy as np
        
        st.subheader('first plot')
        fig = figure(width=500, height=300)
        tapl2Fe3 = st.session_state.resultsFe3Smp[r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)']
        tapl4Fe3 = st.session_state.resultsFe3Smp[r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)']
        fig.scatter(tapl2Fe3, tapl4Fe3)
        
        x = np.linspace(0,1,10)
        fig.line(x, x)
        fig.line(x, x + .02, line_dash='dashed', line_color='orange')
        fig.line(x, x - .02, line_dash='dashed', line_color='orange')
        
        fig.xaxis.axis_label=r"$$Fe^{3+} / \Sigma Fe \textrm{ (2TAPL)}$$"
        fig.yaxis.axis_label=r'$$Fe^{3+} / \Sigma Fe \textrm{ (4TAPL)}$$'
        st.bokeh_chart(fig)
        
        
        st.subheader('second plot')
        fig = figure(width=500, height=300)
        tapl2Betacps = st.session_state.dfMain[r'L$\beta$ (TAP2)']
        tapl2Alphacps = st.session_state.dfMain[r'L$\alpha$ (TAP2)']
        tapl4Betacps = st.session_state.dfMain[r'L$\beta$ (TAP4)']
        tapl4Alphacps = st.session_state.dfMain[r'L$\alpha$ (TAP4)']
        
        fig.scatter(tapl2Betacps, tapl2Alphacps)
        fig.scatter(tapl4Betacps, tapl4Alphacps)
        fig.xaxis.axis_label=r'$$L\beta \textrm{ (net intensities)}$$'
        fig.yaxis.axis_label=r'$$L\alpha \textrm{ (net intensities)}$$'
        #ax2.legend()
        st.bokeh_chart(fig)
        
        
        st.subheader('3')
        fig = figure(width=500, height=300)
        fig.line(st.session_state.dfMain['Point Nr.'], (tapl2Betacps/tapl2Alphacps)/(tapl4Betacps/tapl4Alphacps))
        fig.xaxis.axis_label='Point Nr.'
        fig.yaxis.axis_label=r'$$L\beta/L\alpha \textrm{ (2TAPL)} /L\beta/L\alpha \textrm{ (4TAPL)}$$'
        st.bokeh_chart(fig)
        

        st.subheader('4')
        fig = figure(width=500, height=300)
        fig.scatter(st.session_state.dfMain['Point Nr.'], tapl2Betacps/tapl2Alphacps, legend_label='2TAPL')
        fig.scatter(st.session_state.dfMain['Point Nr.'], tapl4Betacps/tapl4Alphacps, color='olive', legend_label='4TAPL')
        fig.xaxis.axis_label='Point Nr.'
        fig.yaxis.axis_label=r'$$L\beta/L\alpha \textrm{ (counts-ratio)}$$'
        st.bokeh_chart(fig)


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
    
#-------- End Parametrisation

#-------- Start Sample Inspection

    def sampleInspection(sel):
        from bokeh.plotting import figure
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
            noc = st.number_input('Insert the Number of Plot Columns', value=4)
            
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
            
            fil = (st.session_state.dfMain['Name'] == smp) & (st.session_state.dfMain[el])
            xdata = st.session_state.dfMain[fil].loc[:, 'Point Nr.']
            data = st.session_state.dfMain[fil].loc[:, el]
            dat = (el, xdata, data)
            
            st.bokeh_chart(plotStyle(dat))

#-------- End Sample Inspection

#--------  Start Error Considerations

    def errorConsiderations():
        st.write('result variations:\
        Individual samples are plotted along the x-axis.\
        For each sample, the Fetot (top 2 plots) and\
        Lbeta/Lalpha (bottom 2 plots) are changed by the percentage\
        given in the legend. The Fe3+/SumFe is then calculated with\
        the new Fetot or Lbeta/Lalpha. The result is then subtracted\
        from the true Fe3+/SumFe and plotted on the y-axis.\
        sample s.d.:\
        Individual samples/drift monitors are plotted along the x-axis.\
        The 1 s.d. of Lbeta/Lalpha of a single sample is calculated and\
        plotted on the y-axis.')
        
                    
        ##-----------------------------------------------##
        ##-------------  result variations --------------##
        ##-----------------------------------------------##
        
        def errorPercentDeviations():
            global yData1
            global yData2
            global yData3
            global yData4
            
            fig = plt.figure(figsize = (10, 7))
            gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
            ((ax1, ax2), (ax3, ax4)) = gs.subplots(sharex = True, sharey = True)
            tmp=1
            for i in range(-2, 3):
                del tmp
                tmp = dfMeasSmpDataTAP2.copy()
                tmp[r'Fe$_{tot}$'] = tmp[r'Fe$_{tot}$'] * 1 - .1 * i
                yData1 = regressionFitParameters(dfMeasSmpDataTAP2, 'TAP2') - regressionFitParameters(tmp, 'TAP2')
                ax1.plot(yData1, label = '+'+str(int(i *10)) + '%')
        
            ax1.set_title('TAP2')
            #ax1.set_xlabel('sample')
            ax1.set_ylabel(r'absolute deviation of Fe$^{3+}$/$\Sigma$Fe')
            ax1.legend()
        
        
            for i in range(-2, 3):
                del tmp
                tmp = dfMeasSmpDataTAP4.copy()
                tmp[r'Fe$_{tot}$'] = tmp[r'Fe$_{tot}$'] * 1 - .1 * i
                yData2 = regressionFitParameters(dfMeasSmpDataTAP4, 'TAP4') - regressionFitParameters(tmp, 'TAP4')
                ax2.plot(yData2, label = '+'+str(int(i *10)) + '%')
        
            ax2.set_title('TAP4')
            ax2.set_xlabel('sample')
            #ax2.set_ylabel(r'absolute deviation of Fe$^{3+}$/$\Sigma$Fe')
            ax2.legend()
        
            for i in range(-2, 3):
                del tmp
                tmp = dfMeasSmpDataTAP2.copy()
                tmp[r'L$\beta$/L$\alpha$ (TAP2)'] = tmp[r'L$\beta$/L$\alpha$ (TAP2)'] * 1 - .01 * i
                yData3 = regressionFitParameters(dfMeasSmpDataTAP2, 'TAP2') - regressionFitParameters(tmp, 'TAP2')
                ax3.plot(yData3, label = '+'+str(int(i)) + '%')
        
            ax3.set_title('TAP2')
            ax3.set_xlabel('sample')
            ax3.set_ylabel(r'absolute deviation of Fe$^{3+}$/$\Sigma$Fe')
            ax3.legend()
        
        
            for i in range(-2, 3):
                del tmp
                tmp = dfMeasSmpDataTAP4.copy()
                tmp[r'L$\beta$/L$\alpha$ (TAP4)'] = tmp[r'L$\beta$/L$\alpha$ (TAP4)'] * 1 - .01 * i
                yData4 = regressionFitParameters(dfMeasSmpDataTAP4, 'TAP4') - regressionFitParameters(tmp, 'TAP4')
                ax4.plot(yData4, label = '+'+str(int(i)) + '%')
        
            ax4.set_title('TAP4')
            ax4.set_xlabel('sample')
            #ax4.set_ylabel(r'absolute deviation of Fe$^{3+}$/$\Sigma$Fe')
            ax4.legend()
        
            plt.show()
        
        
        ##-----------------------------------------------##
        ##---------------  sample s.d.  ----------------##
        ##-----------------------------------------------##
        
        def errorSmpFe3Dev():
            fig = plt.figure(figsize = (10, 3))
            gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
            (ax1, ax2) = gs.subplots(sharey = True)
        
            LRatioSmp = []
            for smpname in smpList:
                fil = df['Name'] == smpname
                r = df[fil][r'L$\beta$ (TAP2)']/df[fil][r'L$\alpha$ (TAP2)']
                LRatioSmp.append(np.std(r))
        
            ax1.plot(LRatioSmp)
            ax1.set_title('Samples')
            ax1.set_xlabel('sample')
            ax1.set_ylabel(r'abs. 1 s.d. of L$\beta$/L$\alpha$ of a single sample')
            ax1.set_ylim(0,.025)
        
        
            drList = dfdr['Name'].drop_duplicates().tolist()
        
            LRatioDrift = []
            for smpname in drList:
                fil = df['Name'] == smpname
                r = df[fil][r'L$\beta$ (TAP2)']/df[fil][r'L$\alpha$ (TAP2)']
                LRatioDrift.append(np.std(r))
        
            ax2.plot(LRatioDrift)
            ax2.set_title('Drift Monitor')
            ax2.set_xlabel('sample')
        
            return plt.show()
        

#--------  End Error Considerations


#----------------------------------
#--------- Visualisations Side Bar
#----------------------------------

    
    st.sidebar.markdown("### Visualisations")
    plotSel = st.sidebar.radio('Select your Detail:', ('Drift Inspection', 'Comparing La & Lb', 'Parametrisation', 'Sample Inspection', 'Error Considerations'))
    
    if plotSel == 'Drift Inspection':
        st.subheader('Drift Inspection')
        sel = st.selectbox('Select', ('elements', 'Fe3+'))
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


#-----------------------------------------#
#------------ Start Tutorials & Instructions #
#-----------------------------------------#
def tutorials():
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             1- Choose which data shall be reduced. 2- Select the standards used to claculte the fit parameters. 
             Note that you can only choose those also present in the Moessbauer Standard Data file. 3- Click on 'Calculate Results'. 
             4- You can change your selection of standards or whether to use all or only the inspected data anytime.
             However, after each change you need to click 'Calculate Results' again.
             5- Proceed to 'Result Tables'.
         """)
    
    st.markdown(f"# {list(page_names_to_funcs.keys())[4]}")
    st.sidebar.markdown('### Tutorials')
    tutorialSel = st.sidebar.radio('Select your tutorial:', ('Introduction', 'Overview'))
    
    if tutorialSel == 'Introduction':
        st.header('Some general Intro will come soon.')
        st.write('stay tuned!')
    elif tutorialSel == 'Overview':
        st.header('Overview')
        st.video('https://youtu.be/WXv79tpor5s')
        
#------------ End Tutorials & Instructions


#-----------------------------------------#
#------------ Start Method & References --#
#-----------------------------------------#
def method():
    
    with st.sidebar:
        with st.expander("Instructions for this site"):
         st.write("""
             1- Choose which data shall be reduced. 2- Select the standards used to claculte the fit parameters. 
             Note that you can only choose those also present in the Moessbauer Standard Data file. 3- Click on 'Calculate Results'. 
             4- You can change your selection of standards or whether to use all or only the inspected data anytime.
             However, after each change you need to click 'Calculate Results' again.
             5- Proceed to 'Result Tables'.
         """)
    
    st.markdown(f"# {list(page_names_to_funcs.keys())[5]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

#------------ End Method & References


#-----------------------------------------#
#------------ Start Main Page Definitions #
#-----------------------------------------#

page_names_to_funcs = {
    'Start & upload Data': start,
    'Data Reduction': dataReduction,
    'Result Tables': resultTables,
    'Visualisations': visualisations,
    'Tutorials & Instructions': tutorials,
    'Method & References': method
}

demo_name = st.sidebar.radio("Go through your data analysis here", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

