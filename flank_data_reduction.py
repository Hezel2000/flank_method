#!/usr/bin/env python
# coding: utf-8

import streamlit as st

    
def start():
    import streamlit as st
    import pandas as pd
    
#------------ Start Test for Duplicates
    
    def reportDuplicatesInList(l):
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
    
        return res
#------------ End Test for Duplicates

#------------ Start Prepare Dataset
    def prepareDataset():
        dfComplete = st.session_state.dfRaw
        for ind in dfComplete.index:
            measurementPointName = dfComplete['Comment'][ind]
    
            if 'Grid' in measurementPointName:
                dfComplete['Comment'] = dfComplete['Comment'].replace([measurementPointName],measurementPointName.split('Grid')[0])
            elif 'Line' in measurementPointName:
                dfComplete['Comment'] = dfComplete['Comment'].replace([measurementPointName],measurementPointName.split('Line')[0])
    
    
        df = dfComplete.loc[:, ['Point', 'Comment', 'SiO2(Mass%)', 'TiO2(Mass%)', 'Al2O3(Mass%)', 'Cr2O3(Mass%)', 'FeO(Mass%)', 'MnO(Mass%)',
                        'NiO(Mass%)', 'MgO(Mass%)',  'CaO(Mass%)',  'Na2O(Mass%)', 'K2O(Mass%)', 'P2O5(Mass%)', 'Total(Mass%)',
                        'Bi(Net)', 'Ar(Net)', 'Br(Net)', 'As(Net)', 'Current']]
    
    
        df = df.rename(columns = {'Point':'Point Nr.', 'Comment':'Name', 'SiO2(Mass%)':'SiO2', 'TiO2(Mass%)':'TiO2', 'Al2O3(Mass%)':'Al2O3',
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
        global dfdr
        global dfSampleNames
    
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
##-----------  Pre-processsing data  ------------##
##-----------------------------------------------##
        
# Command for getting Fe2+ and Fetot values from the dfMoss dataset
    def extractKnownFe2(stdNameForMatching):
        foundStd = st.session_state.dfMoess[st.session_state.dfMoess['Name'].str.contains(stdNameForMatching)]
        Fe2Value = foundStd['FeO (wt%)'].tolist()[0] * 55.845/(55.845 + 15.9994)
        Fe2ModAbValue = foundStd['Fe2+/SumFe'].tolist()[0]
        return Fe2ModAbValue


    def preProcessingData():        
        # Getting the indices of the samples and standards
        samplesListReIndexed = pd.Series(st.session_state.dfSampleNames.tolist())

        fil = samplesListReIndexed.str.contains('AlmO') | samplesListReIndexed.str.contains('UA5') | samplesListReIndexed.str.contains('UA10') | samplesListReIndexed.str.contains('Damknolle')
                
        samples = samplesListReIndexed[~fil].index.values.tolist()
        standards = samplesListReIndexed[fil].index.values.tolist()
        
        # Getting sample data
        st.session_state.smpList = st.session_state.dfSampleNames.iloc[samples].tolist()

        # First, the indices of the standard measurements must be input from. These are found in the dfSampleNames above
        st.session_state.stdList = st.session_state.dfSampleNames.iloc[standards].tolist()

        # Extracting FeO and Lalpha/Lbeta, the Lbeta/Lalpha ratios are calculated from the measured Lbeta and Lalpha cps, and the data are averaged
        st.session_state.dfMeasStdDataTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdList, 'TAP2')
        st.session_state.dfMeasSmpDataTAP2 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.smpList, 'TAP2')
        st.session_state.dfMeasStdDataTAP4 = extractAndCalculateAverages(st.session_state.dfMain, st.session_state.stdList, 'TAP4')
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
##------  Fit Parameter linear regression  ------##
##-----------------------------------------------##
        
    def regressionFitParameters(inpData, crystal):
        import numpy as np
        
        data = inpData
        if crystal == 'TAP2':
            crystalName = ' (TAP2)'
        else:
            crystalName = ' (TAP4)'
        
        x = st.session_state.dfFitData[r'L$\beta$/L$\alpha$' + crystalName][:4]
        y = st.session_state.dfFitData[r'Fe$_{tot}$'][:4]
        z = st.session_state.dfFitData[r'Fe$^{2+}$'][:4]
    
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
   

#----------------------
    
    st.write("# Welcome to Flank Data Reduction")

    st.sidebar.success("Select what to do next")

    st.markdown("""
        **Start by uploading your single data file.**
        """)
        
    if st.button('Pre-Process Data'):
        prepareDataset()
        subsetsOfDatasets()
        preProcessingData()
        calcRegressionsAndProduceResults()

        st.write('Data successfully pre-processed')

    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
         st.session_state.dfRaw = pd.read_csv(uploaded_file)
         st.write(reportDuplicatesInList(st.session_state.dfRaw.loc[:, 'Comment']))
         st.write(st.session_state.dfRaw)
         
    
    st.subheader('Moessbauer Data')
    st.session_state.dfMoess = pd.read_csv('https://raw.githubusercontent.com/Hezel2000/GeoDataScience/main/data/moessbauer%20standard%20data.csv')
    st.write(st.session_state.dfMoess)

def resultTables():
    import pandas as pd
    
    st.header('Fit Data')
    st.write(st.session_state.dfFitData[:4].round(3))
    
    
    st.header('Fe3+ Std')
    st.write(r'$Fe^{3+}/\Sigma Fe$ deviation from the Moessbauer data should be <0.01-0.015')
    st.write(st.session_state.resultsFe3Std.round(3))
    
    st.header('Fe3+ Smp')
    st.write(r'The error on $Fe^{3+}/\Sigma Fe$ in the Smp is 0.02') 
    st.write(st.session_state.resultsFe3Smp.round(3))

    
    st.header('Drift Data')
    st.write(pd.DataFrame({'Parameter':['A', 'B', 'C', 'D'],
                           'TAP2':st.session_state.fitParametersTAP2,
                           'TAP4':st.session_state.fitParametersTAP4}))
    st.latex(r'''Fe^{2+} = A + B \times \frac{L\beta}{L\alpha} + C \times \Sigma Fe + D \times \Sigma Fe \times \frac{L\beta}{L\alpha}''')
    st.latex(r'''Fe^{3+} = -A - B \times \frac{L\beta}{L\alpha} - C \times \Sigma Fe - D \times \Sigma Fe \times \frac{L\beta}{L\alpha} + Fe_{tot}''')

        
def demo1():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk
    
    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write("""
        This demo shows how to use
        [`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
        to display geospatial data.
        """)

    @st.cache
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


def visualisations():
    import streamlit as st
    import pandas as pd
    from bokeh.plotting import figure, output_file, show
#    from bokeh.models import Panel, Tabs

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
        
        x = st.session_state.dfFitData[r'L$\beta$/L$\alpha$' + crystalName][:4]
        y = st.session_state.dfFitData[r'Fe$_{tot}$'][:4]
        z = st.session_state.dfFitData[r'Fe$^{2+}$'][:4]
    
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
    ##----------------  Drift Plots  ----------------##
    ##-----------------------------------------------##
        
    def driftplots(sel):
        from bokeh.plotting import figure, output_file, show
        from bokeh.models import Span, BoxAnnotation
        import numpy as np
        import pandas as pd
        
        elements = st.session_state.dfMain.columns.tolist()[2:]
        
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
            
            fig = figure(width=500, height=300)
            
            fig.xaxis.axis_label='Point Nr.'
            fig.yaxis.axis_label=el + ' (wt%)'
            
            av_hline = Span(location=av, dimension='width', line_dash='dashed', line_color='brown', line_width=2)
            std_add_hline = Span(location=av+std, dimension='width', line_dash='dashed', line_color='brown', line_width=1)
            std_sub_hline = Span(location=av-std, dimension='width', line_dash='dashed', line_color='brown', line_width=1)
            fig.renderers.extend([av_hline, std_add_hline, std_sub_hline])
            std_box = BoxAnnotation(bottom=av-std, top=av+std, fill_alpha=0.2, fill_color='yellow')
            fig.add_layout(std_box)
            
            fig.line(st.session_state.dfdr.loc[:, 'Point Nr.'], st.session_state.dfdr.loc[:, el])
            fig.circle(st.session_state.dfdr.loc[:, 'Point Nr.'], st.session_state.dfdr.loc[:, el])

    #        ax.add_patch(Rectangle((dfdr.index[0] - 1, np.average(dfdr[el]) -  np.std(dfdr[el])), dfdr.index[-1] + 1,
    #                           2 * np.std(dfdr[el]), color = 'yellow', alpha = .1, zorder = 0))
    #        ax.axhline(y = np.average(dfdr[el]), color = 'brown', linestyle = '--', zorder = 1)
    #        ax.axhline(y = np.average(dfdr[el]) + np.std(dfdr[el]), color = 'brown', linestyle = '-', alpha = .2, zorder = 0)
    #        ax.axhline(y = np.average(dfdr[el]) - np.std(dfdr[el]), color = 'brown', linestyle = '-', alpha = .1, zorder = 0)
    #        ax.set_xlim([dfdr.index[0], dfdr.index[0] + dfdr.index[-1]])
    #        ax.text(.5, .9, 'Drift Monitor', horizontalalignment='center', transform = ax.transAxes)           
    #        ax.text(.5, .72, str(round(np.average(dfdr[el]), 2)) + '±' + str(round(np.std(dfdr[el]), 2))
    #            + ' – rel. std.: ' + str(round(reldev, 2)) + '%',
    #            horizontalalignment='center', transform = ax.transAxes, size = 14, bbox = dict(boxstyle="round",
    #                       ec = 'w', fc = fcColor, alpha = .2))    
            col1.bokeh_chart(fig)
            
            col2.subheader('Statistics')
            resAv = 'average: ' + str(round(av, 2)) + '±' + str(round(std, 2))
            resRelStd ='rel. std.: ' + str(round(reldev, 2)) + '%'
            col2.write(resAv)
            col2.write(resRelStd)
            redDot = 'https://github.com/Hezel2000/microprobe/blob/cc6fa2a89b564786e0b5b68cb6efd11f5ec2ad92/orange_dot.png'
            redCaption = 'worrisome data'
            col2.image(redDot, caption = redCaption)
        
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
        
#-------- Start Parametrisation

    def parametrisationplot():
        from bokeh.plotting import figure, output_file, show
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

#----------------------------------
#----------------------------------
    

    st.header('Visualisations')
    
    st.sidebar.markdown("### Plot")
    plotSel = st.sidebar.selectbox('sel', ('Drift','Parametrisation'))
    
    if plotSel == 'Drift':
        sel = st.selectbox('Select', ('elements', 'Fe3+'))
        driftplots(sel)
    elif plotSel == 'Parametrisation':
        parametrisationplot()
        


#    @st.cache
#    df = st.session_state.dfMain
    
#    sel = st.selectbox('Select', ('elements', 'Fe3+'))
    
#    driftplots(sel)

    

 #   else:
 #       data = df.loc[countries]
 #       data /= 1000000.0
 #       st.write("### Gross Agricultural Production ($B)", data.sort_index())
#
#        data = data.T.reset_index()
#        data = pd.melt(data, id_vars=["index"]).rename(
#            columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
#        )
#        chart = (
#            alt.Chart(data)
#            .mark_area(opacity=0.3)
#            .encode(
#                x="year:T",
#                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
#                color="Region:N",
#            )
#        )
#            st.altair_chart(chart, use_container_width=True)



def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )
        
def test():
    import streamlit as st
    from bokeh.plotting import figure
    
    x = st.session_state.df['SiO2(Mass%)']
    y = st.session_state.df['MgO(Mass%)']
    
#    x = [1, 2, 3, 4, 5]
#    y = [6, 7, 2, 4, 5]

    p = figure(
         title='simple line example',
         x_axis_label='x',
         y_axis_label='y')

    p.scatter(x, y, legend_label='Trend', line_width=2)

    st.bokeh_chart(p, use_container_width=True)




page_names_to_funcs = {
    "Start": start,
    'Result Tables': resultTables,
#    "demo": demo1,
    'Visualisations': visualisations,
    "DataFrame Demo": data_frame_demo,
    'Test': test
}

demo_name = st.sidebar.selectbox("Make a Selection", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

