#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[1]:


import pandas as pd
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 200)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn import linear_model
from ipywidgets import interact
import ipywidgets as widgets
import io
from IPython.display import HTML

from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 44em; }</style>"))

#import modules.flank_method_commands as fmc


# In[2]:


#stdInputList = ['UA5', 'UA10']


# # Toggles

# In[3]:


button_import = widgets.Button(
    description = 'Import & Convert',
    disabled = False,
    button_style = 'warning',
    tooltip = 'click to import & convert',
    icon = 'check'
    )

toggle_import_data = widgets.ToggleButtons(
    options = ['smp data  ', 'info  ', 'moessbauer  '],
    description = 'Display',
    disabled = False,
    button_style = 'info',
    icons = ['search', 'search']
    )

toggle_result_data = widgets.ToggleButtons(
    options = ['fitdata  ', 'Fe3+ Std  ', 'Fe3+ Smp  ', 'drift  '],
    description = 'Display',
    disabled = False,
    button_style = 'info',
#    icons = ['search', 'search']
    )

toggle_visualisations = widgets.ToggleButtons(
    options = ['drift  ', 'Lalpha & Lbeta', 'parametrisation', 'elements  ', 'error considerations', 'smp Fe vs. Fe3+  ', 'std Fe vs. Fe3+  '],
    description = 'Options',
    disabled = False,
    button_style = 'info',
    icons = ['check', 'check', 'check']
    )

toggle_tutorials = widgets.ToggleButtons(
    options = ['Introduction  ', 'Fe 9th order  '],
    description = 'Play',
    disabled = False,
    button_style = 'info',
    icons = ['info', 'info']
    )


# ## Read various datasets and test for duplicate sample/standard names & remove the 'Grid' and 'Line' parts from the comments

# In[4]:


##-----------------------------------------------##
##---------------- Data Import ------------------##
##-----------------------------------------------##

def dataImport():
    global dfMoess
    global dfComplete
    global fileName
    
    if fileUp.value == {}:
        with out:
            print('No CSV loaded')    
    else:
        fileName = list(fileUp.value.keys())[0]
        input_file = list(fileUp.value.values())[0]
        content = input_file['content']
        content = io.StringIO(content.decode('utf-8'))
        dfComplete = pd.read_csv(content)
    
    with out:
        out.clear_output()
        print('sample data imported')
        
#        dfMoess = pd.read_csv('flank data/moessbauer standard data.csv')
        dfMoess = pd.read_csv('https://raw.githubusercontent.com/Hezel2000/GeoDataScience/main/data/moessbauer%20standard%20data.csv')
        print('moessbauer standard dataset imported')

        print(reportDuplicatesInList(dfComplete.loc[:, 'Comment']))

        prepareDataset()
        subsetsOfDatasets()
        preProcessingData()
        calcRegressionsAndProduceResults()
        exportResults()


##-----------------------------------------------##
##--------------- PrepareDataset ----------------##
##-----------------------------------------------##

def prepareDataset():
    global df
    
    for ind in dfComplete.index:
        measurementPointName = dfComplete['Comment'][ind]

        if 'Grid' in measurementPointName:
            dfComplete['Comment'] = dfComplete['Comment'].replace([measurementPointName],measurementPointName.split('Grid')[0])
        elif 'Line' in measurementPointName:
            dfComplete['Comment'] = dfComplete['Comment'].replace([measurementPointName],measurementPointName.split('Line')[0])

    print('renaming successful (-> "Grid" & "Line" parts removed)')

    df = dfComplete.loc[:, ['Point', 'Comment', 'SiO2(Mass%)', 'TiO2(Mass%)', 'Al2O3(Mass%)', 'Cr2O3(Mass%)', 'FeO(Mass%)', 'MnO(Mass%)',
                    'NiO(Mass%)', 'MgO(Mass%)',  'CaO(Mass%)',  'Na2O(Mass%)', 'K2O(Mass%)', 'P2O5(Mass%)', 'Total(Mass%)',
                    'Bi(Net)', 'Ar(Net)', 'Br(Net)', 'As(Net)', 'Current']]

    print('relevant data extracted')

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
    
    df = df.rename(columns = {0:r'L$\beta$/L$\alpha$ (TAP2)', 1:r'L$\beta$/L$\alpha$ (TAP4)'})

    print('column keys successfully reassigned')



##-----------------------------------------------##
##------ Find duplicates in a list and    -------##
##------ produce an according information -------##
##-----------------------------------------------##

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



##-----------------------------------------------##
##-------- Producing 2 subsets into   -----------##
##-------- 1) drift measurements      -----------##
##-------- 2) samples & standards and -----------##
##-----------------------------------------------##
def subsetsOfDatasets():
    global dfdr
    global dfSampleNames

    # a df with only drift measurements
    # drift measurements will be stored in the DataFrame: dfdr
    fil1 = df['Name'].str.startswith('dr')
    drOnly = df[fil1]
    
    selectedStandards = ['AlmO'] #['AlmO', 'UA5', 'UA10', 'Damknolle']

    selStdNames = []
    for i in df['Name']:
        for j in selectedStandards:
            if j in i:
                selStdNames.append(i)

    selStdNames = list(dict.fromkeys(selStdNames))

    fil2 = df['Name'].isin(selStdNames)

    dfdr = pd.concat([drOnly, df[fil2]]).sort_values('Point Nr.')

    # a df with only standards & samples
    dfSampleNames = df[~fil1].loc[:, 'Name'].drop_duplicates()

    print('List with sample names and drift dataset successfully produced')


    
##-----------------------------------------------##
##-- Extract data and calculate the average  ----##
##-- for a sample/standard from its multiple ----##
##-- measurement points                      ----##
##-----------------------------------------------##

def extractAndCalculateAverages(data, l, crystal):
    global d
    
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
#        res.append([i, resFeConcentrations, resLBetaAlphaRatios])
        res.append([d['Point Nr.'].tolist()[0], i, resFeConcentrations, resLBetaAlphaRatios])

    if crystal == 'TAP2':
#        ret = pd.DataFrame(res).rename(columns = {0:'Name', 1:r'Fe$_{tot}$', 2:r'L$\beta$/L$\alpha$ (TAP2)'})
        ret = pd.DataFrame(res).rename(columns = {0:'Point Nr.', 1:'Name', 2:r'Fe$_{tot}$', 3:r'L$\beta$/L$\alpha$ (TAP2)'})        
    else:
#        ret = pd.DataFrame(res).rename(columns = {0:'Name', 1:r'Fe$_{tot}$', 2:r'L$\beta$/L$\alpha$ (TAP4)'})
        ret = pd.DataFrame(res).rename(columns = {0:'Point Nr.', 1:'Name', 2:r'Fe$_{tot}$', 3:r'L$\beta$/L$\alpha$ (TAP4)'})
        
    return ret



##-----------------------------------------------##
##-----------  Pre-processsing data  ------------##
##-----------------------------------------------##
        
# Command for getting Fe2+ and Fetot values from the dfMoss dataset
def extractKnownFe2(stdNameForMatching):
    foundStd = dfMoess[dfMoess['Name'].str.contains(stdNameForMatching)]
    Fe2Value = foundStd['FeO (wt%)'].tolist()[0] * 55.845/(55.845 + 15.9994)
    Fe2ModAbValue = foundStd['Fe2+/SumFe'].tolist()[0]
    return Fe2ModAbValue


def preProcessingData():
        global dfFitData
        global dfMeasStdDataTAP2
        global dfMeasSmpDataTAP2
        global dfMeasStdDataTAP4
        global dfMeasSmpDataTAP4
        global smpList
        global stdList
        
        # Getting the indices of the samples and standards
        samplesListReIndexed = pd.Series(dfSampleNames.tolist())

        fil = samplesListReIndexed.str.contains('AlmO') | samplesListReIndexed.str.contains('UA5') | samplesListReIndexed.str.contains('UA10') | samplesListReIndexed.str.contains('Damknolle')
                
        samples = samplesListReIndexed[~fil].index.values.tolist()
        standards = samplesListReIndexed[fil].index.values.tolist()

        # Getting sample data
        smpList = dfSampleNames.iloc[samples].tolist()

        # First, the indices of the standard measurements must be input from. These are found in the dfSampleNames above
        stdList = dfSampleNames.iloc[standards].tolist()

        # Extracting FeO and Lalpha/Lbeta, the Lbeta/Lalpha ratios are calculated from the measured Lbeta and Lalpha cps, and the data are averaged
        dfMeasStdDataTAP2 = extractAndCalculateAverages(df, stdList, 'TAP2')
        dfMeasSmpDataTAP2 = extractAndCalculateAverages(df, smpList, 'TAP2')
        dfMeasStdDataTAP4 = extractAndCalculateAverages(df, stdList, 'TAP4')
        dfMeasSmpDataTAP4 = extractAndCalculateAverages(df, smpList, 'TAP4')
        
        # Combining measured standard data and required known Fe2+ and Fetot from standard data (-> Moessbauer data)
        combMoessAndMeasStdData = []
        for i in dfMeasStdDataTAP2['Name']:
            res = extractKnownFe2(i.split('_')[0])
            combMoessAndMeasStdData.append(res)


        dfFitData = pd.concat([dfMeasStdDataTAP2, dfMeasStdDataTAP4[r'L$\beta$/L$\alpha$ (TAP4)']], axis = 1)
#        dfFitData = dfFitData.rename(columns = {0 : r'Fe$^{2+}$/$\Sigma$Fe', 1 : r'Fe$^{2+}$'})
        dfFitData = pd.concat([dfFitData, pd.DataFrame(combMoessAndMeasStdData)], axis = 1)
        dfFitData = dfFitData.rename(columns = {0 : r'Fe$^{2+}$/$\Sigma$Fe (Moess)'})

        dfFitData = pd.concat([dfFitData, pd.DataFrame(dfFitData.loc[:, r'Fe$_{tot}$'] * dfFitData.loc[:, r'Fe$^{2+}$/$\Sigma$Fe (Moess)'])], axis = 1)
        dfFitData = dfFitData.rename(columns = {0 : r'Fe$^{2+}$'})
        
        print('Data pre-processing successful')
        

        
##-----------------------------------------------##
##----------  Model linear regression  ----------##
##-----------------------------------------------##
        
def regressionML(inpData, crystal):
    data = inpData
    if crystal == 'TAP2':
        crystalName = ' (TAP2)'
    else:
        crystalName = ' (TAP4)'
    
    X = dfFitData.loc[:, [r'Fe$_{tot}$', r'L$\beta$/L$\alpha$' + crystalName]][:4]   # !!currently only the first 4 stds are used !!
    y = dfFitData.loc[:, r'Fe$^{2+}$'][:4]                      # !!currently only the first 4 stds are used !!

    regr = linear_model.LinearRegression()
    regr.fit(X, y)    # This is required, as it is doing the inital fit

    result = []
    for i in range(len(data['Name'])):
        result.append(regr.predict([[data.iloc[i, 1], data.iloc[i, 2]]])[0])

    return pd.DataFrame(result)    



   
##-----------------------------------------------##
##------  Fit Parameter linear regression  ------##
##-----------------------------------------------##
        
def regressionFitParameters(inpData, crystal):
    global fitParametersTAP2
    global fitParametersTAP4
    
    data = inpData
    if crystal == 'TAP2':
        crystalName = ' (TAP2)'
    else:
        crystalName = ' (TAP4)'
    
    x = dfFitData[r'L$\beta$/L$\alpha$' + crystalName][:4]
    y = dfFitData[r'Fe$_{tot}$'][:4]
    z = dfFitData[r'Fe$^{2+}$'][:4]

    A = [
        [len(x), x.sum(), y.sum(), (x * y).sum()],                         # length(x) sum(x) sum(y) sum(x.*y)
        [x.sum(), (x ** 2).sum(), (x * y).sum(), (y * x ** 2).sum()],          # sum(x) sum(x.^2) sum(x.*y) sum(y.*x.^2)
        [y.sum(), (x * y).sum(), (y ** 2).sum(), (x * y ** 2).sum()],          # sum(y) sum(x.*y) sum(y.^2) sum(x.*y.^2)
        [(x * y).sum(), ((x ** 2) * y).sum(), (x * y ** 2).sum(), ((x ** 2) * (y **2 )).sum()]  # sum(x.*y) sum((x.^2).*y) sum(x.*y.^2) sum((x.^2).*(y.^2))]
        ]

    v = [z.sum(), (z * x).sum(), (z * y).sum(), (x * y * z).sum()]
    
    rfp = np.linalg.inv(A) @ v     # regression parameters

    if crystal == 'TAP2':
        fitParametersTAP2 = rfp
    else:
        fitParametersTAP4 = rfp
    
    res = rfp[0] + rfp[1] * (data[r'L$\beta$/L$\alpha$' + crystalName]) + rfp[2] * data[r'Fe$_{tot}$'] + rfp[3] * (data[r'Fe$_{tot}$'] * data[r'L$\beta$/L$\alpha$' + crystalName])

    resultsFe3FP = (data[r'Fe$_{tot}$'] - res)/data[r'Fe$_{tot}$']

    return resultsFe3FP
    

    
##-----------------------------------------------##
##--  Calculate regressions & produce results  --##
##-----------------------------------------------##

def calcRegressionsAndProduceResults():
    global resultsFe3Std
    global resultsFe3Smp
    
#    resultsFe3StdMLTAP2 = regressionML(dfMeasStdDataTAP2, 'TAP2')
#    resultsFe3SmpMLTAP2 = regressionML(dfMeasSmpDataTAP2, 'TAP2')
#    resultsFe3StdMLTAP4 = regressionML(dfMeasStdDataTAP4, 'TAP4')
#    resultsFe3SmpMLTAP4 = regressionML(dfMeasSmpDataTAP4, 'TAP4')
#    print('model linear regression results successfully produced')

    resultsFe3StdFPTAP2 = pd.DataFrame(regressionFitParameters(dfMeasStdDataTAP2, 'TAP2'))
    resultsFe3SmpFPTAP2 = pd.DataFrame(regressionFitParameters(dfMeasSmpDataTAP2, 'TAP2'))
    resultsFe3StdFPTAP4 = pd.DataFrame(regressionFitParameters(dfMeasStdDataTAP4, 'TAP4'))
    resultsFe3SmpFPTAP4 = pd.DataFrame(regressionFitParameters(dfMeasSmpDataTAP4, 'TAP4'))
    print('model linear regression results successfully produced')

    fe3StdMoessList = []
    for i in dfMeasStdDataTAP2['Name']:
        fil = list(map(lambda x : True if x in i else False, dfMoess['Name']))
        fe3StdMoessList.append(dfMoess[fil]['Fe3+/SumFe'].values[0])
    fe3StdMoessList = pd.DataFrame(fe3StdMoessList)

    resultsFe3Std = pd.concat([dfMeasStdDataTAP2['Point Nr.'], dfMeasStdDataTAP2['Name']
                    , dfMeasStdDataTAP2[r'Fe$_{tot}$'], fe3StdMoessList[0]
#                    ,(dfMeasStdDataTAP2[r'Fe$_{tot}$'] - resultsFe3StdMLTAP2[0]) / resultsFe3StdMLTAP2[0]
#                    ,(dfMeasStdDataTAP2[r'Fe$_{tot}$'] - resultsFe3StdMLTAP2[0]) / resultsFe3StdMLTAP2[0] - fe3StdMoessList[0]
#                    ,(dfMeasStdDataTAP4[r'Fe$_{tot}$'] - resultsFe3StdMLTAP4[0]) / resultsFe3StdMLTAP4[0]
#                    ,(dfMeasStdDataTAP4[r'Fe$_{tot}$'] - resultsFe3StdMLTAP4[0]) / resultsFe3StdMLTAP4[0] - fe3StdMoessList[0]
                    ,resultsFe3StdFPTAP2[0], resultsFe3StdFPTAP2[0] - fe3StdMoessList[0]
                    ,resultsFe3StdFPTAP4[0], resultsFe3StdFPTAP4[0] - fe3StdMoessList[0]], axis = 1,
                    #resultsFe3StdML[0], dfMeasStdDataTAP2['Fetot'] - resultsFe3StdML[0],
                     keys = ['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)', 'Moessbauer'
#                             ,r'Fe$^{3+}$/$\Sigma$Fe (ML, TAP2)', r'$\Delta$ Meas - Moess'
#                             ,r'Fe$^{3+}$/$\Sigma$Fe (ML, TAP4)', r'$\Delta$ Meas - Moess'
                             ,r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)', r'$\Delta$ Meas - Moess'
                             ,r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)', r'$\Delta$ Meas - Moess'
                            ])
    resultsFe3Smp = pd.concat([dfMeasSmpDataTAP2['Point Nr.'], dfMeasSmpDataTAP2['Name']
                    , dfMeasSmpDataTAP2[r'Fe$_{tot}$']
#                    ,(dfMeasSmpDataTAP2[r'Fe$_{tot}$'] - resultsFe3SmpMLTAP2[0]) / resultsFe3SmpMLTAP2[0]
#                    ,(dfMeasSmpDataTAP4[r'Fe$_{tot}$'] - resultsFe3SmpMLTAP4[0]) / resultsFe3SmpMLTAP4[0]
                    ,resultsFe3SmpFPTAP2[0]
                    ,resultsFe3SmpFPTAP4[0]
                    ,resultsFe3SmpFPTAP2[0]-resultsFe3SmpFPTAP4[0]], axis = 1,
                    #resultsFe3StdML[0], dfMeasSmpDataTAP2['Fetot'] - resultsFe3StdML[0],
                     keys = ['Point Nr.', 'Name', r'$\Sigma$Fe (wt%)'
#                             ,r'Fe$^{3+}$/$\Sigma$Fe (ML, TAP2)'
#                             ,r'Fe$^{3+}$/$\Sigma$Fe (ML, TAP4)'
                             ,r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)', r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)', 'TAP2-TAP4'])
#    resultsFe3Smp = resultsFe3Smp.style.set_properties(**{'background-color': 'green'}, subset=[r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)'])

    

##-----------------------------------------------##
##--------------  Export Results  ---------------##
##-----------------------------------------------##

def exportResults():
    # Export results into an Excel-file
    resultsFe3Smp.to_excel('Flank Results Samples.xlsx')
    resultsFe3Std.to_excel('Flank Results Standards.xlsx')
    
    print('results successfully exported to an excel-file')



##-----------------------------------------------##
##----------------  Line Plots  -----------------##
##-----------------------------------------------##

def elementPlots(sel, smp, el, noc):
    if sel == 'Individual':
        lineplot(smp, el)
    elif sel == 'All Samples':
        allelements(el, noc)
    elif sel == 'Single Sample':
        singlesample(smp)

def lineplot(smp, el):
        fil = df['Name'].str.contains(smp)
        xdata = df[fil].loc[:, 'Point Nr.']
        data = df[fil].loc[:, el]
        reldev = 100 * np.std(data)/np.average(data)
        if reldev < 1:
            fcColor = (.5, 0.8, 0)
        elif 1 <= reldev < 5:
            fcColor = 'orange'
        else:
            fcColor = 'r'
            
        startX = data.index[0]

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7, 4))
        ax.plot(xdata, data, marker = 'o', markersize = 5)
        ax.add_patch(Rectangle((startX, np.average(data) -  np.std(data)), startX+len(data) + 1,
                                2 * np.std(data), color = 'yellow', alpha = .1, zorder = 0))
        ax.hlines(y=np.average(data), xmin=startX, xmax=startX+len(data), colors='brown', linestyles='--', lw=1)
        ax.hlines(y=np.average(data) + np.std(data), xmin=startX, xmax=startX+len(data), colors='brown', alpha = .2, linestyles='--', lw=1)
        ax.hlines(y=np.average(data) - np.std(data), xmin=startX, xmax=startX+len(data), colors='brown', alpha = .2, linestyles='--', lw=1)
        ax.text(.5, .9, el + ': ' + str(round(np.average(data), 2)) + '±'
                + str(round(np.std(data), 2)) + ' – rel. std.: '
                + str(round(reldev, 2)) + '%',
                horizontalalignment='center', transform = ax.transAxes, size = 14,
                bbox = dict(boxstyle="round", ec = 'w', fc = fcColor, alpha = .2))
        ax.set_xlabel('Point Nr.')
        ax.set_ylabel(el)
        ax.set_xlim(data.index[0], data.index[0] + len(data) + 1)
        plt.show()
    
def allelements(el, nrOfColumns):
        print('This might take a second or two.')
        nrOfSmp = len(dfSampleNames)
        intPart, decPart = divmod(len(dfSampleNames)/4, 1)
        nrOfPlots = int(decPart * nrOfColumns)
        nrOfRows = int(intPart) + 1

        fig, axes = plt.subplots(nrows = nrOfRows, ncols = 4, figsize = (20, 2.5 * nrOfRows))

        i = 0
        for row in axes:
            for ax in row:
                if i < nrOfSmp:
                    fil = df['Name'].str.contains(dfSampleNames.iloc[i])
                    xdata = df[fil].loc[:, 'Point Nr.']
                    data = df[fil].loc[:, el]
                    reldev = 100 * np.std(data)/np.average(data)
                    if reldev < 1:
                        fcColor = (.5, 0.8, 0)
                    elif 1 <= reldev < 5:
                        fcColor = 'orange'
                    else:
                        fcColor = 'r'
                    ax.plot(xdata, data, marker = 'o', markersize = 5)
                    ax.set_xlabel('Point Nr.')
                    ax.add_patch(Rectangle((data.index[0] - 1, np.average(data) -  np.std(data)), len(data) + 1,
                                           2 * np.std(data), color = 'yellow', alpha = .1, zorder = 0))
                    ax.axhline(y = np.average(data), color = 'brown', linestyle = '--', zorder = 1)
                    ax.axhline(y = np.average(data) + np.std(data), color = 'brown', linestyle = '-', alpha = .2, zorder = 0)
                    ax.axhline(y = np.average(data) - np.std(data), color = 'brown', linestyle = '-', alpha = .2, zorder = 0)
                    ax.set_xlim([data.index[0], data.index[0] + len(data) + 1])
                    ax.text(.5,.9,dfSampleNames.iloc[i], horizontalalignment='center', transform = ax.transAxes)           
                    ax.text(.5, .72, str(round(np.average(data), 2)) + '±' + str(round(np.std(data), 2))
                            + ' – rel. std.: ' + str(round(reldev, 2)) + '%',
                            horizontalalignment='center', transform = ax.transAxes, size = 14, bbox = dict(boxstyle="round",
                                       ec = 'w', fc = fcColor, alpha = .2))
                else:
                    fig.delaxes(ax)
                i += 1
        plt.show()

        
def singlesample(smp):
    elements = df.columns.tolist()[2:]
    
    fig, axes = plt.subplots(nrows = 6, ncols = 3, figsize = (20, 20))

    i = 0
    for row in axes:
        for ax in row:
            fil = df['Name'].str.contains(smp)
            xdata = df[fil].loc[:, 'Point Nr.']
            data = df[fil].loc[:, elements[i]]
            reldev = 100 * np.std(data)/np.average(data)
            if reldev < 1:
                fcColor = (.5, 0.8, 0)
            elif 1 <= reldev < 5:
                fcColor = 'orange'
            else:
                fcColor = 'r'
            ax.plot(xdata, data, marker = 'o', markersize = 5)
            ax.add_patch(Rectangle((data.index[0] - 1, np.average(data) -  np.std(data)), len(data) + 1,
                                   2 * np.std(data), color = 'yellow', alpha = .1, zorder = 0))
            ax.axhline(y = np.average(data), color = 'brown', linestyle = '--', zorder = 1)
            ax.axhline(y = np.average(data) + np.std(data), color = 'brown', linestyle = '-', alpha = .2, zorder = 0)
            ax.axhline(y = np.average(data) - np.std(data), color = 'brown', linestyle = '-', alpha = .1, zorder = 0)
    #        ax.set_xlabel('Nr. of Analysis')
            ax.set_xlim(data.index[0], data.index[0] + len(data) + 1)
            ax.set_xlabel('Point Nr.')
            ax.set_ylabel(elements[i])
    #        ax.set_title(dfSampleNames.iloc[i])
            ax.text(.5, .9, elements[i] + ': ' + str(round(np.average(data), 2)) + '±'
                    + str(round(np.std(data), 2)) + ' – rel. std.: '
                    + str(round(reldev, 2)) + '%',
                    horizontalalignment='center', transform = ax.transAxes, size = 14,
                    bbox = dict(boxstyle="round", ec = 'w', fc = fcColor, alpha = .2))
            i += 1

    plt.show()
    


##-----------------------------------------------##
##----------------  Drift Plots  ----------------##
##-----------------------------------------------##
    
def driftplots(sel, el):
    if sel == 'elements':
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 3))
        ax.plot(dfdr.loc[:, el], marker = 'o', markersize = 5)
        reldev = 100 * np.std(dfdr[el])/np.average(dfdr[el])
    #                elif sel == 'elements  ':
        if reldev < 1:
            fcColor = (.5, 0.8, 0)
        elif 1 <= reldev < 5:
            fcColor = 'orange'
        else:
            fcColor = 'r'
        ax.set_xlabel('Point Nr.')
        ax.set_ylabel(el + ' (wt%)')
        ax.add_patch(Rectangle((dfdr.index[0] - 1, np.average(dfdr[el]) -  np.std(dfdr[el])), dfdr.index[-1] + 1,
                           2 * np.std(dfdr[el]), color = 'yellow', alpha = .1, zorder = 0))
        ax.axhline(y = np.average(dfdr[el]), color = 'brown', linestyle = '--', zorder = 1)
        ax.axhline(y = np.average(dfdr[el]) + np.std(dfdr[el]), color = 'brown', linestyle = '-', alpha = .2, zorder = 0)
        ax.axhline(y = np.average(dfdr[el]) - np.std(dfdr[el]), color = 'brown', linestyle = '-', alpha = .1, zorder = 0)
        ax.set_xlim([dfdr.index[0], dfdr.index[0] + dfdr.index[-1]])
        ax.text(.5, .9, 'Drift Monitor', horizontalalignment='center', transform = ax.transAxes)           
        ax.text(.5, .72, str(round(np.average(dfdr[el]), 2)) + '±' + str(round(np.std(dfdr[el]), 2))
            + ' – rel. std.: ' + str(round(reldev, 2)) + '%',
            horizontalalignment='center', transform = ax.transAxes, size = 14, bbox = dict(boxstyle="round",
                       ec = 'w', fc = fcColor, alpha = .2))    
        plt.show()
    
    else:
        dfdrLRatioTAP2 = dfdr[r'L$\beta$ (TAP2)']/dfdr[r'L$\alpha$ (TAP2)']
        dfdrLRatioTAP4 = dfdr[r'L$\beta$ (TAP4)']/dfdr[r'L$\alpha$ (TAP4)']
        dfdrCalList = pd.concat([dfdr['FeO'], dfdrLRatioTAP2], axis = 1)
        dfdrCalList = dfdrCalList.rename(columns = {'FeO': r'Fe$_{tot}$', 0:r'L$\beta$/L$\alpha$ (TAP2)'})
        dfdrCalList4 = pd.concat([dfdr['FeO'], dfdrLRatioTAP4], axis = 1)
        dfdrCalList4 = dfdrCalList4.rename(columns = {'FeO': r'Fe$_{tot}$', 0:r'L$\beta$/L$\alpha$ (TAP4)'})

        plt.scatter(regressionFitParameters(dfdrCalList, 'TAP2'),
                    regressionFitParameters(dfdrCalList4, 'TAP4'))
        plt.xlabel(r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)')
        plt.ylabel(r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)')
        plt.show()
    


##-----------------------------------------------##
##---------------  L-line Plots  ----------------##
##-----------------------------------------------##

def lLinePlots(sel):
    if sel == 'all':
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 6))

        tapl2Fe3 = resultsFe3Smp[r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)']
        tapl4Fe3 = resultsFe3Smp[r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)']
        ax1.scatter(tapl2Fe3, tapl4Fe3)

        x = np.linspace(0,1,10)
        ax1.plot(x, x, 'b-')
        ax1.plot(x, x + .02, 'g--', x, x - .02, 'g--')
        ax1.set_xlim(0, 1.1 * max(tapl2Fe3))
        ax1.set_ylim(0, 1.1 * max(tapl2Fe3))

        ax1.set_xlabel(r'Fe$^{3+}$/$\Sigma$Fe (FP, TAPL2)')
        ax1.set_ylabel(r'Fe$^{3+}$/$\Sigma$Fe (FP, TAPL4)')


        tapl2Betacps = df[r'L$\beta$ (TAP2)']
        tapl2Alphacps = df[r'L$\alpha$ (TAP2)']
        tapl4Betacps = df[r'L$\beta$ (TAP4)']
        tapl4Alphacps = df[r'L$\alpha$ (TAP4)']

        ax2.scatter(tapl2Betacps, tapl2Alphacps, label = 'TAPL2')
        ax2.scatter(tapl4Betacps, tapl4Alphacps, label = 'TAPL4')
        ax2.set_xlabel(r'L$\beta$ (cps)')
        ax2.set_ylabel(r'L$\alpha$ (cps)')
        ax2.legend()

        ax3.plot((tapl2Betacps/tapl2Alphacps)/(tapl4Betacps/tapl4Alphacps))
        ax3.set_ylabel(r'L$\beta$/L$\alpha$ (TAPL2)/L$\beta$/L$\alpha$ (TAPL4)')
        ax3.set_ylim(.9, 1.3)

        ax4.plot(tapl2Betacps/tapl2Alphacps, label = 'TAPL2')
        ax4.plot(tapl4Betacps/tapl4Alphacps, label = 'TAPL4')
        ax4.set_ylabel(r'L$\beta$/L$\alpha$ (cps-ratio)')
        ax4.legend()

        plt.show()

    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (20, 6))

        tapl2Betacps = df[r'L$\beta$ (TAP2)']
        tapl2Alphacps = df[r'L$\alpha$ (TAP2)']
        tapl4Betacps = df[r'L$\beta$ (TAP4)']
        tapl4Alphacps = df[r'L$\alpha$ (TAP4)']

        ax1.plot((tapl2Betacps/tapl2Alphacps)/(tapl4Betacps/tapl4Alphacps))
        ax1.set_ylabel(r'L$\beta$/L$\alpha$ (TAPL2)/L$\beta$/L$\alpha$ (TAPL4)')
        ax1.set_ylim(.9, 1.3)

        ax2.plot(tapl2Betacps/tapl2Alphacps, label = 'TAPL2')
        ax2.plot(tapl4Betacps/tapl4Alphacps, label = 'TAPL4')
        ax2.set_ylabel(r'L$\beta$/L$\alpha$ (cps-ratio)')
        ax2.legend()

        plt.show()


# # Error Considerations

# In[5]:



def errorVisualisations(sel):
    if sel == 'help':
        print('''
        result variations:
        Individual samples are plotted along the x-axis.
        For each sample, the Fetot (top 2 plots) and
        Lbeta/Lalpha (bottom 2 plots) are changed by the percentage
        given in the legend. The Fe3+/SumFe is then calculated with
        the new Fetot or Lbeta/Lalpha. The result is then subtracted
        from the true Fe3+/SumFe and plotted on the y-axis.
        
        sample s.d.:
        Individual samples/drift monitors are plotted along the x-axis.
        The 1 s.d. of Lbeta/Lalpha of a single sample is calculated and
        plotted on the y-axis.
        ''')
    elif sel == 'result variations':
        errorPercentDeviations()
    else:
        errorSmpFe3Dev()
        
    
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


##-----------------------------------------------##
##--------------  parametrisation  --------------##
##-----------------------------------------------##

def parametrisation():
    Fetot = np.linspace(0, 60, 100)
    ATAP2, BTAP2, CTAP2, DTAP2 = fitParametersTAP2
    ATAP4, BTAP4, CTAP4, DTAP4 = fitParametersTAP4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

    for i in range(10):
        Fe3 = .1 * i
        ax1.plot(Fetot, (-ATAP2 - CTAP2 * Fetot + Fetot - Fetot * Fe3) / (BTAP2 + DTAP2 * Fetot), 'b--', alpha=.2)
        ax1.plot(Fetot, (-ATAP4 - CTAP4 * Fetot + Fetot - Fetot * Fe3) / (BTAP4 + DTAP4 * Fetot), 'g--', alpha=.2)
        ax2.plot(Fetot, (-ATAP2 - CTAP2 * Fetot + Fetot - Fetot * Fe3) / (BTAP2 + DTAP2 * Fetot), 'b--', alpha=.2)
        ax2.plot(Fetot, (-ATAP4 - CTAP4 * Fetot + Fetot - Fetot * Fe3) / (BTAP4 + DTAP4 * Fetot), 'g--', alpha=.2)

    ax1.plot(0, 'b--', alpha=.6, label = 'TAP2')
    ax1.plot(0, 'g--', alpha=.6, label = 'TAP4')
    ax1.scatter(dfMeasSmpDataTAP2[r'Fe$_{tot}$'],dfMeasSmpDataTAP2[r'L$\beta$/L$\alpha$ (TAP2)']
                , 20, color = 'lightblue', linewidth = .5, edgecolors = 'black', label = 'smp (TAP2)')
    ax1.scatter(dfMeasSmpDataTAP4[r'Fe$_{tot}$'],dfMeasSmpDataTAP4[r'L$\beta$/L$\alpha$ (TAP4)']
                , 20, color = 'lightgreen', linewidth = .5, edgecolors = 'black', label = 'smp (TAP4)')
    ax1.scatter(dfFitData[r'Fe$_{tot}$'],dfFitData[r'L$\beta$/L$\alpha$ (TAP2)']
                , 100, color = 'blue', linewidth = .5, edgecolors = 'white', label = 'std (TAP2)')
    ax1.scatter(dfFitData[r'Fe$_{tot}$'],dfFitData[r'L$\beta$/L$\alpha$ (TAP4)']
                , 100, color = 'green', linewidth = .5, edgecolors = 'white', label = 'std (TAP4)')
    ax1.set_xlabel(r'$\Sigma$Fe (wt%)')
    ax1.set_ylabel(r'L$\beta$/L$\alpha$ (net cps-ratio)')
    ax1.set_xlim(0, 60)
    ax1.set_ylim(.2, 2)
    ax1.legend()

    ax2.plot(0, 'b--', alpha=.6, label = 'TAP2')
    ax2.plot(0, 'g--', alpha=.6, label = 'TAP4')
    ax2.scatter(dfMeasSmpDataTAP2[r'Fe$_{tot}$'],dfMeasSmpDataTAP2[r'L$\beta$/L$\alpha$ (TAP2)']
                , 20, color = 'lightblue', linewidth = .5, edgecolors = 'black', label = 'smp (TAP2)')
    ax2.scatter(dfMeasSmpDataTAP4[r'Fe$_{tot}$'],dfMeasSmpDataTAP4[r'L$\beta$/L$\alpha$ (TAP4)']
                , 20, color = 'lightgreen', linewidth = .5, edgecolors = 'black', label = 'smp (TAP4)')
    ax2.scatter(dfFitData[r'Fe$_{tot}$'],dfFitData[r'L$\beta$/L$\alpha$ (TAP2)']
                , 100, color = 'blue', linewidth = .5, edgecolors = 'white', label = 'std (TAP2)')
    ax2.scatter(dfFitData[r'Fe$_{tot}$'],dfFitData[r'L$\beta$/L$\alpha$ (TAP4)']
                , 100, color = 'green', linewidth = .5, edgecolors = 'white', label = 'std (TAP4)')
    ax2.set_xlabel(r'$\Sigma$Fe (wt%)')
    ax2.set_ylabel(r'L$\beta$/L$\alpha$ (net cps-ratio)')
    ax2.set_xlim(0, 12)
    ax2.set_ylim(.5, 1)
    ax2.legend()

    plt.show()


# # Interactive Framework

# In[6]:


tab = widgets.Tab()
out = widgets.Output(layout = {'border':'1px solid blue'})
fileUp = widgets.FileUpload(accept = '', multiple = False)
children = [widgets.VBox([widgets.HBox([fileUp, button_import])
                          ,toggle_import_data
                          , out]),
            widgets.VBox([toggle_result_data,
                         out]),
            widgets.VBox([toggle_visualisations,
                         out]),
            widgets.VBox([toggle_tutorials,
                         out])
           ]
tab.children = children
tab.set_title(0, 'Upload & Inspection')
tab.set_title(1, 'Result Data')
tab.set_title(2, 'Result Visualisations')
tab.set_title(3, 'Tutorials')

def import_clicked(b):
#    data_import()
    dataImport()

def import_data(b):
    sel = toggle_import_data.value
    with out:
        out.clear_output()
        if sel == 'smp data  ':
            print('This might take a sceond or two ...')
            display(df)
        if sel == 'info  ':
            print('You imported the file: ' + fileName)
            print('')
            print('The following features will be implemented soon:')
            print(
                ' Tooltip für Drift-Korrektur Punkte \n TAP -> TAPL'
                )
        elif sel == 'moessbauer  ':
            display(round(dfMoess, 3))

def result_data(b):
    sel = toggle_result_data.value
    with out:
        out.clear_output()
        if sel == 'fitdata  ':
            display(dfFitData[:4].round(3))
            display(pd.DataFrame({'Parameter':['A', 'B', 'C', 'D'], 'TAP2':fitParametersTAP2, 'TAP4':fitParametersTAP4}))
            a = r'Fe^{2+} = A + B \times \frac{L\beta}{L\alpha} + C \times \Sigma Fe + D \times \Sigma Fe \times \frac{L\beta}{L\alpha}'
            b = r'Fe^{3+} = -A - B \times \frac{L\beta}{L\alpha} - C \times \Sigma Fe - D \times \Sigma Fe \times \frac{L\beta}{L\alpha} + Fe_{tot}'
            ax = plt.axes([0,0,0.3,0.3]) #left,bottom,width,height
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            plt.text(0,.5,'$%s$' %a, size=20)
            plt.text(0,0,'$%s$' %b, size=20)

            plt.show()
        elif sel == 'Fe3+ Std  ':
            print(r'$Fe^{3+}/\Sigma Fe$ deviation from the Moessbauer data should be <0.01-0.015')
            display(resultsFe3Std.round(3))
        elif sel == 'Fe3+ Smp  ':
            print(r'The error on $Fe^{3+}/\Sigma Fe$ in the Smp is 0.02') 
            display(resultsFe3Smp.round(3))
        elif sel == 'drift  ':
            display(dfdr.round(3))


def result_visualisation(b):
    elements = df.columns.tolist()[2:]  # required for the 'elements' toggle button
    sel = toggle_visualisations.value
    if sel != {}:
        with out:
            out.clear_output()
            if sel == 'drift  ':
                interact(driftplots, sel = ['elements', 'Fe3+'], el = elements)
            elif sel == 'Lalpha & Lbeta':
                interact(lLinePlots, sel = ['all', 'lines'])
            elif sel == 'parametrisation':
                print('Fit Parameters (FP) for TAP2 are: ' + str(list(np.round(fitParametersTAP2, 2))))
                print('Fit Parameters (FP) for TAP4 are: ' + str(list(np.round(fitParametersTAP4, 2))))
                display(parametrisation())
            elif sel == 'elements  ':
                interact(elementPlots, sel = ['Individual', 'All Samples', 'Single Sample'], smp = dfSampleNames, el = elements, noc = range(1, 5, 1))
            elif sel == 'error considerations':
                interact(errorVisualisations, sel = ['help', 'result variations', 'sample s.d.'])
            elif sel == 'smp Fe vs. Fe3+  ':
#                plt.scatter(resultsFe3Smp.loc[:, r'$\Sigma$Fe (wt%)'], resultsFe3Smp.loc[:, r'Fe$^{3+}$/$\Sigma$Fe (ML, TAP2)'], label = 'ML, TAP2')
#                plt.scatter(resultsFe3Smp.loc[:, r'$\Sigma$Fe (wt%)'], resultsFe3Smp.loc[:, r'Fe$^{3+}$/$\Sigma$Fe (ML, TAP4)'], label = 'ML, TAP4')
                plt.scatter(resultsFe3Smp.loc[:, r'$\Sigma$Fe (wt%)'], resultsFe3Smp.loc[:, r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)'], label = 'FP, TAP2')
                plt.scatter(resultsFe3Smp.loc[:, r'$\Sigma$Fe (wt%)'], resultsFe3Smp.loc[:, r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)'], label = 'FP, TAP4')
                plt.xlabel(r'$\Sigma$Fe (wt%)')
                plt.ylabel(r'Fe$^{3+}$/$\Sigma$Fe')
                plt.legend()
                plt.show()
            elif sel == 'std Fe vs. Fe3+  ':
#                plt.scatter(resultsFe3Std.loc[:, r'$\Sigma$Fe (wt%)'], resultsFe3Std.loc[:, r'Fe$^{3+}$/$\Sigma$Fe (ML, TAP2)'], label = 'ML, TAP2')
#                plt.scatter(resultsFe3Std.loc[:, r'$\Sigma$Fe (wt%)'], resultsFe3Std.loc[:, r'Fe$^{3+}$/$\Sigma$Fe (ML, TAP4)'], label = 'ML, TAP4')
                plt.scatter(resultsFe3Std.loc[:, r'$\Sigma$Fe (wt%)'], resultsFe3Std.loc[:, r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)'], label = 'FP, TAP2')
                plt.scatter(resultsFe3Std.loc[:, r'$\Sigma$Fe (wt%)'], resultsFe3Std.loc[:, r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)'], label = 'FP, TAP4')
                plt.xlabel(r'$\Sigma$Fe (wt%)')
                plt.ylabel(r'Fe$^{3+}$/$\Sigma$Fe')
                plt.legend()
                plt.show()

                            
def tutorials(b):
    sel = toggle_tutorials.value
    if sel != {}:
        with out:
            out.clear_output()
            if sel == 'Introduction  ':
                print('Intro to the Flank Method data reduction')
                display(HTML('<iframe src="https://player.vimeo.com/video/700522290?h=2d4511dd62" width="640" height="404" frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe>'))
            elif sel == 'Fe 9th order  ':
                print('kommt auch bald')

button_import.on_click(import_clicked)
toggle_result_data.observe(result_data)
toggle_import_data.observe(import_data, 'value')
toggle_visualisations.observe(result_visualisation, 'value')
toggle_tutorials.observe(tutorials, 'value')


# In[7]:


tab


# In[8]:


import plotly.graph_objects as go


# In[42]:


A, B, C, D = [-9.82, 18.14, 0.1, 0.09]

LRatio, Fetot = np.linspace(0.2, 2, 100), np.linspace(0, 60, 100)
xGrid, yGrid = np.meshgrid(Fetot, LRatio)
z = A + B * xGrid + C * yGrid + D * xGrid * yGrid

fig = go.Figure(go.Surface(x=xGrid, y=yGrid, z=z, showscale = False))


fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[0,2],),
        yaxis = dict(nticks=4, range=[0,60],),
        zaxis = dict(nticks=4, range=[0, .2],),),
        width=700, height=500
#    margin=dict(r=20, l=10, b=10, t=10)
)


fig.show()


# In[38]:


yGrid


# In[37]:


x = np.linspace(-5, 80, 10)
y = np.linspace(-5, 60, 10)
xGrid, yGrid = np.meshgrid(y, x)
xGrid

