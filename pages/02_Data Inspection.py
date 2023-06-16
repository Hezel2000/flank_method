import streamlit as st


def datainspection():
    import streamlit as st
    import pandas as pd
    # from bokeh.plotting import figure, output_file, show
#    from bokeh.models import Panel, Tabs


# --------  Start Linear Regression with Fit Parameters

    def regressionFitParameters(inpData, crystal):
        import streamlit as st
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

        if crystal == 'TAP2':
            st.session_state.fitParametersTAP2 = rfp
        else:
            st.session_state.fitParametersTAP4 = rfp

        res = rfp[0] + rfp[1] * (data[r'L$\beta$/L$\alpha$' + crystalName]) + rfp[2] * \
            data[r'Fe$_{tot}$'] + rfp[3] * \
            (data[r'Fe$_{tot}$'] * data[r'L$\beta$/L$\alpha$' + crystalName])

        resultsFe3FP = (data[r'Fe$_{tot}$'] - res)/data[r'Fe$_{tot}$']

        return resultsFe3FP

# --------  End Linear Regression with Fit Parameters

# -------- Start Fe3+ Results Plot

    def resultsplotPlot():
        import streamlit as st
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, Text
        import numpy as np

        Fetot = np.linspace(0, 60, 100)
        ATAP2, BTAP2, CTAP2, DTAP2 = st.session_state.fitParametersTAP2
        ATAP4, BTAP4, CTAP4, DTAP4 = st.session_state.fitParametersTAP4

        figParam = figure(plot_width=600, plot_height=400)

        for i in range(11):
            Fe3 = .1 * i
            figParam.line(Fetot, (-ATAP2 - CTAP2 * Fetot + Fetot - Fetot * Fe3) /
                          (BTAP2 + DTAP2 * Fetot), line_color='blue', line_alpha=.3)
            figParam.line(Fetot, (-ATAP4 - CTAP4 * Fetot + Fetot - Fetot * Fe3) /
                          (BTAP4 + DTAP4 * Fetot), line_color='orange', line_alpha=.3)

        text_x = []
        text_y = []
        text_plot = []
        Fetot_plot = 47
        for i in range(11):
            text_x.append(Fetot_plot)
            text_y.append((-ATAP2 - CTAP2 * Fetot_plot + Fetot_plot -
                          Fetot_plot * i/10) / (BTAP2 + DTAP2 * Fetot_plot))
            text_plot.append(str(i/10))
        source = ColumnDataSource(dict(x=text_x, y=text_y, text=text_plot))
        glyph = Text(x='x', y='y', text='text',
                     text_color='blue', text_alpha=.3, text_font_size={'value': '25'})
        figParam.add_glyph(source, glyph)

        figParam.circle(st.session_state.dfMeasSmpDataTAP2[r'Fe$_{tot}$'], st.session_state.dfMeasSmpDataTAP2[r'L$\beta$/L$\alpha$ (TAP2)'],
                        size=5, legend_label='TAP2')
        figParam.circle(st.session_state.dfMeasSmpDataTAP4[r'Fe$_{tot}$'], st.session_state.dfMeasSmpDataTAP4[r'L$\beta$/L$\alpha$ (TAP4)'],
                        size=5, fill_color='orange', line_color='orange', legend_label='TAP4')
        figParam.scatter(st.session_state.dfFitData[r'Fe$_{tot}$'], st.session_state.dfFitData[r'L$\beta$/L$\alpha$ (TAP2)'],
                         size=8, line_color='black')
        figParam.scatter(st.session_state.dfFitData[r'Fe$_{tot}$'], st.session_state.dfFitData[r'L$\beta$/L$\alpha$ (TAP4)'],
                         size=8, fill_color='orange', line_color='black')

        figParam.xaxis.axis_label = 'Sum Fe (wt%)'
        figParam.yaxis.axis_label = r'Lb/La (net cps-ratio)'
        figParam.axis.minor_tick_in = -3
        figParam.axis.minor_tick_out = 6

        st.bokeh_chart(figParam)

        st.subheader('3D Presentation')
        st.write('Sample & Standard data are displayed')

        import plotly.express as px
        el = st.session_state.output_file.columns

        col1, col2 = st.columns([1, 2])
        with col1:
            xaxis3d = st.selectbox('x-axis', el, index=28)
            yaxis3d = st.selectbox('y-axis', el, index=3)
            zaxis3d = st.selectbox('z-axis', el, index=4)
            color3d = st.selectbox('point colour', el, index=8)
        with col2:
            fig = px.scatter_3d(st.session_state.output_file,
                                x=xaxis3d, y=yaxis3d, z=zaxis3d, color=color3d)
            st.plotly_chart(fig)

# -------- End Fe3+ Results Plot

# --------  Start Drift Inspection

    def driftplots(sel):
        import streamlit as st
        from bokeh.plotting import figure, output_file, ColumnDataSource
        from bokeh.models import Span, BoxAnnotation
        import numpy as np

        elements = list(reversed(st.session_state.dfMain.columns.tolist()[3:]))
        if sel == 'Composition of drift standards':
            el = st.selectbox('Select', elements)

            av = np.average(st.session_state.dfdr[el])
            std = np.std(st.session_state.dfdr[el])

            reldev = 100 * \
                np.std(st.session_state.dfdr[el]) / \
                np.average(st.session_state.dfdr[el])

            col1, col2 = st.columns([3, 1])
            col1.subheader('Drift Monitor')

            TOOLTIPS = [('Name', '@Name'),
                        ('Point Nr.', '@{Point Nr.}'),
                        (el, '@'+el)]

            fig = figure(width=500, height=300, tooltips=TOOLTIPS)

            fig.line(
                st.session_state.dfdr.loc[:, 'Point Nr.'], st.session_state.dfdr.loc[:, el])
            output_file("toolbar.html")
            source = ColumnDataSource(st.session_state.dfdr.to_dict('list'))

            fig.circle('Point Nr.', el, size=4, source=source)

            fig.xaxis.axis_label = 'Point Nr.'
            fig.yaxis.axis_label = el + ' (wt%)'

            av_hline = Span(location=av, dimension='width',
                            line_dash='dashed', line_color='brown', line_width=2)
            std_add_hline = Span(location=av+std, dimension='width',
                                 line_dash='dashed', line_color='brown', line_width=1)
            std_sub_hline = Span(location=av-std, dimension='width',
                                 line_dash='dashed', line_color='brown', line_width=1)
            fig.renderers.extend([av_hline, std_add_hline, std_sub_hline])
            std_box = BoxAnnotation(
                bottom=av-std, top=av+std, fill_alpha=0.2, fill_color='yellow')
            fig.add_layout(std_box)

            col1.bokeh_chart(fig)

            col2.subheader('Statistics')
            resAv = 'average: ' + str(round(av, 2)) + '±' + str(round(std, 2))
            resRelStd = 'rel. s.d.: ' + str(round(reldev, 2)) + '%'
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
            fig.xaxis.axis_label = r'''Fe$^{3+}$/$\Sigma$Fe (FP, TAP2)'''
            fig.yaxis.axis_label = r'Fe$^{3+}$/$\Sigma$Fe (FP, TAP4)'

            x = np.linspace(
                0, 1.1 * st.session_state.resultsFe3Drift['Fe$^{3+}$/$\Sigma$Fe (2TAPL)'].max(), 10)
            fig.line(x, x)
            fig.line(x, x + .01, line_dash='dashed', line_color='orange')
            fig.line(x, x - .01, line_dash='dashed', line_color='orange')

            st.bokeh_chart(fig)


# --------  End Drift Inspection

# -------- Start Sample Inspection

    def sampleInspection(sel):
        import streamlit as st
        from bokeh.plotting import figure, output_file, ColumnDataSource
        from bokeh.models import Span, BoxAnnotation, Label
        from bokeh.layouts import gridplot
        import numpy as np

        def plotStyle(data):
            av = np.nanmean(data[2])
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

            av_hline = Span(location=av, dimension='width',
                            line_dash='dashed', line_color='brown', line_width=2)
            std_add_hline = Span(location=av+std, dimension='width',
                                 line_dash='dashed', line_color='brown', line_width=1)
            std_sub_hline = Span(location=av-std, dimension='width',
                                 line_dash='dashed', line_color='brown', line_width=1)
            fig.renderers.extend([av_hline, std_add_hline, std_sub_hline])
            std_box = BoxAnnotation(
                bottom=av-std, top=av+std, fill_alpha=0.2, fill_color='yellow')
            fig.add_layout(std_box)

            statistics = Label(x=130, y=90, x_units='screen', y_units='screen',
                               text=str(data[0]) + '\n' + str(round(av, 2)) + '±' + str(
                                   round(std, 2)) + ' wt%  –  ' + 'rel. s.d.:' + str(round(reldev, 2)) + '%',
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
                fil = st.session_state.dfMain['Name'] == i
                xdata = st.session_state.dfMain[fil].loc[:, 'Point Nr.']
                data = st.session_state.dfMain[fil].loc[:, el]
                dat = (i, xdata, data)
                plotList.append(dat)

            grid_layout = gridplot([plotStyle(i)
                                    for i in plotList], ncols=int(noc))
            st.bokeh_chart(grid_layout)

        elif sel == 'Select one sample, display all elements':
            elements = st.session_state.dfMain.columns.tolist()[3:]
            smpNames = st.session_state.dfSampleNames
            smp = st.selectbox('Select a Sample', smpNames)
            noc = st.number_input('Insert the Number of Plot Columns', value=3)

            plotList = []
            for i in elements:
                fil = (st.session_state.dfMain['Name'] == smp)
                xdata = st.session_state.dfMain[fil].loc[:, 'Point Nr.']
                data = st.session_state.dfMain[fil].loc[:, i]
                dat = (i, xdata, data)
                plotList.append(dat)

            grid_layout = gridplot([plotStyle(i)
                                    for i in plotList], ncols=int(noc))
            st.bokeh_chart(grid_layout)

        elif sel == 'Select one sample and one element':
            elements = st.session_state.dfMain.columns.tolist()[3:]
            smpNames = st.session_state.dfSampleNames
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

            fig = figure(width=500, height=300, tooltips=TOOLTIPS)

            fig.line(smpSel.loc[:, 'Point Nr.'], smpSel.loc[:, el])
            output_file("toolbar.html")
            source = ColumnDataSource(smpSel.to_dict('list'))

            fig.circle('Point Nr.', el, size=4, source=source)

            fig.xaxis.axis_label = 'Point Nr.'
            fig.yaxis.axis_label = el + ' (wt%)'

            av_hline = Span(location=av, dimension='width',
                            line_dash='dashed', line_color='brown', line_width=2)
            std_add_hline = Span(location=av+std, dimension='width',
                                 line_dash='dashed', line_color='brown', line_width=1)
            std_sub_hline = Span(location=av-std, dimension='width',
                                 line_dash='dashed', line_color='brown', line_width=1)
            fig.renderers.extend([av_hline, std_add_hline, std_sub_hline])
            std_box = BoxAnnotation(
                bottom=av-std, top=av+std, fill_alpha=0.2, fill_color='yellow')
            fig.add_layout(std_box)

            col1.bokeh_chart(fig)

            col2.subheader('Statistics')
            resAv = 'average: ' + str(round(av, 2)) + '±' + str(round(std, 2))
            resRelStd = 'rel. s.d.: ' + str(round(reldev, 2)) + '%'
            col2.write(resAv)
            col2.write(resRelStd)
            if reldev < 1:
                col2.color_picker('good data', '#39EC39')
            elif 1 <= reldev < 5:
                col2.color_picker('check data', '#F7CF0F')
            else:
                col2.color_picker('worrisome data', '#FF0000')


# -------- End Sample Inspection

# --------  Start datainspection ResInsp


    def visResInsp():
        import streamlit as st
        from bokeh.plotting import figure

        st.markdown('<h4>Samples</h4>', unsafe_allow_html=True)
        st.session_state.el = st.session_state.smp_output_file.drop(
            columns=['Name', 'index']).columns

        col1, col2 = st.columns([1, 2])
        with col1:
            xaxis = st.selectbox('x-axis samples', st.session_state.el)
            yaxis = st.selectbox(
                'y-axis samples', st.session_state.el, index=2)
        with col2:
            fig = figure(width=600, height=300)
            fig.scatter(st.session_state.smp_output_file[xaxis],
                        st.session_state.smp_output_file[yaxis])
            fig.xaxis.axis_label = xaxis
            fig.yaxis.axis_label = yaxis
            st.bokeh_chart(fig)

        st.markdown('<h4>Standards</h4>', unsafe_allow_html=True)

        fil = st.session_state.std_output_file['Name'].isin(
            st.session_state.stdSelection)
        st.session_state.disp_sel_std = st.session_state.std_output_file[fil]
        st.session_state.disp_std = st.session_state.std_output_file[~fil]

        st.session_state.el2 = st.session_state.std_output_file.drop(
            columns=['Name', 'index']).columns

        col1, col2 = st.columns([1, 2])
        with col1:
            xaxis = st.selectbox('x-axis standards', st.session_state.el2)
            yaxis = st.selectbox('y-axis standards',
                                 st.session_state.el2, index=7)
        with col2:
            fig = figure(width=600, height=300)
            fig.scatter(st.session_state.disp_std[xaxis],
                        st.session_state.disp_std[yaxis], marker='circle', size=6, fill_color='blue', line_color='darkgrey', legend_label='std')
            fig.scatter(st.session_state.disp_sel_std[xaxis],
                        st.session_state.disp_sel_std[yaxis], marker='circle', size=6, fill_color='orange', line_color='darkgrey', legend_label='param std')
            fig.xaxis.axis_label = xaxis
            fig.yaxis.axis_label = yaxis
            st.bokeh_chart(fig)

# --------  End datainspection ResInsp

# --------  Start Error Considerations

    def errorConsiderations():
        import streamlit as st
        from bokeh.plotting import figure
        from bokeh.models import Range1d
        from bokeh.layouts import gridplot

    ## -----------------------------------------------##
    ## -------------  result variations --------------##
    ## -----------------------------------------------##

        def errorPercentDeviations():
            colorList = ['olive', 'orange']

            fig1 = figure(title='2TAPL', width=400, height=250)
            tmp = 1
            for i in [1, 2]:
                del tmp
                tmp = st.session_state.dfMeasSmpDataTAP2.copy()
                tmp[r'Fe$_{tot}$'] = tmp[r'Fe$_{tot}$'] * .01 * (100 - i)
                yData1 = regressionFitParameters(
                    st.session_state.dfMeasSmpDataTAP2, 'TAP2') - regressionFitParameters(tmp, 'TAP2')
                fig1.line(range(len(yData1)), yData1,
                          color=colorList[i-1], legend_label=str(i) + '%')

            fig1.xaxis.axis_label = 'sample'
            fig1.yaxis.axis_label = r'$$\textrm{absolute deviation of } Fe^{3+} / \Sigma Fe$$'
            fig1.y_range = Range1d(0, 1.3 * yData1.max())

            fig2 = figure(title='4TAPL', width=400, height=250)
            for i in [1, 2]:
                del tmp
                tmp = st.session_state.dfMeasSmpDataTAP4.copy()
                tmp[r'Fe$_{tot}$'] = tmp[r'Fe$_{tot}$'] * .01 * (100 - i)
                yData2 = regressionFitParameters(
                    st.session_state.dfMeasSmpDataTAP4, 'TAP4') - regressionFitParameters(tmp, 'TAP4')
                fig2.line(range(len(yData2)), yData2,
                          color=colorList[i-1], legend_label=str(i) + '%')

            fig2.xaxis.axis_label = 'sample'
            fig2.yaxis.axis_label = r'$$\textrm{absolute deviation of } Fe^{3+} / \Sigma Fe$$'
            fig2.y_range = Range1d(0, 1.3 * yData1.max())

            fig3 = figure(title='2TAPL', width=400, height=250)
            for i in [1, 2]:
                del tmp
                tmp = st.session_state.dfMeasSmpDataTAP2.copy()
                tmp[r'L$\beta$/L$\alpha$ (TAP2)'] = tmp[r'L$\beta$/L$\alpha$ (TAP2)'] * .01 * (
                    100 - i)
                yData3 = regressionFitParameters(
                    st.session_state.dfMeasSmpDataTAP2, 'TAP2') + regressionFitParameters(tmp, 'TAP2')
                fig3.line(range(len(yData3)), yData3,
                          color=colorList[i-1], legend_label=str(i) + '%')

            fig3.xaxis.axis_label = 'sample'
            fig3.yaxis.axis_label = r'$$\textrm{absolute deviation of } Fe^{3+} / \Sigma Fe$$'
            fig3.y_range = Range1d(0, 1.3 * yData3.max())

            fig4 = figure(title='4TAPL', width=400, height=250)
            for i in [1, 2]:
                del tmp
                tmp = st.session_state.dfMeasSmpDataTAP4.copy()
                tmp[r'L$\beta$/L$\alpha$ (TAP4)'] = tmp[r'L$\beta$/L$\alpha$ (TAP4)'] * .01 * (
                    100 - i)
                yData4 = regressionFitParameters(
                    st.session_state.dfMeasSmpDataTAP4, 'TAP4') + regressionFitParameters(tmp, 'TAP4')
                fig4.line(range(len(yData4)), yData4,
                          color=colorList[i-1], legend_label=str(i) + '%')

            fig4.xaxis.axis_label = 'sample'
            fig4.yaxis.axis_label = r'$$\textrm{absolute deviation of } Fe^{3+} / \Sigma Fe$$'
            fig4.y_range = Range1d(0, 1.3 * yData3.max())

            grid_layout = gridplot([fig1, fig2, fig3, fig4], ncols=2)
            st.bokeh_chart(grid_layout)

    ## -----------------------------------------------##
    ## ---------------  sample s.d.  ----------------##
    ## -----------------------------------------------##

        def errorSmpFe3Dev():
            import streamlit as st
            from bokeh.plotting import figure
            import numpy as np

            fig = figure(title='Samples', width=400, height=250)

            LRatioSmp = []
            for smpname in st.session_state.smpList:
                fil = st.session_state.dfMain['Name'] == smpname
                r = st.session_state.dfMain[fil][r'L$\beta$ (TAP2)'] / \
                    st.session_state.dfMain[fil][r'L$\alpha$ (TAP2)']
                LRatioSmp.append(np.std(r))

            fig.line(range(len(LRatioSmp)), LRatioSmp)
            fig.xaxis.axis_label = 'sample'
            fig.yaxis.axis_label = r'abs. 1 s.d. of Lb/Ls of a single sample'
            # fig.set_ylim(0,.025)
            st.bokeh_chart(fig)

            drList = st.session_state.dfdr['Name'].drop_duplicates().tolist()

            fig = figure(title='Drift Monitor', width=400, height=250)
            LRatioDrift = []
            for smpname in drList:
                fil = st.session_state.dfMain['Name'] == smpname
                r = st.session_state.dfMain[fil][r'L$\beta$ (TAP2)'] / \
                    st.session_state.dfMain[fil][r'L$\alpha$ (TAP2)']
                LRatioDrift.append(np.std(r))

            fig.line(range(len(LRatioDrift)), LRatioDrift)
            fig.xaxis.axis_label = 'sample'
            fig.yaxis.axis_label = r'abs. 1 s.d. of Lb/La of a single sample'
            st.bokeh_chart(fig)


# ------------------
# ---- Start Website
# ------------------

        sel = st.radio('', ('How Fe3+ changes when Fetot and Lb/La change',
                            'abs. 1 s.d. of Lb/La of Samples and Drift Monitor'), horizontal=True)

        if sel == 'How Fe3+ changes when Fetot and Lb/La change':
            errorPercentDeviations()
        else:
            errorSmpFe3Dev()

# --------  End Error Considerations

# --------  Start Comparing Lalpha & Lbeta

    def comparinglalphalbeta():
        import streamlit as st
        from bokeh.plotting import figure
        import numpy as np

        plwidth = 400
        plheight = 250

        col1, col2, col3 = st.columns(3)

        col1.markdown('**Sample measurements only**')
        fig = figure(width=plwidth, height=plheight)
        tapl2Fe3 = st.session_state.resultsFe3Smp[r'Fe$^{3+}$/$\Sigma$Fe (2TAPL)']
        tapl4Fe3 = st.session_state.resultsFe3Smp[r'Fe$^{3+}$/$\Sigma$Fe (4TAPL)']
        fig.scatter(tapl2Fe3, tapl4Fe3, color='teal')

        x = np.linspace(0, 1, 10)
        fig.line(x, x)
        fig.line(x, x + .02, line_dash='dashed', line_color='orange')
        fig.line(x, x - .02, line_dash='dashed', line_color='orange')

        fig.xaxis.axis_label = r"$$Fe^{3+} / \Sigma Fe \textrm{ (2TAPL)}$$"
        fig.yaxis.axis_label = r'$$Fe^{3+} / \Sigma Fe \textrm{ (4TAPL)}$$'
        col1.bokeh_chart(fig)

        col1.markdown('**All individual measurements**')
        fig = figure(width=plwidth, height=plheight)
        tapl2Betacps = st.session_state.dfMain[r'L$\beta$ (TAP2)']
        tapl2Alphacps = st.session_state.dfMain[r'L$\alpha$ (TAP2)']
        tapl4Betacps = st.session_state.dfMain[r'L$\beta$ (TAP4)']
        tapl4Alphacps = st.session_state.dfMain[r'L$\alpha$ (TAP4)']

        fig.scatter(tapl2Betacps, tapl2Alphacps, legend_label='2TAPL')
        fig.scatter(tapl4Betacps, tapl4Alphacps,
                    color='olive', legend_label='4TAPL')
        fig.xaxis.axis_label = r'$$L\beta \textrm{ (net intensities)}$$'
        fig.yaxis.axis_label = r'$$L\alpha \textrm{ (net intensities)}$$'
        col1.bokeh_chart(fig)

        col3.markdown('**All individual measurements**')
        fig = figure(width=plwidth, height=plheight)
        fig.line(st.session_state.dfMain['Point Nr.'], (tapl2Betacps /
                                                        tapl2Alphacps)/(tapl4Betacps/tapl4Alphacps), color='teal')
        fig.xaxis.axis_label = 'Point Nr.'
        fig.yaxis.axis_label = r'$$L\beta/L\alpha \textrm{ (2TAPL)} /L\beta/L\alpha \textrm{ (4TAPL)}$$'
        col3.bokeh_chart(fig)

        col3.markdown('**All individual measurements**')
        fig = figure(width=plwidth, height=plheight)
        fig.scatter(st.session_state.dfMain['Point Nr.'],
                    tapl2Betacps/tapl2Alphacps, legend_label='2TAPL')
        fig.scatter(st.session_state.dfMain['Point Nr.'], tapl4Betacps /
                    tapl4Alphacps, color='olive', legend_label='4TAPL')
        fig.xaxis.axis_label = 'Point Nr.'
        fig.yaxis.axis_label = r'$$L\beta/L\alpha \textrm{ (counts-ratio)}$$'
        col3.bokeh_chart(fig)


# --------  End Comparing Lalpha & Lbeta


# ----------------------------------
# ------------ Side Bar ------------
# ----------------------------------

    plotSel = st.sidebar.radio('Select your Detail', ('Fe3+ Results Plot', 'Drift Inspection',
                                                      'Sample Inspection', 'Results Inspection', 'Comparing La & Lb', 'Error Considerations'))

    if plotSel == 'Fe3+ Results Plot':
        st.subheader('Fe3+ Results Plot')
        resultsplotPlot()
    elif plotSel == 'Drift Inspection':
        st.subheader('Drift Inspection')
        sel = st.radio('Choose what to inspect', ('Composition of drift standards',
                                                  'Fe3+ using 2TAPL vs. Fe3+ using 4TAPL'), horizontal=True)
        driftplots(sel)
    elif plotSel == 'Sample Inspection':
        st.subheader('Sample Inspection')
        sel = st.selectbox('Select', ('Select one element, display all samples',
                                      'Select one sample, display all elements', 'Select one sample and one element'))
        sampleInspection(sel)
    elif plotSel == 'Results Inspection':
        st.subheader('Results Inspection')
        visResInsp()
    elif plotSel == 'Comparing La & Lb':
        st.subheader('Comparing La & Lb')
        comparinglalphalbeta()
    elif plotSel == 'Error Considerations':
        st.subheader('Error Considerations')
        errorConsiderations()

    with st.sidebar:
        with st.expander("Instructions for this site"):
            st.write("""
                Use the various data inspection tools to analyse your data and optimise your upload file.
                Check the 'Tutorials & Instructions' resource on how to do this.
            """)

    with st.sidebar:
        with st.expander("Info Fe3+ Results Plot"):
            st.write("""
                The plot visualises the formula, with which the Fe3+ in the samples are calculated.
            """)

    with st.sidebar:
        with st.expander("Info Drift Inspection"):
            st.write("""
                Check the composition of the dirft monintor measurements and identify the stability of/variations duringduring the measurement campaign.
                Also check how the Fe3+ abundances of the two analyser crystals compare.
            """)

    with st.sidebar:
        with st.expander('Info Sample Inspection'):
            st.write("""
                This provides comprehensive possibilities to check the composition of all samples in various overviews to high granularity.
            """)

    with st.sidebar:
        with st.expander('Info Results Inspection'):
            st.write("""
                This provides comprehensive possibilities to check the composition of all results in various overviews to high granularity.
            """)

    with st.sidebar:
        with st.expander("Info Comparing La & Lb"):
            st.write("""
                The plots provide insights to potential issues during the measurements.
            """)

    with st.sidebar:
        with st.expander('Info Error Considerations'):
            st.write("""
        How Fe3+ changes when FeT and Lb/La change-- Individual samples are plotted along the x-axis.\
        For each sample, the FeT (top 2 plots) and\
        Lbeta/Lalpha (bottom 2 plots) are changed by the percentage\
        given in the legend. The Fe3+/FeT is then calculated with\
        the new Fetot or Lbeta/Lalpha. The result is then subtracted\
        from the true Fe3+/FeT and plotted on the y-axis.\
            ---  sample s.d.--
        Individual samples/drift monitors are plotted along the x-axis.\
        The 1 s.d. of Lbeta/Lalpha of a single sample is calculated and\
        plotted on the y-axis.
            """)


if st.session_state.dfFitData is not None:
    datainspection()
else:
    st.write("Please upload data in 'Data Upload and Reduction'")
