import calendar
import pandas as pd
from frontend import BaseProcessor
import argparse
import datetime
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from numerize import numerize

def custom_numerize(val):
    try:
        vals = float(val)
    except Exception:
        return val
    formatted = numerize.numerize(abs(vals), 1)
    output = f'({formatted})' if vals < 0 else formatted
    return f'${output}'

def create_plot(df, title, metric, func_form_choice, cur_month: int, dol_values):
    palette = px.colors.qualitative.Plotly
    x = df['Month']

    # discover vins and colors
    vins = list(dict.fromkeys(c.rsplit(' ', 1)[0] for c in df.columns if c != 'Month'))
    color_map = {v: palette[i % len(palette)] for i, v in enumerate(vins)}

    # simple helpers
    actual_cols_for = lambda v: [c for c in df.columns if c.startswith(v) and 'forecast' not in c.lower()]
    forecast_cols_for = lambda v: [c for c in df.columns if c.startswith(v) and 'forecast' in c.lower()]

    fig = go.Figure()

    # plot actuals and forecasts (dashed)
    for v in vins:
        for a in actual_cols_for(v):
            fig.add_trace(go.Scatter(x=x, y=df[a], mode='lines',
                                     name=f"{v} in {func_form_choice.get(v,'')}",
                                     line=dict(color=color_map[v]), connectgaps=True))
        fcols = forecast_cols_for(v)
        if fcols:
            fig.add_trace(go.Scatter(x=x, y=df[fcols[0]], mode='lines',
                                     line=dict(color=color_map[v], dash='dot'),
                                     showlegend=False, connectgaps=True))

    #prepare combined last-vintage series (actuals overwritten by forecast where present)
    if vins:
        v_last = vins[-1]
        a_cols = actual_cols_for(v_last)
        fcols = forecast_cols_for(v_last)

        y_base = df[a_cols[-1]].copy() if a_cols else pd.Series(np.nan, index=df.index)

        # show only labels for dol_values (no numeric line / markers)
        if dol_values:
            dol_values = dict(dol_values) 
            months_set = set(df['Month'].tolist())
            # Keep only months that exist in df and have a non-null dol value
            x_d = [m for m, d in dol_values.items() if m in months_set and pd.notna(d)]
            # For each month, find the last-vintage y position and place an annotation with the dollar text
            for mx in x_d:
                # find index/row in df for this month
                idx = df.index[df['Month'] == mx]
                if len(idx) == 0:
                    continue
                ix = idx[0]
                y_pos = y_base.iloc[ix]
                if pd.isna(y_pos):
                    # if no y position (NaN), skip
                    continue
                #dollar_text = f"${int(round(dol_values[mx])):,.0f}"
                dollar_text = custom_numerize(dol_values[mx])

                # annotate at the last-vintage line data coordinate (yref='y'), slightly above the line
                fig.add_annotation(x=mx, y=y_pos, xref='x', yref='y',
                                   text=dollar_text, showarrow=False,
                                   font=dict(color=color_map[v_last], size=11),
                                   xanchor='left', yanchor='bottom', yshift=10)

    fig.update_layout(title=title, xaxis_title='Month', yaxis_title=f"{metric} Ratio",
                      template='plotly_white', legend=dict(y=1.05, orientation='h', groupclick="toggleitem"))
    return fig

def create_plot2(df, title, metric, func_form_choice, cur_month: int, dol_values=None):
    palette = px.colors.qualitative.Plotly
    x = df['Month']

    # discover vins and colors
    vins = list(dict.fromkeys(c.rsplit(' ', 1)[0] for c in df.columns if c != 'Month'))
    color_map = {v: palette[i % len(palette)] for i, v in enumerate(vins)}

    # simple helpers
    actual_cols_for = lambda v: [c for c in df.columns if c.startswith(v) and 'forecast' not in c.lower()]
    forecast_cols_for = lambda v: [c for c in df.columns if c.startswith(v) and 'forecast' in c.lower()]

    fig = go.Figure()

    # plot actuals and forecasts (dashed)
    for v in vins:
        for a in actual_cols_for(v):
            fig.add_trace(go.Scatter(
                x=x, y=df[a], mode='lines',
                name=f"{v} in {func_form_choice.get(v,'')}",
                line=dict(color=color_map[v]),
                connectgaps=True
            ))
        fcols = forecast_cols_for(v)
        if fcols:
            fig.add_trace(go.Scatter(
                x=x, y=df[fcols[0]], mode='lines',
                line=dict(color=color_map[v], dash='dot'),
                showlegend=False,
                connectgaps=True
            ))

    # prepare combined last-vintage series (actuals overwritten by forecast where present)
    if vins:
        v_last = vins[-1]
        a_cols = actual_cols_for(v_last)
        fcols = forecast_cols_for(v_last)

        y_base = df[a_cols[-1]].copy() if a_cols else pd.Series(np.nan, index=df.index)

        # optional: plot dol_values as a separate dashed line on right axis (kept if you want it)
        x_d = []
        y_d = []
        if dol_values:
            dol_values = dict(dol_values)
            months_set = set(df['Month'].tolist())
            x_d = [m for m, d in dol_values.items() if m in months_set and pd.notna(d)]
            y_d = [dol_values[m] for m in x_d]

            if x_d:
                fig.add_trace(go.Scatter(
                    x=x_d, y=y_d, mode='lines+markers',
                    name=f"{v_last} $",
                    line=dict(color=color_map[v_last], dash='dash'),
                    marker=dict(size=6),
                    yaxis='y2',
                    hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"
                ))
                # optional secondary axis label
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Dollar'))

        # -------- place labels with respect to the SOLID last-vintage line (y_base) --------
        if dol_values:
            # iterate months that have dollar values and exist in df
            for mx, dval in dol_values.items():
                if mx not in set(df['Month'].tolist()) or pd.isna(dval):
                    continue
                # find index for this month
                idx = df.index[df['Month'] == mx]
                if len(idx) == 0:
                    continue
                ix = idx[0]
                # IMPORTANT: use y_base (solid line) for annotation y position
                y_pos = y_base.iloc[ix]
                if pd.isna(y_pos):
                    continue
                dollar_text = custom_numerize(dval)
                # place the dollar label slightly above the solid line point
                fig.add_annotation(
                    x=mx, y=y_pos,
                    xref='x', yref='y',
                    text=dollar_text,
                    showarrow=False,
                    font=dict(color=color_map[v_last], size=11),
                    xanchor='left', yanchor='bottom', yshift=10
                )

    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title=f"{metric} Ratio",
        template='plotly_white',
        legend=dict(y=1.05, orientation='h', groupclick="toggleitem")
    )
    return fig


def melt_df(df,id_vars,value_vars,var_name,value_name):        
    df_long = df.melt(id_vars,value_vars,var_name,value_name)
    return df_long

class FunctionalForms(BaseProcessor):
    """
    This class converts dollar values metrics into anchoring to January ratio. It requires a composition of Base Processor prior to running
    Steps:
    1. Call Base Processor to get the vintages of interest in proper form
    2. Compute array division to get ratios anchored to Jan
    3. Plot using plotly
    """
        
    def __init__(self,data,portfolio,metric,date_choices,obs_period):
        super().__init__(data,portfolio,metric,date_choices,obs_period)
    
    def df_plot(self,df1,processed_df=None):
        """
        this function is specifically only for manipulating a datframe such that it can be used FF plotting without the recommender
        """ 
        df_stacked = processed_df if processed_df is not None else self.base_process()
        df_transpose = df1
        
        # create dollar value column for all columns containing forecast
        obs_year = [k for k, v in self.date_choices.items() if "2025" in str(v)]
        filt_df = df_stacked[(df_stacked['vintage'].isin(obs_year)) & (df_stacked['scenario'] == 'actuals')]
        df_stacked2 = df_transpose.merge(filt_df[[1]],how='cross')
        df_stacked2['combined_dollars'] = df_stacked2[f"{obs_year[0]} forecast"]*df_stacked2[1]
        
        # call plotting function 
        df_data = self.data
        min_forecast = df_data[df_data['scenario'] == 'forecast'][['as_of_dt']].min()
        min_date = min_forecast['as_of_dt']
        cur_month = int(min_date.split('-')[1])
        dol_values = dict(zip(df_stacked2['Month'],df_stacked2['combined_dollars']))
        return display(create_plot2(df_stacked2.drop(columns={1,'combined_dollars'}),f"{self.portfolio} Functional Forms",self.metric,self.date_choices,cur_month,dol_values))
        
    def ratio_formatting(self,processed_df=None):   
        # Call the Base Processor 
        df_stacked = processed_df if processed_df is not None else self.base_process()
        
        # creating ratios anchored to January 
        month_cols = [c for c in df_stacked.columns if isinstance(c,(int,float))]
        base_col = month_cols[0]
        
        base_vals = df_stacked[base_col].replace({0:np.nan})
        ratios_df = df_stacked[month_cols].div(base_vals,axis=0)
        df_final = pd.concat([df_stacked[['vintage','scenario']],ratios_df],axis=1)
        
        #melt the dataframe to begin plotting
        ratio_cols = [c for c in df_final.columns if isinstance(c,(int,float))]
        df_long = df_final.melt(['vintage','scenario'],ratio_cols,'Month',self.metric+'_ratio')
        
        # return in plotting format to be called by plotly function
        df_transpose = df_long.pivot_table(index='Month',columns=['vintage','scenario'],values=self.metric+'_ratio').reset_index()
        df_transpose.columns = [' '.join(map(str, col)).strip() if isinstance(col, tuple) else col 
                        for col in df_transpose.columns.values]
        return df_transpose
