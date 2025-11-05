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

def create_plot(df, title, metric, func_form_choice, cur_month: int):
    """
    Clean plot: df with 'Month' column, columns named like "<vin> ...",
    Returns a plotly Figure.
    """
    palette = px.colors.qualitative.Plotly
    x = df['Month']

    # discover vins and map colors
    vins = [c.rsplit(' ', 1)[0] for c in df.columns if c != 'Month']
    vins = list(dict.fromkeys(vins))  # keep order, unique
    color_map = {v: palette[i % len(palette)] for i, v in enumerate(vins)}

    # helper to find col names
    def actual_cols_for(v): return [c for c in df.columns if c.startswith(v) and 'forecast' not in c.lower()]
    def forecast_cols_for(v): return [c for c in df.columns if c.startswith(v) and 'forecast' in c.lower()]

    fig = go.Figure()

    # plot actuals
    for v in vins:
        for a in actual_cols_for(v):
            fig.add_trace(go.Scatter(
                x=x, y=df[a],
                mode='lines',
                name=f"{v} in {func_form_choice.get(v, '')}",
                line=dict(color=color_map[v]),
                connectgaps=True
            ))

    # plot forecasts + connectors
    pos_list = df.index[df['Month'] == cur_month].tolist()
    if pos_list:
        pos = pos_list[0]
        prior = max(0, pos - 1)

        for v in vins:
            fcols = forecast_cols_for(v)
            if not fcols:
                continue
            fcol = fcols[0]
            y_fore = df[fcol].iloc[prior:].reset_index(drop=True)
            x_fore = x.iloc[prior:].reset_index(drop=True)

            # dashed forecast trace (hidden legend to avoid dupes)
            fig.add_trace(go.Scatter(
                x=x_fore, y=y_fore,
                mode='lines',
                line=dict(color=color_map[v], dash='dot'),
                showlegend=False,
                connectgaps=True
            ))

            # connector: last valid actual up to (cur_month + 1), first valid forecast point
            a_cols = actual_cols_for(v)
            if not a_cols:
                continue
            a = a_cols[-1]  # pick last actual column if multiple

            # find position of (cur_month + 1)
            next_month_pos = df.index[df['Month'] == cur_month + 1].tolist()
            if next_month_pos:
                limit_pos = next_month_pos[0] + 1  # include it in slice
            else:
                limit_pos = len(df)  # fallback if not found

            # actual values up to (cur_month)
            act_slice = df[a].iloc[:cur_month]
            act_nonna = np.where(act_slice.notna().to_numpy())[0]
            fore_nonna = np.where(y_fore.notna().to_numpy())[0]

            if act_nonna.size and fore_nonna.size:
                last_act_pos = act_nonna[-1]
                first_fore_pos = prior + int(fore_nonna[2])
                if last_act_pos != first_fore_pos:
                    fig.add_trace(go.Scatter(
                        x=[x.iloc[last_act_pos], x.iloc[first_fore_pos]],
                        y=[df[a].iloc[last_act_pos], df[fcol].iloc[first_fore_pos]],
                        mode='lines',
                        line=dict(color=color_map[v], width=2,dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))


    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title=f"{metric} Ratio",
        template="plotly_white",
        legend=dict(y=1.05, orientation='h')
    )

    return fig

def melt_df(df,id_vars,value_vars,var_name,value_name):        
    df_long = df.melt(id_vars,value_vars,var_name,value_name)
    return df_long

class FunctionalForms():
    """
    This class converts dollar values metrics into anchoring to January ratio. It requires a composition of Base Processor prior to running
    Steps:
    1. Call Base Processor to get the vintages of interest in proper form
    2. Compute array division to get ratios anchored to Jan
    3. Plot using plotly
    """
        
    def __init__(self,metric:str,base: Optional['BaseProcessor'] = None):
        self.metric = metric
        self.base = base
        
    def get_processed(self):
        # calls instance of Base Processor
        if self.base is None:
            raise ValueError("No BaseProcessor provided. Either pass base or supply processed_df to ratio_formatting.")
        return self.base.base_process()
    
    def ratio_formatting(self,processed_df=None):   
        # Call the Base Processor 
        df_stacked = processed_df if processed_df is not None else self.get_processed()
        
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
