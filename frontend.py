import argparse
import datetime
import pandas as pd
import os
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import plotly.graph_objects as go

def vintage_tag(row,vin_yr):
    try:
        yr = int(row['vintage'])
    except Exception:
        return row['vintage']   # preserve strings like '2012 & Prior'
    if yr <= int(vin_yr):
        return f"{vin_yr} & Prior"
    else:
        return str(row['vintage'])

class BaseProcessor:
    def __init__(self,data,portfolio:str,metric:str,date_choices:dict,products:list,obs_period:int):
        self.data = data
        self.port = portfolio
        self.metric = metric
        self.date_choices = date_choices
        self.products = products
        self.obs_period = obs_period
    
    def func_forms_df(self,df,dfs):
        df1 = df[df['portfolio'] == self.port]
        dfs[self.port+'_df'] = df1.pivot_table(index=['vintage','scenario'],columns='start_ind',values=self.metric,aggfunc='sum').reset_index()
        return dfs[self.port+'_df']
    
    def func_forms(self,vin:str,dfs):
        df_port = dfs[self.port+'_df']
        df_filt = df_port[(df_port['vintage'] == vin)]
        month_cols = [c for c in df_filt.columns if isinstance(c,(int,float)) and c <= self.obs_period]
        df_filt = df_filt[['vintage','scenario',*month_cols]]
        return df_filt
    
    def base_process(self,*,date_choices: dict = None):
        """
        This class processes raw data such that it can be used for recommendation in forecasting. 
        1. For selected vintage and observation period, check if it old book or new book. If old book change the vintage formatting.
        2. Create a new column representing observation period in months
        3. Filter to positive observation period as negative numbers represents months before the observaion period started
        4. Create a hash that stores key: portfolio Value: vintage x observation month adb 
        5. Format hash to contain vintages and observation year 
        """
        dfs = {}
        to_create = []
        
        # check if we want to include all products or remove some
        cur_df = self.data.copy()
        if self.products is not None and len(self.products)>0:
            cur_df = cur_df[~cur_df['products'].isin(self.products)]
        
        # update date choices if necessary 
        date_choices = date_choices if date_choices is not None else self.date_choices
        
        # check if we are working with new book or old book FF
        for vin,date in date_choices.items():
            start_date = pd.to_datetime(f"{date}-01-01")
            # Old book vintage 
            if 'Prior' in vin:
                vin_yr = vin.split('&')[0].strip()
                # call vintage tagging 
                cur_df['ob_vintage'] = cur_df.apply(lambda row: vintage_tag(row,vin_yr),axis=1)
                # use ob vintage tagging to populate hash
                df1 = cur_df.groupby(['as_of_dt','portfolio','ob_vintage','scenario']).agg({'adb':'sum','cc_ar':'sum'}).reset_index()
                df1 = df1.rename(columns={'ob_vintage':'vintage'})
            else:
                df1 = cur_df
                
            # calculate how many months since this start_date for each record. This will act as observation period
            df1['start_ind'] = (
                (pd.to_datetime(df1['as_of_dt']).dt.year - start_date.year) * 12 +
                (pd.to_datetime(df1['as_of_dt']).dt.month - start_date.month)
            ).astype(int) + 1
            
            # filter out negative numbers (representing previous as_of_dates i.e. out of observation period)
            df1 = df1[df1['start_ind']>=1]
            
            # populate hash with correct vintages
            self.func_forms_df(df1,dfs)
            #display(self.func_forms_df(df1,dfs))
            # create stacked vintages for plotting 
            df2 = self.func_forms(vin,dfs)
            to_create.append(df2)
        df_stacked = pd.concat(to_create,ignore_index=True)
        
        # fill the Nan in forecast with the actuals value - Old use above
        cols_to_replace = [c for c in df_stacked.columns if c not in ['vintage','scenario']]
        # get actuals first 
        df_actuals = (df_stacked[df_stacked['scenario'] == 'actuals']
                     .groupby('vintage').first().reset_index()
                     )
        # store in dictionary to be called later 
        actuals_map = df_actuals.set_index('vintage')[cols_to_replace].to_dict(orient='index')

        # get forecasted rows
        is_forecast = df_stacked['scenario'] == 'forecast'

        # check if the value in NaN in forecasted row: replace Nan with value stored in dictionary 
        # For each column, replace NaNs in forecast rows by the corresponding actuals value for that vintage
        for c in cols_to_replace:
            mask = is_forecast & df_stacked[c].isna()
            if not mask.any():
                continue
            # map vintage -> actuals value for column c, default to NaN if not found
            df_stacked.loc[mask, c] = (
                df_stacked.loc[mask, 'vintage']
                .map(lambda v: actuals_map.get(v, {}).get(c, np.nan))
            )
        return df_stacked

        

        

