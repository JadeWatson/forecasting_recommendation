import argparse
import datetime
import pandas as pd
import os
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
from functional_forms import FunctionalForms,melt_df,create_plot
from frontend import BaseProcessor
from typing import Optional
from contextlib import contextmanager
from numerize import numerize

def change_summary(df,metric):
    """Summarizes the changes between previous forecasted value and new foreacsted value
    Note: if a forecasted month is not changed, it will not be summarized here but will be summarized in the final summary
    """
    df1 = df.copy()
    diff = df1[f"new_{metric}"] - df1[f"{metric}"]
    
    def _custom_numerize(val):
        try:
            vals = float(val)
        except Exception:
            return val
        formatted = numerize.numerize(abs(vals), 1)
        output = f'({formatted})' if vals < 0 else formatted
        return f'${output}'
    
    df1['BoY'] = diff.apply(_custom_numerize)
    
    # Compute mean of the numeric diff column
    mean_diff = diff.mean()

    total_row = {col: '' for col in df1.columns}  # blanks for other cols
    total_row['BoY'] = _custom_numerize(mean_diff)

    df1.loc['Average'] = total_row
    return df1

class FinalProcessor(BaseProcessor):
    """
    This class prepares and edits the final recommendations for cc ar and adb forecasting. Inherites properties from Base Processor and composes an instance of ratio_formatting.
    Process:
    1. Process data use Base Processor to get tracking vintages in proper format
    2. Compute row-wise weighted sum to get the new ratio to be applied to changing vintage
    3. Process data using Base Processor for the changing vintage 
    4. Calculate new suggested metric to months of choice
    5. Only return changing dates x vintage x portfolio
    """
    def __init__(self,data,portfolio,metric,date_choices,products,obs_period,vin_change:dict,weights:list,application_months:list,
                ratio_converter: Optional['FunctionalForms']=None
                ):
        # inherited inputs from Base Processor
        super().__init__(data,portfolio,metric,date_choices,products,obs_period)
        # dependecy injection for FunctionalForms 
        self.ratio_converter = ratio_converter or FunctionalForms(metric=metric,base=self)
        # unique inputs for recommender
        self.vin_change = vin_change
        self.weights = weights
        self.months = application_months
    
    def proposed_change(self,row):
        """This function accomodates for changing of only particular months as well as all depending on the input months list. I.e. include all forecasted months if you want to change all, otherwise will only be applied to selected months"""
        if (str(row['as_of_dt']) in self.months):
            return row["new_ratio"]
        else:
            return row[str(f"{self.vin_year} actuals")]
    
    # def mean_gap_ratio(self):
    #     return
    
    def recommendation(self):
        self.vin_year = list(self.vin_change.keys())[0]
        obs_year = self.vin_change[self.vin_year]
        df_data = self.data
        
        # get min forecast date such that we remove such month for the forecasted portion
        min_forecast = df_data[df_data['scenario'] == 'forecast'][['as_of_dt']].min()
        min_date = min_forecast['as_of_dt']
        self.data = df_data[~((df_data['scenario'] == 'forecast') & (df_data['as_of_dt'] == min_date))]
        
        # get the processed DF once
        dff_stacked = self.base_process()
        
        # compute the ratio table from FunctionalForms
        df_ratio = self.ratio_converter.ratio_formatting(processed_df=dff_stacked)
        df_ratio = df_ratio.set_index('Month')
        
        # get row-wise weighted sum to caluclate the new ratio or call the similarity metric here to return the new ratio
        df_ratio['new_ratio'] = (df_ratio * self.weights).sum(axis=1)
        dff = df_ratio.reset_index()
        
        # get the anchor metric value to apply to the new ratio -- Note: need to pass in a new date.choices 
        dff_stacked2 = self.base_process(date_choices=self.vin_change)
        
        # Adjust new ratio such that its only applied to input months 
        df_ratio2 = self.ratio_converter.ratio_formatting(processed_df=dff_stacked2)
        df_ratio2 = df_ratio2.set_index('Month')
        
        ratio_plot = df_ratio.merge(df_ratio2,on='Month').reset_index()
        ratio_plot['as_of_dt'] = ratio_plot['Month'].apply(lambda m: pd.to_datetime(f"{obs_year}-01-01") + pd.DateOffset(months=int(m)-1))
        ratio_plot['as_of_dt'] = pd.to_datetime(ratio_plot['as_of_dt']).dt.strftime('%Y-%m-%d')
        
        # update new ratio
        ratio_plot['new_ratio'] = ratio_plot.apply(lambda row: self.proposed_change(row),axis=1)
        
        # filter dataframe for plotting 
        to_plot = ratio_plot.drop(columns={'as_of_dt'})
        
        # call plotting funcrtion in functional forms to see updated output
        plot_new_forms = self.date_choices
        plot_new_forms[f"{self.vin_year} actuals"] = str(obs_year)
        plot_new_forms["new_ratio"] = str(obs_year)
        display(create_plot(to_plot,'Adjusted Forms',metric=self.metric,func_form_choice=plot_new_forms,cur_month=10))
        
        # start building out final recommendation dataframe
        dff_stacked2 = dff_stacked2[dff_stacked2['scenario'] == 'actuals'] 
        df = ratio_plot[['as_of_dt','new_ratio',f"{self.vin_year} forecast"]].merge(dff_stacked2[[1]],how='cross')
        df[f"new_{self.metric}"] = df['new_ratio']*df[1]
        df[self.metric] = df[f"{self.vin_year} forecast"]*df[1]
        
        # Build dataframe to match correct format of as_of_dt x portfolio x vintage x prev metric x new metric
        df['vintage'] = self.vin_year
        df['portfolio'] = self.port
        df_filt = df[df['as_of_dt'] > min_date][['as_of_dt','portfolio','vintage',self.metric,'new_'+self.metric]]
        df_filt['scenario'] = 'forecast'
        return df_filt

    

    

