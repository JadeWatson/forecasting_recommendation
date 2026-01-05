import argparse
import datetime
import pandas as pd
import os
import tqdm
from google.cloud.bigquery import Client, QueryJobConfig
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
from numerize import numerize


def query_call(query):
    client = Client()
    job = client.query(query)
    date_df = job.to_dataframe()
    return date_df

def extrapolation(df,month:str,metric:str):
    """This function extrapolates remaining cycles if not equal to 26.
    Step 1: calculate the MoM for equivalent number of billing cycles 
    Step 2: Apply MoM to future values to get implied billing cycle metric
    """
    df['mom_'+metric] = (df[metric+'_filt_cycles']/df['prev_'+metric]) - 1   
    # get implied metric
    df['implied_'+metric] = df.groupby(['portfolio','vintage'])[metric].shift(1) + (df.groupby(['portfolio','vintage'])[metric].shift(1)*df['mom_'+metric])
    # overwrite for current month
    df[metric] = df.apply(lambda row: row['implied_'+metric] if row['as_of_dt'] == pd.to_datetime(month).date() else row[metric],axis=1)
    return df

def edit_forcasted_data(df,metrics,products):
    # create proper products
    df1 = df.pivot_table(index=['as_of_dt','portfolio','vintage'],columns='product',values=metrics,aggfunc=sum).reset_index()
    
    for m in metrics:
        if m == 'cc_ar':
            # pick only ('cc_ar', product) where product not in ['AMAZON','LOWES']
            cols = [c for c in df1.columns if c[0] == m and c[1] not in products]
            df1['sum_' + m] = df1[cols].sum(axis=1)
        else:
            # normal case: sum across all subcolumns for that metric
            df1['sum_' + m] = df1[m].sum(axis=1)
    df1.columns = [f"{c[0]}_{c[1]}" for c in df1.columns]
    df1 = df1[['as_of_dt_','portfolio_','vintage_','sum_cc_ar_','sum_adb_']]
    df1.columns = ['as_of_dt','portfolio', 'vintage','cc_ar','adb']
    return df1

# try the variance tracker 
def get_summary(df,month,portfolio,metric):
    """
    This function is used for the variance tracker. Calculates the variance between previous RnO and Actuals.
    """
    def _custom_numerize(val):
        try:
            vals = float(val)
        except Exception:
            return val
        formatted = numerize.numerize(abs(vals), 1)
        output = f'({formatted})' if vals < 0 else formatted
        return f'${output}'
    
    def _flatten_cols(df):
        df.columns = [f"{val}_{col}" if col else val for val, col in df.columns]
        df.set_index('new_vintage',inplace=True)
        return df
        
    df_month = df[(df['as_of_dt'] == month) & (df['new_vintage'].isin(['2019 & Prior', '2020', '2021', '2022', '2023', '2024', '2025']))]
    
    # check if the summary is required at portfolio level
    if portfolio != 'all':
        df_summary = df_month.pivot_table(index=['portfolio','new_vintage'],columns='scenario',values=['cc_ar','adb'],aggfunc='sum').reset_index()
        df_summary = _flatten_cols(df_summary)
        df_final = df_summary[(df_summary['portfolio'] == portfolio)][[f"{metric}_actuals",f"{metric}_forecast"]]
        print(f'{portfolio} Variance Tracking')
    else: 
        df_summary = df_month.pivot_table(index='new_vintage',columns='scenario',values=['cc_ar','adb'],aggfunc='sum').reset_index()
        df_summary = _flatten_cols(df_summary)
        df_final = df_summary[[f"{metric}_actuals",f"{metric}_forecast"]]
        print("Overall Variance Tracking")
    
    df_final = df_final.copy()
    df_final.loc['Total'] = df_final.sum(axis=0)
    
    # create variances
    df_final['Variance $'] = df_final[f"{metric}_actuals"]-df_final[f"{metric}_forecast"]
    df_final['Variance %'] = round(((df_final[f"{metric}_actuals"] / df_final[f"{metric}_forecast"]) -1)*100.0,2)
    
    # format with numerize 
    format_df = df_final.copy()
    numerize_cols = [c for c in format_df.columns if c != 'Variance %']
    format_df[numerize_cols] = format_df[numerize_cols].apply(lambda col: col.map(_custom_numerize))
    format_df['Variance %'] = format_df['Variance %'].apply(lambda x: f"({round(abs(x),2)}%)" if x<0 else f"{round(x,2)}%")
    return format_df

class DataExtrapolation:
    def __init__(self,month,bu,output_dir,future_data,products):
        self.month = month
        self.bu = bu
        self.output_dir = output_dir
        self.future_data = future_data
        self.products = products
        
    def process(self):        
        file_path = f"{self.output_dir}/pre_R&O_log.txt"
        os.makedirs(self.output_dir, exist_ok=True)

        time1 = datetime.datetime.now()  

        with open(file_path, "w") as f:
            f.write('\nINITIATING DATA PULL PROCESS\n')
            f.write(f'Starting the pull process for pre {self.month} R&O at : {time1} \n')
            f.flush() 
           
        product_list = "'"+"','".join(self.products)+"'"
        
        # Pull all actuals until current month -1 
        query = f"""
        SELECT
          as_of_dt,bu,acct_bill_cyc_cd,portfolio,vintage_dt as vintage,
          SUM(adb) AS adb,
          SUM(case when product not in ({product_list}) then cc_ar else 0 end) AS cc_ar
        FROM
          common.ar_adb_cycle_hist
        WHERE
          1=1
          AND bu = "{self.bu}"
          AND portfolio IN ('PROP','SAC','LOC','PIF')
        GROUP BY
          1,2,3,4,5
        ORDER BY
          1,2,3;
        """
        pre_data = query_call(query)
        
        # pull actuals for current month
        query = f"""
        SELECT
          as_of_dt,bu,acct_bill_cyc_cd,portfolio,vintage_dt as vintage,
          SUM(adb) AS adb,
          SUM(case when product not in ({product_list}) then cc_ar else 0 end) AS cc_ar
        FROM
          common.ar_adb_cycle_cur_month
        WHERE
          1=1
          AND bu = "{self.bu}"
          AND portfolio IN ('PROP','SAC','LOC','PIF')
        GROUP BY
          1,2,3,4,5
        ORDER BY
          1,2,3;
        """
    
        cur_data = query_call(query)
        
        df = pd.concat([pre_data,cur_data],ignore_index=True)
        
        time2 = datetime.datetime.now()
        
        # update to number of cycles
        curr_bcycl = df[df['as_of_dt'] == pd.to_datetime(self.month)]['acct_bill_cyc_cd'].nunique()
        print(curr_bcycl)
        print(df[df['as_of_dt'] == pd.to_datetime(self.month)]['acct_bill_cyc_cd'].unique())
        print(f"Latest Available Actual Month: {self.month}\n")
        print(f"Latest Available Billing Cycle: {curr_bcycl}\n")

        with open(file_path, "a") as f:
            f.write(f'Latest Available Actual Month: {self.month}\n')
            f.write(f'Latest Available Billing Cycle: {curr_bcycl}\n')
            f.flush() 

        # formatted for future extrapolation purpose
        df_grouped = df.groupby(['as_of_dt','portfolio','vintage'])[['cc_ar','adb']].sum().reset_index().sort_values(by='as_of_dt')
        
        # may need to change 26 
        if curr_bcycl != 26:
            print(f"Extrapolating for: {self.month}")
            with open(file_path, "a") as f:
                f.write(f'Extrapolating for: {self.month}\n')
                f.flush() 
            # filter to previous month with equal number of current cycles
            cur_cyc_list = df[df['as_of_dt'] == pd.to_datetime(self.month)]['acct_bill_cyc_cd'].unique()
            df_ext = df[df['acct_bill_cyc_cd'].isin(cur_cyc_list)]
            df_ext = df_ext.groupby(['as_of_dt','portfolio','vintage'])[['cc_ar','adb']].sum().reset_index().sort_values(by='as_of_dt')
            df_ext[['prev_cc_ar','prev_adb']] = df_ext.groupby(['portfolio','vintage'])[['cc_ar','adb']].shift(1)
            # merge to apply MoM
            df_mid = df_grouped.merge(df_ext,on=['as_of_dt','portfolio','vintage'],how='left',suffixes=('', '_filt_cycles')).sort_values(by='as_of_dt')
            
            metric = ['cc_ar','adb']
            for m in metric:
                # get implied metrics
                dff = extrapolation(df_mid,self.month,m)
                # send updated dataframe into function 
                df_mid = dff
        else:
            print(f"No need for extrapolation")
            with open(file_path, "a") as f:
                f.write(f'No need for extrapolation\n')
                f.flush()
            dff = df.groupby(['as_of_dt','portfolio','vintage'])[['cc_ar', 'adb']].sum().reset_index().sort_values('as_of_dt')

        # put in desired format
        dff = dff[['as_of_dt','portfolio','vintage','cc_ar','adb']]

        time3 = datetime.datetime.now()
        
        # get new vintage tag for variance tracker only for 2019 & Prior
        dff = dff.copy()
        dff.loc[:, 'new_vintage'] = dff.apply(lambda row: "2019 & Prior" if row['vintage'] <= '2019' else row['vintage'], axis=1)
        # following line not mandatory, only used for safety in case we want to run a previous RnO
        dff = dff[dff['as_of_dt'] <= pd.to_datetime(self.month)]
        dff.loc[:,'scenario'] = 'actuals'

        with open(file_path, "a") as f:
            f.write('Getting forcasted data for remaining portion of the year\n')
            f.flush() 
        
        # get forecasted data for the remainder of the year
        df_rno = pd.read_excel(f"{self.future_data}")
        df_rno = df_rno.iloc[:-3,]
        df_rno.columns = ['as_of_dt','portfolio', 'vintage','product','cc_ar','adb']
        df_rno = df_rno[df_rno['as_of_dt'] >= pd.to_datetime(self.month)]
        
        # format to remove products from cc_ar
        df_rno1 = edit_forcasted_data(df_rno,metric,self.products)
        
        # build out final
        df_rno1['scenario'] = 'forecast'
        df_rno1['new_vintage'] = df_rno1['vintage']
        
        # merge to get final df
        df_final = pd.concat([dff,df_rno1],ignore_index=True)
        df_final['as_of_dt'] = pd.to_datetime(df_final['as_of_dt'])       
        
        # before writing filter out the rno current month and only keep actual
        #df_output = df_final[~((df_final['scenario'] == 'forecast') & (df_final['as_of_dt'] == self.month))]
        df_output = df_final.drop(columns={'new_vintage'})
        df_output.to_csv(self.output_dir+self.month+".csv",index=False)

        time4 = datetime.datetime.now()

        with open(file_path,"a") as f:
            f.write(f"Finished Data Pull for forecasted {self.month} completed in {time4-time1}\n")
            f.flush()
            
        return df_final

"""
def main():
    parser = argparse.ArgumentParser("R&O Data")
    parser.add_argument("--month", type=str, default = '2024-01-01')
    parser.add_argument("--business_unit", type=str, default ="Small Business")
    parser.add_argument("--output_dir", type=str,default = os.getcwd())
    parser.add_argument("--future_data", type=str)
    args = parser.parse_args()
    
    # variable decalration
    month = args.month
    bu = args.business_unit
    output_dir = args.output_dir
    future_data = args.future_data

    # intialize data process 
    dataset = get_data(month,bu,output_dir,rno_path)
    df = dataset.process()
    
if __name__ == "__main__":
    main()
    """
