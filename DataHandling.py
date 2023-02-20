###Imports that we use in our project
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pprint import pprint
from time import sleep
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import statistics
import seaborn as sns

df=pd.read_csv("FinalProject.csv")
df1=df.copy()
columns_to_drop = ['TotalCloudsCover','TotalLowCloudsCover','LowCloudsBase','LowCloudsType','MediumCloudsType','HightCloudsType','CurrentWeather','PastWeather','Visibility','Pressureatstationlevel','Pressureatsealevel',"Station"]
df1 = df1.drop(columns=columns_to_drop)
print(df1.head())
df1.drop_duplicates()
mask = (df1['DateandTime'] > '2021-11-01') & (df1['DateandTime'] < '2022-04-01')  
mask = mask | (df1['DateandTime'] > '2020-11-01') & (df1['DateandTime'] < '2021-04-1')
mask = mask | (df1['DateandTime'] > '2018-11-01') & (df1['DateandTime'] < '2019-4-1')
df1 = df1[mask]
###Here we are making the coulmns "WindDirection" and "WindSpeed" to numeric one.
###That help up to fill those coulmns that have missing values with fillna function.
###We also make "DateandTime" column to datetime format, that help us to work with the dates on that column.
df1['WindDirection'] = pd.to_numeric(df1['WindDirection'], errors='coerce')
df1['WindSpeed'] = pd.to_numeric(df1['WindSpeed'], errors='coerce')
df1['DateandTime'] = pd.to_datetime(df1['DateandTime'], format='%d/%m/%Y %H:%M')

###This function help us fill the missing values, she get the column and year that need to be filling 
###and the year we need to take it from
###Then she run all over the month and for each month she collect the mesian of the years we need to take our data from
###put them all in a list and make a median to this 
###Now we take the median we get and fill all the missing values in the the same month

median_values = []
def fill_wind_direction_missing_values(column_to_fill,year_to_fill, years_to_take, start_month, end_month):
    format = '%m/%d/%Y'
    end_month =  end_month + 12 if  start_month > end_month else end_month
    for m in range(start_month, end_month + 1):
        effective_month = m if m <= 12 else (m % 12)
        median_values = []
        for y in years_to_take:
            start = datetime.datetime(y, effective_month, 1)
            end = start + relativedelta(months=+1)
            print(f"(df1['DateandTime'] >= {start.strftime(format)}) & (df1['DateandTime'] < {end.strftime(format)})")
            mask = (df1['DateandTime'] >= start.strftime(format)) & (df1['DateandTime'] < end.strftime(format))
            median_values.append(df1[mask][column_to_fill].median())
        median = statistics.median(median_values)
        fill_start = datetime.datetime(year_to_fill if effective_month <= 12 else year_to_fill + 1, effective_month, 1)
        fill_end = fill_start + relativedelta(months=+1)
        mask = (df1['DateandTime'] >= fill_start.strftime(format)) & (df1['DateandTime'] < fill_end.strftime(format))
        print(f"(df1['DateandTime'] >= {fill_start.strftime(format)}) & (df1['DateandTime'] < {fill_end.strftime(format)}")
        df1.loc[mask, column_to_fill] = df1[mask][column_to_fill].fillna(median)
        df1[mask].info()
###Here we using the function and fiil the missing vals in the year 2019
fill_wind_direction_missing_values('WindDirection',2018, [2020,2021], 11, 12)
fill_wind_direction_missing_values('WindDirection',2019, [2021,2022], 1, 3)
fill_wind_direction_missing_values('WindSpeed',2018, [2020,2021], 11, 12)
fill_wind_direction_missing_values('WindSpeed',2019, [2021,2022], 1, 3)
df1.info()
df1.WindDirection.shape
df1.to_csv('AfterCleaning_V2.csv', index=False)





