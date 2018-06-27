import pandas as pd
import pymrmr
df=pd.read_csv('G:\data for manuscripts\AVIRIS20100517\\all_new\ROIs\discretise-20180625.csv')
pymrmr.mRMR(df,'MID',15)