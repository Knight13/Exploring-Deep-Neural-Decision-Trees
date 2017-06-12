import numpy as np
import pandas as pd
import sys

df = pd.read_csv("2008.csv")

#df = pd.concat([df, pd.get_dummies(df['UniqueCarrier'], prefix = 'Carrier')], axis = 1)
#df = pd.concat([df, pd.get_dummies(df['Origin'], prefix = 'Origin')], axis = 1)
#df = pd.concat([df, pd.get_dummies(df['Dest'], prefix = 'Dest')], axis = 1)



df['Total Delay'] = (df['ArrDelay'] + df['DepDelay'])

df = df.drop(['Year', 'UniqueCarrier','FlightNum', 'TailNum', 'Origin', 'Dest','CancellationCode', 'ArrDelay', 'DepDelay','CarrierDelay', 
              'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay'], 1)
 
df = df.dropna()

df['Delay'] = np.where(df['Total Delay'] > 0.0, 'yes', 'no')  
 
df = df.drop('Total Delay', 1) 

cols = df.columns.tolist()

df_features = df[cols[:-1]]

df_labels = df[cols[-1:]]

df_labels = pd.concat([df_labels, pd.get_dummies(df_labels['Delay'], prefix = 'Delay')], axis = 1)  

df_labels = df_labels.drop('Delay', 1)

feature = df_features.as_matrix()
feature = np.array(feature, dtype = np.float64)

label = df_labels.as_matrix()
label = np.array(label, dtype = np.float64)


