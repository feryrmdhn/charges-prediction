import sys
import os
import boto3
import numpy as np
import pandas as pd
from io import StringIO
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dotenv import load_dotenv
from get_data import load_dataframe

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.utils.utils import get_s3_client

load_dotenv()

bucket_name = os.getenv('AWS_BUCKET_NAME')
prefix = os.getenv('AWS_BUCKET_PREFIX') 

df = load_dataframe()

df = df.drop(columns=['region'])

label_encoder = LabelEncoder()
df['sex_encoded'] = label_encoder.fit_transform(df['sex'])
df['smoker_encoded'] = label_encoder.fit_transform(df['smoker'])

df = df.drop(columns=['sex', 'smoker'])
df = df.rename(columns={'sex_encoded': 'sex', 'smoker_encoded': 'smoker'}) # Change columns name as default

def handle_outliers_iqr(df, columns):
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75) 
        IQR = Q3 - Q1 
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df_clean[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    return df_clean

columns_to_fix = ['bmi', 'charges']
df_cleaned = handle_outliers_iqr(df, columns_to_fix)

def upload_dataframe_to_s3(df, key, s3_client):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=csv_buffer.getvalue()
    )
    print("File uploaded!")

s3_client = get_s3_client()
file_key = f"{prefix}/insurance_data_cleaned.csv"

if __name__ == '__main__':
    upload_dataframe_to_s3(
        df_cleaned,
        file_key,
        s3_client
    )
    
