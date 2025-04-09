import sys
import os
import boto3
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.utils.utils import get_s3_client

load_dotenv()

bucket_name = os.getenv('AWS_BUCKET_NAME')
prefix = os.getenv('AWS_BUCKET_PREFIX') 

if not bucket_name or not prefix:
    raise ValueError("Environment variables AWS_BUCKET_NAME and AWS_BUCKET_PREFIX must be set first!")

file_key = f"{prefix}/insurance.csv"

s3 = get_s3_client()

def get_data_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read().decode('utf-8')  # jika CSV
    df = pd.read_csv(StringIO(body))
    return df

def load_dataframe():
    return get_data_from_s3(bucket_name, file_key)

if __name__ == '__main__':
    df = load_dataframe()