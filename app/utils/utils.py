from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")

if API_KEY is None:
    raise ValueError("API_KEY environment variable is not set!")

API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

def validate_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or not isinstance(api_key, str) or api_key.strip() == "":
        raise HTTPException(status_code=403, detail="API key is required!")
    return api_key

def get_s3_client():
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID_ML'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ML'),
        region_name=os.getenv('AWS_REGION')
    )
    return session.client('s3')

def get_sagemaker_client():
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID_ML'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ML'),
        region_name=os.getenv('AWS_REGION')
    )
    return session.client('sagemaker')