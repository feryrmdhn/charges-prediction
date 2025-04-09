from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import boto3
import os
import json
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError

load_dotenv()

ENDPOINT_NAME = os.getenv("AWS_ENDPOINT_NAME")
REGION = os.getenv("AWS_REGION")

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

router = APIRouter()

class HealthFeatures(BaseModel):
    age: int
    bmi: float
    children: int
    sex: int
    smoker: int

@router.post("/v1/predict")
def predict(features: HealthFeatures):

    try:
        df_input = pd.DataFrame([features.dict()])
        payload = df_input.values.tolist()

        print("JSON Payload:", json.dumps(payload))

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload).encode('utf-8')
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        return {"prediction": result}

    except (BotoCoreError, ClientError) as e:
        # Tangkap kesalahan dari boto3
        error_message = str(e)
        raise HTTPException(status_code=500, detail=f"AWS Error: {error_message}")

    except Exception as e:
        # Tangkap kesalahan umum
        error_message = str(e)
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {error_message}")
