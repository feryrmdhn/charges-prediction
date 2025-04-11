from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import sys
import pandas as pd
import boto3
import os
import json
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.utils.utils import validate_api_key

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

@router.post("/v1/predict", description="Predicted charges for a smoker's insurance costs")
async def predict(features: HealthFeatures, api_key: str = Depends(validate_api_key)):

    try:
        df_input = pd.DataFrame([features.dict()])
        payload = df_input.values.tolist()

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload).encode('utf-8')
        )

        result = json.loads(response["Body"].read().decode("utf-8"))

        if not isinstance(result, list) or not result:
            raise ValueError("Invalid response format from model")

        prediction = round(result[0], 2)

        return {
            "status": 200,
            "message": "success",
            "result": prediction
        }

    # Catch error from AWS sagemaker
    except (BotoCoreError, ClientError) as e:
        error_message = str(e)
        raise HTTPException(status_code=500, detail=f"AWS Error: {error_message}")

    except Exception as e:
        error_message = str(e)
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {error_message}")
