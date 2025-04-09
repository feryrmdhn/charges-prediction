from fastapi import FastAPI
from dotenv import load_dotenv
from app.services import predict
from app.utils.utils import validate_api_key

app = FastAPI(
    title = "API"
)

app.include_router(predict.router)