from fastapi import FastAPI
from dotenv import load_dotenv
from app.services import predict
from app.utils.utils import validate_api_key

app = FastAPI(
    title = "API"
)

@app.get("/") # ddefault
def health_check():
    return {"status": "ok"}

app.include_router(predict.router)