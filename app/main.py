from fastapi import FastAPI

app = FastAPI(
    title = "MLOps with AWS"
)

@app.get("/")
def home():
    return {"message": "Halo dari FastAPI dalam folder app!"}