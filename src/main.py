from fastapi import FastAPI
from src.routes import router

app = FastAPI(title="Adaptive Diagnostic Engine")

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Adaptive Testing API running"}