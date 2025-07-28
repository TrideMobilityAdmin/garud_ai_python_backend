from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import routes

app = FastAPI(
    title="Garud AI APIs",
    description="API for aircraft maintenance estimation and analysis",
    version="1.0.0",
)

app.include_router(routes.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:3000"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Garud AI APIs!"}