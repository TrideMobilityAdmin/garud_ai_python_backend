from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import routes
from app.api.v1.maintenance_routes import maintenance_router
from app.utils.maintenance_utils import initialize_assets

app = FastAPI(
    title="Garud AI APIs",
    description="API for aircraft maintenance estimation and analysis",
    version="1.0.0",
)

# Initialize assets on startup
@app.on_event("startup")
async def startup_event():
    """Initialize assets when the application starts"""
    print("Initializing maintenance prediction assets...")
    success = await initialize_assets()
    if not success:
        print("WARNING: Failed to initialize maintenance assets")
    else:
        print("Maintenance assets initialized successfully")

# Include existing routes
app.include_router(routes.router)

# Include maintenance routes with proper prefix
app.include_router(maintenance_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Garud AI APIs!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)