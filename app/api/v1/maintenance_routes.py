from fastapi import APIRouter, HTTPException, status
from app.models.maintenance_models import PredictRequest, PredictResponse, HealthResponse
from app.services.prediction_service import (
    predict_tasks, get_health_status, debug_aircraft_type, get_stats, clear_cache
)

maintenance_router = APIRouter(tags=["Task Prediction API"])

@maintenance_router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(payload: PredictRequest):
    """
    Enhanced prediction endpoint with preserved old code logic
    """
    if not payload.tasks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tasks are required for prediction."
        )
    
    if not payload.aircraft_type.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="aircraft_type is required"
        )
    
    # Validate task descriptions
    for i, task in enumerate(payload.tasks):
        if not task.task_description.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"task_description is required for task {i}"
            )
    
    try:
        result = await predict_tasks(payload)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@maintenance_router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health Check Endpoint
    """
    return await get_health_status()

@maintenance_router.get("/debug/{aircraft_type}")
async def debug_aircraft_type_endpoint(aircraft_type: str):
    """
    Debug endpoint to check aircraft type data
    """
    try:
        result = await debug_aircraft_type(aircraft_type)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@maintenance_router.get("/stats")
async def get_stats_endpoint():
    """Get comprehensive API statistics"""
    return await get_stats()

@maintenance_router.post("/clear_cache")
async def clear_cache_endpoint():
    """Clear LRU caches to free memory"""
    try:
        result = await clear_cache()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )