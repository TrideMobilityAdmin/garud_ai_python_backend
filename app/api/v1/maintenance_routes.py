from typing import List
from fastapi import APIRouter, HTTPException, status
from app.services.maintenance_prediction_service import MaintenancePredictionService
from app.models.maintenance_models import MaintenanceTasksInput, MaintenanceTaskResponse

maintenance_router = APIRouter(tags=["Maintenance Estimation"])

# Initialize service
maintenance_service = MaintenancePredictionService()

@maintenance_router.post("/predict", response_model=MaintenanceTaskResponse)
async def predict_maintenance_tasks(payload: MaintenanceTasksInput):
    """
    ORIGINAL API ENDPOINT: Predict maintenance tasks MHS and details with REPLACEMENT sum logic
    Ultra-fast prediction endpoint with order preservation - EXACT SAME AS ORIGINAL
    """
    if not payload.tasks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tasks are required for maintenance prediction."
        )
    
    if not payload.aircraft_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aircraft type is required for maintenance prediction."
        )

    try:
        # Convert pydantic models to dict format expected by service - ORIGINAL FORMAT
        tasks_dict = [{"task_number": task.task_number, "task_description": task.task_description} 
                     for task in payload.tasks]
        
        result = await maintenance_service.predict_tasks(
            tasks=tasks_dict,
            aircraft_type=payload.aircraft_type,
            aircraft_reg=payload.aircraft_reg
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@maintenance_router.get("/health")
async def maintenance_health_check():
    """Health check for maintenance service - ORIGINAL ENDPOINT"""
    return await maintenance_service.health_check()

@maintenance_router.get("/stats")  
async def maintenance_stats():
    """Get maintenance service statistics - ORIGINAL ENDPOINT"""
    return await maintenance_service.get_stats()

@maintenance_router.post("/clear_cache")
async def clear_maintenance_cache():
    """Clear maintenance service caches - ORIGINAL ENDPOINT"""
    return await maintenance_service.clear_cache()