from typing import List
from fastapi import APIRouter, HTTPException, status
from app.services.estima_services import defect_investigator, most_probable_defects,event_log_management
from app.services.defect_services import defects_prediction
from app.models.models import TasksInput, MostProbableDefectsInput, DefectInvestigatorInput

router = APIRouter(prefix="/api/v1", tags=["API's"])

@router.post("/most_probable_defects")
async def most_probable_defects_route(payload: MostProbableDefectsInput):
    """
    Endpoint to get the most probable defects based on provided parameters.
    """
    if (
        not payload.aircraft_model or 
        not payload.check_category or 
        not payload.customer_name or 
        payload.customer_name_consideration is None
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All input fields are required for most probable defects prediction."
        )

    result = await most_probable_defects(
        payload.aircraft_age,
        payload.aircraft_model,
        payload.check_category,
        payload.customer_name,
        payload.customer_name_consideration
    )
    return result

@router.post("/defect_investigator")
async def defect_investigator_route(payload: DefectInvestigatorInput):
    """
    Endpoint to investigate defects based on provided parameters.
    """
    if (
        not payload.task_number or 
        not payload.log_item_number or 
        not payload.defect_desc or 
        not payload.corrective_action
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All fields are required for defect investigation."
        )

    result = await defect_investigator(
        payload.task_number,
        payload.log_item_number,
        payload.defect_desc,
        payload.corrective_action
    )
    return result

@router.post("/defects_prediction")
async def defects_prediction_route(payload: TasksInput):
    """
    Endpoint to predict defects based on aircraft details.
    """
    if not payload.tasks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tasks are required for defect prediction."
        )

    result = await defects_prediction(payload.tasks)
    return result
@router.get("/hanger_planning")
async def hanger_planning_route():
    """
    Endpoint to get hanger planning data.
    """
    # Placeholder for hanger planning logic
    result=await event_log_management()
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No hanger planning data found."
        )
    
    return result