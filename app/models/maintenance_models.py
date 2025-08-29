from pydantic import BaseModel
from typing import List, Optional

class Task(BaseModel):
    task_number: str
    task_description: str

class PredictRequest(BaseModel):
    tasks: List[Task]
    aircraft_type: str
    aircraft_reg: Optional[str] = None

class TaskResult(BaseModel):
    task_number: str
    task_description: str
    total_mhs: float
    status: str
    task_type: str
    similarity_score: float
    best_match_found: str
    matched_description: Optional[str]
    combined_from: List[str]
    # Removed age_difference and matched_age as requested

class PredictResponse(BaseModel):
    results: List[TaskResult]
    processing_time_ms: int
    embedding_time_ms: int
    task_processing_time_ms: int
    total_tasks: int
    available_tasks: int
    total_available_mhs: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    special_tasks_loaded: bool
    ad_sb_loaded: bool
    combination_recipes_loaded: bool
    precomputed_matrices: int
    aircraft_types: List[str]
    total_special_tasks: int
    total_ad_sb_tasks: int
    total_recipes: int
    cache_info: dict