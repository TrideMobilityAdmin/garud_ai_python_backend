from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Task(BaseModel):
    task_number: str
    task_description: str

class MaintenanceTasksInput(BaseModel):
    tasks: List[Task]
    aircraft_type: str
    aircraft_reg: Optional[str] = None

class MHS(BaseModel):
    max: float
    min: float  
    avg: float
    est: float

class SparePartDetails(BaseModel):
    partId: str
    desc: str
    qty: float
    price: float
    part_type: str
    prob: float

class TaskDetail(BaseModel):
    cluster: str
    description: str
    skill: List[str]
    mhs: MHS
    prob: float
    spare_parts: List[SparePartDetails]

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

class MaintenanceTaskResponse(BaseModel):
    results: List[TaskResult]
    processing_time_ms: int
    embedding_time_ms: int
    task_processing_time_ms: int
    total_tasks: int