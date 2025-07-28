from pydantic import BaseModel
from typing import List

class TasksInput(BaseModel):
    tasks: List[str]

class MostProbableDefectsInput(BaseModel):
    aircraft_age: float
    aircraft_model: str
    check_category: List[str]
    customer_name: str
    customer_name_consideration: bool

class DefectInvestigatorInput(BaseModel):
    task_number: List[str]
    log_item_number: str
    defect_desc: str
    corrective_action: str
