from pydantic import BaseModel
from typing import List

class HealthRecord(BaseModel):
    recordDate: str
    fatigueLevel: int
    dizzinessLevel: int
    mood: str
    sleepHours: float

class PredictRequest(BaseModel):
    userId: int
    records: List[HealthRecord]

class SideEffectTrainRecord(BaseModel):
    userId: int
    medicineId: int
    fatigueLevel: int
    dizzinessLevel: int
    mood: str
    sleepHours: float
    sideEffectOccurred: bool
    sideEffects: List[str]

class SideEffectTrainRequest(BaseModel):
    data: List[SideEffectTrainRecord]

class SideEffectPredictRequest(BaseModel):
    userId: int
    medicineId: int
    medicineName: str 
    fatigueLevel: int
    dizzinessLevel: int
    moodEncoded: int
    sleepHours: float

class SideEffectPredictResponse(BaseModel):
    probabilities: dict
    feedback: str


class SideEffectBatchPredictRequest(BaseModel):
    data: List[SideEffectPredictRequest]

