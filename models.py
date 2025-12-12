from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date
from enum import Enum

class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class Chronotype(str, Enum):
    MORNING = "morning"
    NIGHT = "night"
    FLEXIBLE = "flexible"

class StudyPlanRequest(BaseModel):
    subjects: List[str] = Field(..., min_items=1, max_items=10, example=["Math", "Physics"])
    subject_difficulty: List[Difficulty] = Field(..., example=["Hard", "Medium"])
    daily_hours_available: float = Field(..., ge=0.5, le=12.0, example=3.0)
    focus_duration: int = Field(default=45, ge=20, le=90, example=45)
    break_duration: int = Field(default=10, ge=5, le=30, example=10)
    stress_tolerance: float = Field(default=0.7, ge=0.1, le=1.0, example=0.7)
    goal: str = Field(..., example="Prepare for final exams")
    deadline: date = Field(..., example="2024-12-31")
    chronotype: Chronotype = Field(default=Chronotype.FLEXIBLE)
    
    @validator('subject_difficulty')
    def validate_difficulty_length(cls, v, values):
        if 'subjects' in values and len(v) != len(values['subjects']):
            raise ValueError('Subject difficulty must match subjects count')
        return v

class StudySession(BaseModel):
    session_number: int
    subject: str
    difficulty: Difficulty
    start_time: str
    end_time: str
    duration_minutes: int
    break_minutes: int
    study_method: str

class DailyPlan(BaseModel):
    date: str
    day_name: str
    sessions: List[StudySession]
    total_study_minutes: int
    total_break_minutes: int

class StudyPlanResponse(BaseModel):
    success: bool
    message: str
    plan_id: str
    total_days: int
    total_hours: float
    daily_plans: List[DailyPlan]
    generated_at: str