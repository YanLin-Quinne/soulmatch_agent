"""Data schema definitions for OkCupid profiles"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class OkCupidProfile(BaseModel):
    """OkCupid profile data model"""
    
    # Basic Info
    age: Optional[int] = None
    status: Optional[str] = None  # single, seeing someone, married, etc.
    sex: Optional[str] = None  # m/f
    orientation: Optional[str] = None  # straight, gay, bisexual
    body_type: Optional[str] = None
    
    # Lifestyle
    diet: Optional[str] = None
    drinks: Optional[str] = None
    drugs: Optional[str] = None
    smokes: Optional[str] = None
    
    # Background
    education: Optional[str] = None
    job: Optional[str] = None
    income: Optional[int] = None
    religion: Optional[str] = None
    ethnicity: Optional[str] = None
    
    # Personality
    sign: Optional[str] = None  # zodiac sign
    speaks: Optional[str] = None  # languages
    location: Optional[str] = None
    
    # Essays (0-9)
    essay0: Optional[str] = None  # self summary
    essay1: Optional[str] = None  # what I'm doing with my life
    essay2: Optional[str] = None  # I'm really good at
    essay3: Optional[str] = None  # first thing people notice
    essay4: Optional[str] = None  # favorite books, movies, shows, music, food
    essay5: Optional[str] = None  # six things I could never do without
    essay6: Optional[str] = None  # I spend a lot of time thinking about
    essay7: Optional[str] = None  # on a typical Friday night
    essay8: Optional[str] = None  # most private thing I'm willing to admit
    essay9: Optional[str] = None  # you should message me if


class ExtractedFeatures(BaseModel):
    """Features extracted from profile using LLM"""
    
    # Communication style
    communication_style: Literal[
        "direct", "indirect", "humorous", "serious", "casual", "formal"
    ] = "casual"
    communication_confidence: float = Field(ge=0.0, le=1.0)
    
    # Values
    core_values: list[str] = Field(default_factory=list)  # e.g., ["family", "career", "adventure"]
    values_confidence: float = Field(ge=0.0, le=1.0)
    
    # Interests
    interest_categories: dict[str, float] = Field(default_factory=dict)
    # e.g., {"music": 0.8, "sports": 0.3, "travel": 0.9}
    
    # Personality traits (Big Five)
    openness: Optional[float] = Field(None, ge=0.0, le=1.0)
    conscientiousness: Optional[float] = Field(None, ge=0.0, le=1.0)
    extraversion: Optional[float] = Field(None, ge=0.0, le=1.0)
    agreeableness: Optional[float] = Field(None, ge=0.0, le=1.0)
    neuroticism: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Relationship goals
    relationship_goals: str = "casual"  # casual, serious, friendship, unsure
    goals_confidence: float = Field(ge=0.0, le=1.0)
    
    # Overall summary
    personality_summary: str = ""


class PersonaProfile(BaseModel):
    """Complete persona profile for an agent"""
    
    # Original OkCupid data
    original_profile: OkCupidProfile
    
    # Extracted features
    features: ExtractedFeatures
    
    # Metadata
    profile_id: str
    is_bot: bool = True
    
    # System prompt for role-playing
    system_prompt: str = ""
    
    # Feature vector for matching (normalized)
    feature_vector: Optional[list[float]] = None
