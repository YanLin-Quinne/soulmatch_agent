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


# ═══════════════════════════════════════════════════════════════════
# Extended Features (v2.0)
# ═══════════════════════════════════════════════════════════════════

class MBTIType(str):
    """MBTI 16 personality types"""
    pass  # ISTJ, ISFJ, INFJ, INTJ, ISTP, ISFP, INFP, INTP, ESTP, ESFP, ENFP, ENTP, ESTJ, ESFJ, ENFJ, ENTJ

class AttachmentStyle(str):
    """Attachment theory styles"""
    pass  # secure, anxious, avoidant, disorganized

class LoveLanguage(str):
    """Five love languages"""
    pass  # words_of_affirmation, quality_time, receiving_gifts, acts_of_service, physical_touch

class RelationshipStatus(str):
    """Relationship progression stages"""
    pass  # stranger, acquaintance, crush, dating, committed

class RelationshipType(str):
    """Type of relationship"""
    pass  # love, friendship, family, other


class ExtendedFeatures(BaseModel):
    """Extended 42-dimensional feature space (v2.0)"""

    # MBTI (6 dimensions)
    mbti_type: Optional[str] = None  # 16 types
    mbti_confidence: float = 0.0
    mbti_ei: float = 0.5  # Extraversion-Introversion axis
    mbti_sn: float = 0.5  # Sensing-Intuition axis
    mbti_tf: float = 0.5  # Thinking-Feeling axis
    mbti_jp: float = 0.5  # Judging-Perceiving axis

    # Attachment style (3 dimensions)
    attachment_style: Optional[str] = None  # secure/anxious/avoidant/disorganized
    attachment_anxiety: float = 0.5
    attachment_avoidance: float = 0.5

    # Love languages (2 dimensions)
    primary_love_language: Optional[str] = None
    secondary_love_language: Optional[str] = None

    # Trust trajectory (3 dimensions)
    trust_score: float = 0.5
    trust_velocity: float = 0.0  # change rate per turn
    trust_history: list[float] = Field(default_factory=list)

    # Relationship state labels (4 dimensions)
    relationship_status: str = "stranger"
    relationship_type: str = "other"
    sentiment_label: str = "neutral"
    can_advance: Optional[bool] = None


class RelationshipSnapshot(BaseModel):
    """Snapshot of relationship state at a specific turn"""
    turn: int
    sentiment: str
    rel_type: str
    rel_status: str
    trust_score: float
    emotion_valence: float
    timestamp: Optional[str] = None


class RelationshipPredictionResult(BaseModel):
    """Output from RelationshipPredictionAgent"""

    # Core predictions
    sentiment: str  # positive/neutral/negative
    sentiment_confidence: float
    rel_type: str  # love/friendship/family/other
    rel_type_probs: dict[str, float] = Field(default_factory=dict)
    rel_status: str  # stranger→acquaintance→crush→dating→committed
    rel_status_probs: dict[str, float] = Field(default_factory=dict)

    # Conformal prediction: can advance?
    can_advance: bool
    advance_prediction_set: list[str] = Field(default_factory=list)
    advance_coverage_guarantee: float = 0.9

    # Time-series prediction t+1
    next_status_prediction: str = ""
    next_status_probs: dict[str, float] = Field(default_factory=dict)

    # Milestone report (turn 10/30 only)
    milestone_report: Optional[dict] = None

    # Internal state
    snapshot: Optional[RelationshipSnapshot] = None
    reasoning_trace: str = ""
