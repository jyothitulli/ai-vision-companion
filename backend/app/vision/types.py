from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class HorizontalPosition(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class DistanceBand(str, Enum):
    VERY_NEAR = "very_near"
    NEAR = "near"
    MID = "mid"
    FAR = "far"
    UNKNOWN = "unknown"


class Movement(str, Enum):
    STATIONARY = "stationary"
    MOVING = "moving"
    APPROACHING = "approaching"
    RECEDING = "receding"
    UNKNOWN = "unknown"


class PathRelevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class BoundingBox(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(ge=0, le=1)
    height: float = Field(ge=0, le=1)


class SceneObject(BaseModel):
    id: int
    type: str
    confidence: float
    bbox: BoundingBox
    position: HorizontalPosition = HorizontalPosition.CENTER
    distance_estimate: Optional[float] = None
    distance_range: str = "unknown"
    distance_band: DistanceBand = DistanceBand.UNKNOWN
    movement: Movement = Movement.UNKNOWN
    movement_direction: Optional[str] = None
    path_relevance: PathRelevance = PathRelevance.NONE
    specialized: bool = False
    track_id: Optional[int] = None
    relationships: list[str] = Field(default_factory=list)


class PathObstacle(BaseModel):
    object_id: int
    type: str
    distance_range: str
    position: HorizontalPosition
    confidence: float


class PathAnalysis(BaseModel):
    path_status: Literal["clear", "partially_blocked", "blocked", "uncertain"]
    obstacles: list[PathObstacle] = Field(default_factory=list)
    notes: str = ""


class SceneUnderstanding(BaseModel):
    scene_type: str
    description: str
    confidence: float
    uncertain: bool = False


class HazardEvent(BaseModel):
    event_id: str
    type: str
    object_id: Optional[int] = None
    priority: Priority
    confidence: float
    message: str
    cooldown_key: str
    should_announce: bool = True


class PipelineResult(BaseModel):
    objects: list[SceneObject]
    path: PathAnalysis
    scene: SceneUnderstanding
    events: list[HazardEvent]
    answer: str
    confidence: float
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    ocr_text: Optional[str] = None
