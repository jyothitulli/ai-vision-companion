from app.reasoning.nlg import Intent
from app.services.assistance import AssistanceManager
from app.vision.types import (
    HazardEvent,
    PathAnalysis,
    PipelineResult,
    Priority,
    SceneUnderstanding,
)


def _result(message: str, announce: bool = True) -> PipelineResult:
    return PipelineResult(
        objects=[],
        path=PathAnalysis(path_status="partially_blocked"),
        scene=SceneUnderstanding(
            scene_type="unknown",
            description="I cannot confidently determine the scene.",
            confidence=0.2,
            uncertain=True,
        ),
        events=[
            HazardEvent(
                event_id="e1",
                type="path_obstruction",
                priority=Priority.P1,
                confidence=0.8,
                message=message,
                cooldown_key="path_obstruction:chair",
                should_announce=announce,
            )
        ],
        answer=message,
        confidence=0.8,
    )


def test_assistance_does_not_repeat_same_sentence() -> None:
    manager = AssistanceManager()
    manager.start("s")
    intent = Intent(mode="look", question="continuous")
    first = manager.ingest("s", _result("A chair is approximately one to two meters ahead and may obstruct your path."), intent)
    second = manager.ingest("s", _result("A chair is approximately one to two meters ahead and may obstruct your path."), intent)
    assert first is not None
    assert second is None
