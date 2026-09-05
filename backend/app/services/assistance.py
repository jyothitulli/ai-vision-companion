from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.reasoning.nlg import Intent, ResponseGenerator
from app.vision.types import PipelineResult, Priority


@dataclass
class AssistanceState:
    session_id: str
    last_announcement: str | None = None
    last_path_status: str | None = None
    frames: int = 0
    last_emit_s: float = 0.0
    started_s: float = field(default_factory=time.monotonic)


class AssistanceManager:
    def __init__(self) -> None:
        self._sessions: dict[str, AssistanceState] = {}
        self._nlg = ResponseGenerator()

    def start(self, session_id: str) -> AssistanceState:
        state = AssistanceState(session_id=session_id)
        self._sessions[session_id] = state
        return state

    def stop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def is_active(self, session_id: str) -> bool:
        return session_id in self._sessions

    def ingest(self, session_id: str, result: PipelineResult, intent: Intent) -> str | None:
        state = self._sessions.get(session_id)
        if state is None:
            state = self.start(session_id)
        state.frames += 1
        announcement = self._nlg.assistance_announcement(result)
        if announcement is None:
            if result.path.path_status == "clear" and state.last_path_status != "clear":
                state.last_path_status = "clear"
                text = "Your path appears relatively clear in this view. I cannot guarantee it is safe to walk."
                state.last_announcement = text
                state.last_emit_s = time.monotonic()
                return text
            return None
        # Never repeat the same sentence back-to-back.
        if announcement == state.last_announcement:
            return None
        p0 = any(event.priority == Priority.P0 and event.should_announce for event in result.events)
        now = time.monotonic()
        if not p0 and now - state.last_emit_s < 4:
            return None
        state.last_announcement = announcement
        state.last_path_status = result.path.path_status
        state.last_emit_s = now
        return announcement


assistance_manager = AssistanceManager()
