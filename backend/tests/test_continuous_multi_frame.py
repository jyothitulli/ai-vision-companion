"""Multi-frame continuous assistance, tracking, deduplication, and movement test suite."""

from __future__ import annotations

import time
import pytest

from app.vision.hazards.prioritizer import EventPrioritizer
from app.vision.interfaces import RawDetection
from app.vision.tracking.bytetrack import ByteTrackTracker
from app.vision.types import (
    BoundingBox,
    DistanceBand,
    HazardEvent,
    HorizontalPosition,
    Movement,
    PathAnalysis,
    PathRelevance,
    Priority,
    SceneObject,
)


def test_continuous_assistance_four_frame_sequence() -> None:
    """Test 4-frame approaching obstacle sequence with deduplication and state progression."""
    tracker = ByteTrackTracker(max_history=10, iou_threshold=0.3)
    prioritizer = EventPrioritizer(cooldown_s=5.0)

    # Simulated timestamps (1 second apart)
    t0 = 100.0

    # Frame 1: Person far, small box (3m away, left)
    det1 = [
        RawDetection(
            class_name="person",
            confidence=0.88,
            bbox_xyxy=(50.0, 100.0, 150.0, 300.0),
        )
    ]
    objs1 = tracker.update(det1, t0)
    assert len(objs1) == 1
    track_id = objs1[0].id

    # Enrich object 1
    objs1[0] = objs1[0].model_copy(
        update={
            "position": HorizontalPosition.LEFT,
            "distance_band": DistanceBand.MID,
            "distance_range": "two to three meters",
            "path_relevance": PathRelevance.LOW,
            "movement": Movement.STATIONARY,
        }
    )
    events1 = prioritizer.prioritize(objs1, PathAnalysis(path_status="clear", obstacles=[]), now=t0)
    announceable1 = [e for e in events1 if e.should_announce]
    assert len(announceable1) == 1
    assert "person" in announceable1[0].message.lower()

    # Frame 2: Same person, same distance (3m away), 1s later -> DEDUPLICATED (cooldown active)
    t1 = t0 + 1.0
    det2 = [
        RawDetection(
            class_name="person",
            confidence=0.89,
            bbox_xyxy=(50.0, 100.0, 150.0, 300.0),
        )
    ]
    objs2 = tracker.update(det2, t1)
    assert objs2[0].id == track_id  # Track maintained
    objs2[0] = objs2[0].model_copy(
        update={
            "position": HorizontalPosition.LEFT,
            "distance_band": DistanceBand.MID,
            "distance_range": "two to three meters",
            "path_relevance": PathRelevance.LOW,
            "movement": Movement.STATIONARY,
        }
    )
    events2 = prioritizer.prioritize(objs2, PathAnalysis(path_status="clear", obstacles=[]), now=t1)
    announceable2 = [e for e in events2 if e.should_announce]
    # Person should NOT be re-announced within cooldown
    assert len(announceable2) == 0

    # Frame 3: Same person, moving closer (growing box: 2m away), approaching detected
    t2 = t0 + 2.0
    det3 = [
        RawDetection(
            class_name="person",
            confidence=0.91,
            bbox_xyxy=(45.0, 80.0, 175.0, 350.0),  # Box grew by ~50%
        )
    ]
    objs3 = tracker.update(det3, t2)
    assert objs3[0].id == track_id
    mvt, direction = tracker.movement_for(track_id)
    assert mvt == Movement.APPROACHING

    objs3[0] = objs3[0].model_copy(
        update={
            "position": HorizontalPosition.LEFT,
            "distance_band": DistanceBand.NEAR,
            "distance_range": "one to two meters",
            "path_relevance": PathRelevance.MEDIUM,
            "movement": Movement.APPROACHING,
        }
    )
    # Since movement changed to approaching, check event generation
    events3 = prioritizer.prioritize(objs3, PathAnalysis(path_status="clear", obstacles=[]), now=t2)
    # Approaching person event has cooldown key nearby_person:{track_id}
    # Still within cooldown for basic notification, but movement is correctly stored
    assert objs3[0].movement == Movement.APPROACHING

    # Frame 4: Obstacle reaches VERY_NEAR (close obstacle escalation)
    t3 = t0 + 3.0
    det4 = [
        RawDetection(
            class_name="person",
            confidence=0.95,
            bbox_xyxy=(30.0, 50.0, 220.0, 420.0),  # Substantially larger
        )
    ]
    objs4 = tracker.update(det4, t3)
    objs4[0] = objs4[0].model_copy(
        update={
            "position": HorizontalPosition.LEFT,
            "distance_band": DistanceBand.VERY_NEAR,
            "distance_range": "less than one meter",
            "path_relevance": PathRelevance.HIGH,
            "movement": Movement.APPROACHING,
        }
    )
    events4 = prioritizer.prioritize(objs4, PathAnalysis(path_status="partially_blocked", obstacles=[]), now=t3)
    announceable4 = [e for e in events4 if e.should_announce]
    # Escalated to close_obstacle P0 hazard, which has a distinct cooldown key
    assert len(announceable4) >= 1
    assert any("close" in e.message.lower() for e in announceable4)
