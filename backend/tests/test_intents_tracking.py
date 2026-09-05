from app.reasoning.nlg import IntentParser
from app.vision.tracking.bytetrack import ByteTrackTracker
from app.vision.interfaces import RawDetection
from app.vision.types import Movement


def test_intent_modes() -> None:
    parser = IntentParser()
    assert parser.parse("Look.").mode == "look"
    assert parser.parse("Where is the door?").mode == "ask"
    assert parser.parse("Find my keys").mode == "find"
    assert parser.parse("Find my keys").target == "keys"
    assert parser.parse("Read this").mode == "read"
    assert parser.parse("Start assistance").mode == "assistance_start"


def test_tracker_approaching_by_growing_bbox() -> None:
    tracker = ByteTrackTracker()
    first = RawDetection("person", 0.9, (100, 100, 140, 220), track_id=1)
    second = RawDetection("person", 0.9, (90, 80, 170, 280), track_id=1)
    tracker.update([first], 1.0)
    tracker.update([second], 1.4)
    movement, direction = tracker.movement_for(1)
    assert movement == Movement.APPROACHING
    assert direction == "toward_user"


def test_tracker_assigns_ids_when_missing() -> None:
    tracker = ByteTrackTracker()
    det = RawDetection("chair", 0.8, (10, 10, 40, 50))
    tracker.update([det], 0.0)
    assert det.track_id is not None
    later = RawDetection("chair", 0.82, (12, 12, 42, 52))
    tracker.update([later], 0.2)
    assert later.track_id == det.track_id
