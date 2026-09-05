# AI pipeline

1. Frame preprocessing: downscale long side to 960, keep aspect ratio.
2. Object detection: Ultralytics YOLO11n (COCO).
3. Specialized proxies: traffic light → traffic_signal (low confidence cap). Stairs/curbs are not claimed from COCO.
4. Depth: Depth Anything V2 Small relative map. Median sampled inside each box, scene-normalized, mapped to coarse bands.
5. Tracking: ByteTrack IDs when Ultralytics provides them; IoU fallback otherwise. Area growth ⇒ approaching (uncalibrated).
6. Spatial reasoning: left/center/right from bbox center; distance bands from relative depth; path relevance from center corridor × nearness.
7. Path analysis: obstruction classes in the forward corridor.
8. Hazards: P0–P3, cooldown, dedup.
9. Scene: conservative indoor/road/hallway heuristics. Low evidence ⇒ "I cannot confidently determine the scene."
10. Intent + NLG: templates. Confidence hedging. Crossing language never says "cross now" or "safe."

Continuous assistance analyzes periodic frames and speaks only cooldown-eligible P0/P1 (and selected P2) events.
