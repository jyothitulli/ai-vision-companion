# Architecture

## Components

- **Mobile (Expo):** camera, microphone, platform TTS, large high-contrast controls, voice-first modes.
- **API (FastAPI):** request validation, rate limits, metadata persistence, orchestration.
- **Vision runtime:** detector, depth, tracker, spatial engine, path analyzer, hazard prioritizer, scene understander, OCR, STT.
- **Database:** PostgreSQL in production (`docker-compose`). SQLite fallback for local machines without Postgres.

## Request flow

```
Camera frame (JPEG, in memory)
  → POST /api/vision/{analyze,ask,read,find}
  → preprocess (max side 960)
  → ObjectDetector (YOLO11)
  → DepthEstimator (Depth Anything V2 Small)
  → ObjectTracker (ByteTrack-style)
  → SpatialReasoningEngine
  → PathAnalyzer + SceneUnderstander + EventPrioritizer
  → ResponseGenerator (templates; no LLM spatial invention)
  → JSON { answer, objects, events, confidence, latencies }
  → Mobile expo-speech TTS
```

Frames are not written to disk. SQL tables store questions, answers, latencies, and event metadata only.

## Replaceable interfaces

`ObjectDetector`, `DepthEstimator`, `ObjectTracker`, `OCRProvider`, `SpeechRecognizer`, `SpecializedDetector`.

## Offline / edge

The mobile app can later run a lightweight detector locally. The backend remains the place for heavier depth, OCR, and open-vocabulary find. No proprietary cloud LLM is required for spatial answers.
