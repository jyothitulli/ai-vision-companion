# API

Base URL: `http://localhost:8000`

Optional header: `X-API-Key` (required when `APP_ENV` is not development with the default key).

| Method | Path | Body |
| --- | --- | --- |
| GET | `/api/health` | |
| POST | `/api/vision/analyze` | multipart `file` |
| POST | `/api/vision/ask` | `file`, `question` |
| POST | `/api/vision/read` | `file`, `question` |
| POST | `/api/vision/find` | `file`, `target` |
| POST | `/api/assistance/start` | `session_id` |
| POST | `/api/assistance/stop` | `session_id` |
| POST | `/api/assistance/frame` | `file`, `session_id` |
| POST | `/api/speech/transcribe` | `file` (audio) |

Image limit: `MAX_UPLOAD_BYTES` (default 8 MiB). Types: JPEG, PNG, WebP.

Example success:

```json
{
  "success": true,
  "mode": "look",
  "answer": "A chair is approximately one to two meters ahead and may obstruct your path.",
  "objects": [],
  "events": [],
  "confidence": 0.89,
  "processing_time_ms": 423
}
```

User-facing errors never include stack traces.
