# Vision Companion

AI-powered multimodal assistive perception for visually impaired people.

This is not a guaranteed navigation or collision-avoidance system. It never tells a user to cross a road or that walking forward is safe.

## Problem

A visually impaired person cannot easily understand the visual environment. Vision Companion turns a smartphone camera plus speech into spoken spatial descriptions:

- What is in front of me?
- Where is the door?
- Is anyone near me?
- Is the path clear?
- Read this.
- Find my bottle.

## Solution

On-device capture (camera, mic, TTS) plus a Python vision backend:

Camera → preprocess → YOLO detection → Depth Anything V2 → tracking → spatial reasoning → scene/path/hazards → spoken answer

Spatial facts are produced by computer vision modules, not by a cloud LLM inventing locations.

## Architecture

See `docs/architecture.md` and `docs/ai-pipeline.md`.

```
mobile/     React Native (Expo) + TypeScript
backend/   FastAPI + YOLO11 + Depth Anything V2 + Whisper + OCR
datasets/   Fine-tuning data layout (no personal photos committed)
training/   Train / failure-collection scripts
evaluation/ Metrics report generator
```

## Environment notes (this machine)

- Empty repository at start of implementation
- Node.js 22 available
- No system Python; portable CPython 3.12.8 lives in `.tools/python/`
- No NVIDIA GPU detected (`nvidia-smi` missing) — models default to CPU
- Git was not installed at setup time
- PostgreSQL is the production database. Local default falls back to SQLite if `DATABASE_URL` is unset

## Setup

1. Copy `.env.example` to `.env`
2. Create a virtualenv with the portable interpreter:

```powershell
.\.tools\python\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

3. Run the API:

```powershell
cd backend
$env:PYTHONPATH = "."
..\..\..\project\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

From `C:\Users\mahal\project`:

```powershell
$env:PYTHONPATH = "C:\Users\mahal\project\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

4. Mobile:

```powershell
cd mobile
npm install
npx expo start
```

Set `EXPO_PUBLIC_API_URL` to your PC's LAN IP when using a physical phone, e.g. `http://192.168.1.10:8000`.

First vision request downloads YOLO / depth / Whisper weights. That can take several minutes on CPU.

## Testing

```powershell
$env:PYTHONPATH = "C:\Users\mahal\project\backend"
$env:ENABLE_MODEL_WARMUP = "false"
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

Mobile: `cd mobile; npm test` after `npm install`.

## Models

Documented in `docs/models.md`.

## Privacy

Camera frames are processed in memory and not stored by default. See `docs/privacy.md`.

## Limitations

- Monocular depth is relative, not metric. Spoken distances are approximate bands.
- Pretrained COCO YOLO does not reliably detect stairs, curbs, ramps, or tactile paving.
- CPU inference is slower than the eventual real-time target. Do not claim real-time until measured.
- PaddleOCR is the preferred OCR interface; RapidOCR is used if Paddle is unavailable on Windows.

## Future work

- Fine-tune accessibility classes
- Calibrate depth with stereo or a ranging sensor
- On-device YOLO for offline LOOK
- Piper TTS option
