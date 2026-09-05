# Models

| Role | Default | Notes |
| --- | --- | --- |
| Detection | `yolo11n.pt` | Ultralytics, COCO. Swap via `YOLO_MODEL`. |
| Open-vocab find | `yolov8s-worldv2.pt` | Used in FIND when enabled. |
| Depth | `depth-anything/Depth-Anything-V2-Small-hf` | Relative depth only. |
| Tracking | ByteTrack (Ultralytics + local IoU) | No claimed m/s. |
| OCR | PaddleOCR if importable, else RapidOCR ONNX | Same `OCRProvider` API. |
| STT | faster-whisper `tiny` | Replaceable `SpeechRecognizer`. |
| TTS | Expo Speech (platform) | Abstraction in `mobile/src/speech/tts.ts`. |
| Accessibility classes | interface + optional fine-tuned YOLO | Not pretended to work from COCO. |

Weights download on first use into Ultralytics / Hugging Face caches. Optional copies can be placed in `backend/models/weights/`.
