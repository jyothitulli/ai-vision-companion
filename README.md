# Around You — AI Vision Companion for Visually Impaired Users

> **Multimodal assistive perception with voice-first interaction and TalkBack accessibility.**

*Around You* is an assistive technology application designed to help visually impaired individuals understand their surroundings through real-time computer vision, spatial reasoning, and voice-first interaction.

> [!WARNING]
> **Safety Notice**: *Around You* is an assistive perception aid, not a collision-avoidance or primary navigation system. It does not replace a white cane or guide dog, and never instructs users to cross roadways or assume clear passage.

---

## Key Capabilities

- **Voice-First Experience**: Starts with an immediate spoken greeting (*"Vision assistant ready. How can I help?"*) and supports hands-free natural voice commands.
- **Spatial Object Awareness**: Uses YOLO detection and spatial reasoning to report object locations (left, right, center, close, mid, far) without hallucinating false precision.
- **Target Item Localization**: Locates everyday personal items (*"Where is my water bottle?"* $\to$ *"A bottle is slightly to your right, about 1 meter away"*).
- **OCR Text Reading**: Extracts and speaks text from signage, labels, and documents.
- **Continuous Assistance Mode**: Periodically monitors the scene and only speaks when significant changes or new obstacles appear.
- **Full TalkBack Accessibility**: High-contrast theme, minimum 56dp touch targets, semantic accessibility labels, hints, and live regions.

---

## Architecture Overview

The system is organized into a modular mobile frontend and a high-performance vision backend:

```
├── mobile/                  # React Native (Expo) accessibility-first mobile app
│   ├── App.tsx             # Main screen with voice-first lifecycle and TalkBack support
│   ├── src/
│   │   ├── api/client.ts   # Resilient HTTP API client with timeout guards
│   │   ├── screens/        # HelpScreen and SettingsScreen
│   │   ├── speech/         # commands.ts (voice parser), stt.ts, tts.ts
│   │   ├── state/          # App state definitions & accessibility announcements
│   │   ├── theme.ts        # High-contrast theme (#0B0F14, #F5C542, #F4F7FB)
│   │   ├── ui/modes.ts     # Definitions for the 8 accessible control buttons
│   │   └── __tests__/      # Automated test suites for commands, accessibility, & api
├── backend/                # FastAPI computer vision server
│   ├── app/
│   │   ├── api/            # Endpoints: /api/vision/*, /api/speech/*, /api/assistance/*
│   │   ├── vision/         # YOLO11n detector, spatial reasoning, OCR, scene analysis
│   │   └── services/       # Model warmup and runtime management
│   └── tests/              # Comprehensive pytest test suite (58 tests)
```

---

## Voice Commands Reference

The assistant understands natural spoken requests:

| Spoken Command | Action | Example Utterances |
|---|---|---|
| **Look / Scene** | Captures image and describes objects and layout | *"Look"*, *"What is in front of me?"*, *"Describe surroundings"* |
| **Find Item** | Finds a specific target with spatial direction | *"Find my keys"*, *"Where is the bottle?"*, *"Locate chair"* |
| **Read Text** | Performs OCR on signage, menus, or labels | *"Read this"*, *"Read text"*, *"What does this say?"* |
| **Repeat** | Replays the last spoken answer without re-analyzing | *"Repeat"*, *"Repeat that"*, *"Say that again"* |
| **Stop** | Instantly silences speech and cancels actions | *"Stop"*, *"Cancel"*, *"Be quiet"*, *"Pause"* |
| **Continuous Assistance** | Toggles continuous background scene monitoring | *"Start assistance"*, *"Stop assistance"* |
| **Camera Control** | Turns the camera preview on or off | *"Open camera"*, *"Close camera"*, *"Turn off camera"* |
| **Help** | Speaks available commands and opens help screen | *"Help"*, *"What can you do?"*, *"Commands"* |
| **Settings / Back** | Opens settings or returns to main view | *"Settings"*, *"Go back"*, *"Close"* |

---

## Accessible Touch Controls

For users navigating via Android TalkBack or iOS VoiceOver, 8 high-contrast, large-target buttons (min 56dp height) are available:

1. **START LISTENING / STOP LISTENING**: Toggles voice recording for spoken commands.
2. **CAPTURE & ANALYZE**: Single-action snapshot and scene description.
3. **REPEAT RESPONSE**: Instantly replays the previous spoken response.
4. **STOP SPEAKING**: Silences active text-to-speech synthesis immediately.
5. **CAMERA TOGGLE**: Enables or disables the camera stream to save battery.
6. **START / STOP ASSISTANCE**: Toggles hands-free continuous change detection.
7. **HELP**: Opens the spoken reference and accessible guide.
8. **SETTINGS**: Configures backend endpoint, speech rate, and privacy preferences.

---

## Getting Started

### 1. Backend Server Setup

Requirements: Python 3.11+, PyTorch (CPU or CUDA).

```bash
# Clone the repository
git clone https://github.com/jyothitulli/ai-vision-companion.git
cd ai-vision-companion

# Set up Python virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

# Run the backend API server
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Mobile App Setup (React Native Expo)

Requirements: Node.js 18+, npm.

```bash
cd mobile

# Install dependencies
npm install

# Start the Expo development server
npm start

# Run on Android emulator or connected device
npm run android
```

To connect a physical smartphone to your local backend, set the environment variable:
```bash
# In mobile/.env or before starting Expo:
EXPO_PUBLIC_API_URL=http://<YOUR_COMPUTER_LAN_IP>:8000
```
Or use the live deployed production backend:
```bash
EXPO_PUBLIC_API_URL=https://ai-vision-companion.onrender.com
```

---

## Testing & Quality Assurance

### Mobile Test Suite
Run Jest tests for command parsing, accessibility labels, and API resilience:
```bash
cd mobile
npm test
```
Verify TypeScript types:
```bash
cd mobile
npm run typecheck
```

### Backend Test Suite
Run full test coverage across spatial reasoning, vision pipelines, safety checks, and API routes:
```bash
pytest backend/tests
```

---

## Privacy & Ethical AI

- **No Remote Frame Storage**: Captured camera frames are processed in memory and immediately discarded.
- **Grounded Spatial Facts**: Directional advice (*"to your left"*, *"in front"*) is derived strictly from bounding box geometry and depth bands, never generated arbitrarily by generative language models.
- **Offline First Considerations**: TTS uses on-device system speech synthesis (`expo-speech`) to minimize latency and data transmission.
