export interface AccessibilityControl {
  id: string;
  label: string;
  hint: string;
  role: "button";
}

export const ACCESSIBILITY_BUTTONS = [
  {
    id: "start_listening",
    label: "START LISTENING",
    hint: "Start listening for voice commands such as look, find, or read",
    role: "button" as const,
  },
  {
    id: "stop_listening",
    label: "STOP LISTENING",
    hint: "Cancel active voice listening",
    role: "button" as const,
  },
  {
    id: "toggle_camera",
    label: "CAMERA TOGGLE",
    hint: "Toggle camera feed on or off to conserve battery",
    role: "button" as const,
  },
  {
    id: "capture_analyze",
    label: "CAPTURE & ANALYZE",
    hint: "Take a photo now and announce objects and surroundings",
    role: "button" as const,
  },
  {
    id: "stop_speaking",
    label: "STOP SPEAKING",
    hint: "Immediately silence the speech assistant",
    role: "button" as const,
  },
  {
    id: "repeat_response",
    label: "REPEAT RESPONSE",
    hint: "Replay the last spoken description or answer",
    role: "button" as const,
  },
  {
    id: "help",
    label: "HELP",
    hint: "Hear instructions and list of available voice commands",
    role: "button" as const,
  },
  {
    id: "settings",
    label: "SETTINGS",
    hint: "Adjust backend URL, voice settings, and view privacy info",
    role: "button" as const,
  },
] as const;

// Backward-compatibility export for legacy references
export const PRIMARY_CONTROLS = [
  { id: "look", label: "LOOK", hint: "Describe the current view" },
  { id: "ask", label: "ASK", hint: "Ask a question about the view" },
  { id: "assistance", label: "START ASSISTANCE", hint: "Toggle continuous assistance" },
  { id: "read", label: "READ", hint: "Read visible text" },
  { id: "find", label: "FIND OBJECT", hint: "Find a nearby object by voice" },
  { id: "voice", label: "VOICE", hint: "Start voice command" },
  { id: "settings", label: "SETTINGS", hint: "Open settings" },
] as const;
