export type AppStateName =
  | "IDLE"
  | "LISTENING"
  | "CAPTURING"
  | "PROCESSING"
  | "SPEAKING"
  | "CONTINUOUS_ASSISTANCE"
  | "ERROR"
  | "PERMISSION_REQUIRED";

export const stateAnnouncement: Record<AppStateName, string> = {
  IDLE: "Ready. Double tap a mode or say look, ask, read, find, or start assistance.",
  LISTENING: "Listening.",
  CAPTURING: "Capturing.",
  PROCESSING: "Processing.",
  SPEAKING: "Speaking.",
  CONTINUOUS_ASSISTANCE: "Continuous assistance is on.",
  ERROR: "Something went wrong.",
  PERMISSION_REQUIRED: "Camera or microphone permission is required.",
};
