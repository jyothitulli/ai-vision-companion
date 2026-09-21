export type ParsedCommand =
  | { type: "LOOK" }
  | { type: "FIND"; target: string }
  | { type: "READ"; question?: string }
  | { type: "START_ASSISTANCE" }
  | { type: "STOP_ASSISTANCE" }
  | { type: "REPEAT" }
  | { type: "STOP" }
  | { type: "HELP" }
  | { type: "OPEN_CAMERA" }
  | { type: "CLOSE_CAMERA" }
  | { type: "SETTINGS" }
  | { type: "BACK" }
  | { type: "ASK"; question: string }
  | { type: "UNKNOWN"; raw: string };

/**
 * Parses a transcribed voice utterance into an actionable accessibility command.
 */
export function parseVoiceCommand(rawUtterance: string): ParsedCommand {
  const clean = rawUtterance.trim().toLowerCase();
  if (!clean) {
    return { type: "UNKNOWN", raw: "" };
  }

  // Stop / Cancel commands (highest priority)
  if (
    clean === "stop" ||
    clean === "cancel" ||
    clean === "be quiet" ||
    clean === "shut up" ||
    clean === "pause" ||
    clean.startsWith("stop speaking") ||
    clean === "quiet"
  ) {
    return { type: "STOP" };
  }

  // Repeat previous answer
  if (
    clean === "repeat" ||
    clean === "repeat that" ||
    clean === "repeat response" ||
    clean === "say that again" ||
    clean === "say again" ||
    clean === "what did you say"
  ) {
    return { type: "REPEAT" };
  }

  // Help command
  if (
    clean === "help" ||
    clean === "commands" ||
    clean === "what can you do" ||
    clean === "show help" ||
    clean === "how do i use this"
  ) {
    return { type: "HELP" };
  }

  // Camera toggles
  if (
    clean === "open camera" ||
    clean === "turn on camera" ||
    clean === "start camera" ||
    clean === "camera on"
  ) {
    return { type: "OPEN_CAMERA" };
  }
  if (
    clean === "close camera" ||
    clean === "turn off camera" ||
    clean === "stop camera" ||
    clean === "camera off"
  ) {
    return { type: "CLOSE_CAMERA" };
  }

  // Settings & Navigation
  if (clean === "settings" || clean === "open settings") {
    return { type: "SETTINGS" };
  }
  if (clean === "go back" || clean === "back" || clean === "close" || clean === "exit") {
    return { type: "BACK" };
  }

  // Continuous assistance
  if (
    clean.includes("start assistance") ||
    clean.includes("enable assistance") ||
    clean.includes("turn on assistance") ||
    clean === "continuous assistance"
  ) {
    return { type: "START_ASSISTANCE" };
  }
  if (
    clean.includes("stop assistance") ||
    clean.includes("disable assistance") ||
    clean.includes("turn off assistance")
  ) {
    return { type: "STOP_ASSISTANCE" };
  }

  // Reading / OCR
  if (
    clean === "read" ||
    clean === "read this" ||
    clean === "read text" ||
    clean.startsWith("read the text") ||
    clean.startsWith("what does this say") ||
    clean.startsWith("what does it say")
  ) {
    return { type: "READ", question: clean };
  }

  // Finding objects
  if (
    clean.startsWith("find ") ||
    clean.startsWith("where is ") ||
    clean.startsWith("where's ") ||
    clean.startsWith("locate ") ||
    clean.startsWith("search for ")
  ) {
    let target = clean
      .replace(/^find\s+(my\s+|the\s+|a\s+)?/, "")
      .replace(/^where\s+is\s+(my\s+|the\s+|a\s+)?/, "")
      .replace(/^where's\s+(my\s+|the\s+|a\s+)?/, "")
      .replace(/^locate\s+(my\s+|the\s+|a\s+)?/, "")
      .replace(/^search\s+for\s+(my\s+|the\s+|a\s+)?/, "")
      .trim();

    target = target.replace(/[.?]+$/, "");
    return { type: "FIND", target: target || "object" };
  }

  // Look / Scene description
  if (
    clean === "look" ||
    clean === "what is in front of me" ||
    clean === "what is in front of me?" ||
    clean === "what's in front of me" ||
    clean === "what's around" ||
    clean === "what is around me" ||
    clean === "what is around me?" ||
    clean === "describe surroundings" ||
    clean === "describe my surroundings" ||
    clean === "describe the scene" ||
    clean === "describe the room" ||
    clean === "what do you see" ||
    clean === "what do you see?"
  ) {
    return { type: "LOOK" };
  }

  // If none of the specific verbs match, treat as a general scene question
  return { type: "ASK", question: rawUtterance };
}
