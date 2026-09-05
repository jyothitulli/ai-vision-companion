import * as Speech from "expo-speech";

export interface SpeechSynthesizer {
  speak(text: string): Promise<void>;
  stop(): void;
}

export class PlatformTTS implements SpeechSynthesizer {
  speak(text: string): Promise<void> {
    return new Promise((resolve) => {
      if (!text.trim()) {
        resolve();
        return;
      }
      Speech.stop();
      Speech.speak(text, {
        rate: 0.92,
        pitch: 1.0,
        language: "en-US",
        onDone: () => resolve(),
        onStopped: () => resolve(),
        onError: () => resolve(),
      });
    });
  }

  stop(): void {
    Speech.stop();
  }
}

const defaultEngine: SpeechSynthesizer = new PlatformTTS();

export function speak(text: string): Promise<void> {
  return defaultEngine.speak(text);
}

export function stopSpeaking(): void {
  defaultEngine.stop();
}
