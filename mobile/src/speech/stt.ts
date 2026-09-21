import { Audio } from "expo-av";

let activeRecording: Audio.Recording | null = null;
let recordingAbortController: { abort: () => void } | null = null;

export async function recordUtterance(durationMs = 3000): Promise<string | null> {
  // If a recording is already active, stop it first
  await cancelRecording();

  try {
    const permission = await Audio.requestPermissionsAsync();
    if (!permission.granted) {
      return null;
    }

    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
    });

    const recording = new Audio.Recording();
    activeRecording = recording;

    await recording.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
    await recording.startAsync();

    let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    await new Promise<void>((resolve) => {
      recordingAbortController = {
        abort: () => {
          cancelled = true;
          if (timeoutHandle) clearTimeout(timeoutHandle);
          resolve();
        },
      };
      timeoutHandle = setTimeout(() => {
        resolve();
      }, durationMs);
    });

    recordingAbortController = null;

    if (cancelled) {
      try {
        await recording.stopAndUnloadAsync();
      } catch {
        // ignore unload errors on cancel
      }
      activeRecording = null;
      return null;
    }

    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    activeRecording = null;
    return uri;
  } catch {
    activeRecording = null;
    recordingAbortController = null;
    return null;
  }
}

export async function cancelRecording(): Promise<void> {
  if (recordingAbortController) {
    recordingAbortController.abort();
    recordingAbortController = null;
  }
  if (activeRecording) {
    try {
      await activeRecording.stopAndUnloadAsync();
    } catch {
      // ignore errors on forced stop
    }
    activeRecording = null;
  }
}
