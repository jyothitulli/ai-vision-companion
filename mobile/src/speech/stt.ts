import { Audio } from "expo-av";

export async function recordUtterance(): Promise<string | null> {
  const permission = await Audio.requestPermissionsAsync();
  if (!permission.granted) {
    return null;
  }
  await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
  const recording = new Audio.Recording();
  await recording.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
  await recording.startAsync();
  await new Promise((resolve) => setTimeout(resolve, 2800));
  await recording.stopAndUnloadAsync();
  return recording.getURI();
}
