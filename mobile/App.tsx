import { StatusBar } from "expo-status-bar";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  AccessibilityInfo,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { healthCheck, postSession, transcribe, uploadFrame } from "./src/api/client";
import { recordUtterance } from "./src/speech/stt";
import { speak, stopSpeaking } from "./src/speech/tts";
import { AppStateName, stateAnnouncement } from "./src/state/appState";
import { colors } from "./src/theme";
import { SettingsScreen } from "./src/screens/SettingsScreen";

const SESSION_ID = "mobile-default";

type Mode = "look" | "ask" | "read" | "find" | "assistance";

export default function App() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [appState, setAppState] = useState<AppStateName>("IDLE");
  const [statusLine, setStatusLine] = useState("Ready.");
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [assistanceOn, setAssistanceOn] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const assistanceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const assistanceActiveRef = useRef(false);
  const isAssistanceLoopRunning = useRef(false);
  const isLookRunning = useRef(false);

  useEffect(() => {
    healthCheck().then(setBackendUp);
  }, []);

  useEffect(() => {
    AccessibilityInfo.announceForAccessibility(stateAnnouncement[appState]);
  }, [appState]);

  const listeningLabel = useMemo(() => {
    if (backendUp === false) {
      return "BACKEND UNAVAILABLE";
    }
    if (!permission?.granted) {
      return "PERMISSION REQUIRED";
    }
    return appState.replaceAll("_", " ");
  }, [appState, backendUp, permission?.granted]);

  async function captureUri(): Promise<string | null> {
    if (!cameraRef.current) {
      return null;
    }
    setAppState("CAPTURING");
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.6, skipProcessing: true });
    return photo?.uri ?? null;
  }

  async function runMode(mode: Mode, extra: Record<string, string> = {}) {
    if (mode === "look") {
      if (isLookRunning.current) return;
      isLookRunning.current = true;
    }
    try {
      if (!permission?.granted) {
        setAppState("PERMISSION_REQUIRED");
        await speak("Camera permission is required.");
        return;
      }
      const uri = await captureUri();
      if (!uri) {
        setAppState("ERROR");
        await speak("I couldn't capture a photo.");
        return;
      }
      setAppState("PROCESSING");
      setStatusLine("Processing the scene.");
      const path =
        mode === "look"
          ? "/api/vision/analyze"
          : mode === "ask"
            ? "/api/vision/ask"
            : mode === "read"
              ? "/api/vision/read"
              : mode === "find"
                ? "/api/vision/find"
                : "/api/assistance/frame";
      const result = await uploadFrame(uri, path, { session_id: SESSION_ID, ...extra });
      const answer = result.answer || "I couldn't process the image. Please try again.";
      setStatusLine(answer);
      setAppState("SPEAKING");
      await speak(answer);
      setAppState(assistanceActiveRef.current ? "CONTINUOUS_ASSISTANCE" : "IDLE");
    } catch {
      setAppState("ERROR");
      setStatusLine("I couldn't process the image. Please try again.");
      await speak("I couldn't process the image. Please try again.");
      setAppState(assistanceActiveRef.current ? "CONTINUOUS_ASSISTANCE" : "IDLE");
    } finally {
      if (mode === "look") {
        isLookRunning.current = false;
      }
    }
  }

  async function voiceCommand() {
    setAppState("LISTENING");
    setStatusLine("Listening.");
    await speak("Listening.");
    const uri = await recordUtterance();
    if (!uri) {
      setAppState("PERMISSION_REQUIRED");
      await speak("Microphone permission is required.");
      return;
    }
    const text = (await transcribe(uri)).toLowerCase();
    if (!text.trim()) {
      await speak("I didn't catch that.");
      setAppState("IDLE");
      return;
    }
    if (text.includes("start assistance")) {
      await toggleAssistance(true);
      return;
    }
    if (text.includes("stop assistance")) {
      await toggleAssistance(false);
      return;
    }
    if (text.includes("read")) {
      await runMode("read", { question: text });
      return;
    }
    if (text.includes("find")) {
      const target = text.replace("find my", "").replace("find", "").trim() || "bottle";
      await runMode("find", { target });
      return;
    }
    if (text.includes("settings")) {
      setShowSettings(true);
      await speak("Settings.");
      setAppState("IDLE");
      return;
    }
    if (text.trim() === "look" || text.includes("what's around")) {
      await runMode("look");
      return;
    }
    await runMode("ask", { question: text });
  }

  async function findByVoice() {
    setAppState("LISTENING");
    await speak("What should I find?");
    const uri = await recordUtterance();
    if (!uri) {
      setAppState("PERMISSION_REQUIRED");
      await speak("Microphone permission is required.");
      return;
    }
    const text = (await transcribe(uri)).toLowerCase();
    const target = text.replace("find my", "").replace("find", "").trim() || "bottle";
    await runMode("find", { target });
  }

  async function startAssistanceLoop() {
    if (isAssistanceLoopRunning.current) return;
    isAssistanceLoopRunning.current = true;
    try {
      while (assistanceActiveRef.current) {
        if (!permission?.granted) break;
        await runMode("assistance");
        if (!assistanceActiveRef.current) break;
        await new Promise((resolve) => {
          assistanceTimer.current = setTimeout(resolve, 3500);
        });
        assistanceTimer.current = null;
      }
    } finally {
      isAssistanceLoopRunning.current = false;
      if (assistanceTimer.current) {
        clearTimeout(assistanceTimer.current);
        assistanceTimer.current = null;
      }
    }
  }

  async function toggleAssistance(on: boolean) {
    setAssistanceOn(on);
    assistanceActiveRef.current = on;
    if (assistanceTimer.current) {
      clearTimeout(assistanceTimer.current);
      assistanceTimer.current = null;
    }
    if (!on) {
      await postSession("/api/assistance/stop", SESSION_ID);
      setAppState("IDLE");
      await speak("Continuous assistance is off.");
      return;
    }
    await postSession("/api/assistance/start", SESSION_ID);
    setAppState("CONTINUOUS_ASSISTANCE");
    await speak("Continuous assistance is on. I will only announce important changes.");
    void startAssistanceLoop();
  }

  if (!permission) {
    return <View style={styles.screen} />;
  }
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.screen}>
        <Text style={styles.title} accessibilityRole="header">
          VISION COMPANION
        </Text>
        <Text style={styles.status}>Camera permission is required.</Text>
        <Pressable
          style={styles.button}
          accessibilityRole="button"
          accessibilityLabel="Grant camera permission"
          onPress={requestPermission}
        >
          <Text style={styles.buttonText}>GRANT PERMISSION</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar style="light" />
      <Text style={styles.title} accessibilityRole="header">
        VISION COMPANION
      </Text>
      <Text
        style={styles.status}
        accessibilityRole="text"
        accessibilityLiveRegion="polite"
        accessibilityLabel={`Listening status ${listeningLabel}. ${statusLine}`}
      >
        LISTENING STATUS: {listeningLabel}
      </Text>
      {showSettings ? (
        <SettingsScreen onClose={() => setShowSettings(false)} />
      ) : (
        <CameraView ref={cameraRef} style={styles.camera} facing="back" accessible={false} />
      )}
      <View style={styles.grid}>
        <ModeButton label="LOOK" hint="Describe the current view" onPress={() => void runMode("look")} />
        <ModeButton label="ASK" hint="Ask a question about the view" onPress={() => void voiceCommand()} />
        <ModeButton
          label={assistanceOn ? "STOP ASSISTANCE" : "START ASSISTANCE"}
          hint="Toggle continuous assistance"
          onPress={() => void toggleAssistance(!assistanceOn)}
        />
        <ModeButton label="READ" hint="Read visible text" onPress={() => void runMode("read", { question: "Read this." })} />
        <ModeButton label="FIND OBJECT" hint="Find a nearby object by voice" onPress={() => void findByVoice()} />
        <ModeButton label="VOICE" hint="Start voice command" onPress={() => void voiceCommand()} />
        <ModeButton label="SETTINGS" hint="Open settings" onPress={() => setShowSettings(true)} />
      </View>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Stop speaking"
        onPress={() => {
          stopSpeaking();
          setAppState(assistanceOn ? "CONTINUOUS_ASSISTANCE" : "IDLE");
        }}
      >
        <Text style={styles.muted}>Tap to stop speech. Primary interaction is voice.</Text>
      </Pressable>
    </SafeAreaView>
  );
}

function ModeButton({ label, hint, onPress }: { label: string; hint: string; onPress: () => void }) {
  return (
    <Pressable
      style={styles.button}
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityHint={hint}
    >
      <Text style={styles.buttonText}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg, padding: 16, gap: 10 },
  title: { color: colors.accent, fontSize: 28, fontWeight: "800", letterSpacing: 1 },
  status: { color: colors.text, fontSize: 18, fontWeight: "600" },
  camera: { flex: 1, minHeight: 180, borderRadius: 12, overflow: "hidden" },
  grid: { gap: 10 },
  button: {
    backgroundColor: colors.surface,
    minHeight: 56,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
  },
  buttonText: { color: colors.text, fontSize: 20, fontWeight: "800" },
  muted: { color: colors.muted, fontSize: 14, textAlign: "center", marginBottom: 8 },
});
