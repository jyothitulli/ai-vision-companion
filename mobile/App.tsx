import { StatusBar } from "expo-status-bar";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  AccessibilityInfo,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { healthCheck, postSession, transcribe, uploadFrame } from "./src/api/client";
import { cancelRecording, recordUtterance } from "./src/speech/stt";
import { speak, stopSpeaking } from "./src/speech/tts";
import { parseVoiceCommand } from "./src/speech/commands";
import { AppStateName, stateAnnouncement } from "./src/state/appState";
import { colors } from "./src/theme";
import { SettingsScreen } from "./src/screens/SettingsScreen";
import { HelpScreen } from "./src/screens/HelpScreen";

const SESSION_ID = "mobile-default";

type Mode = "look" | "ask" | "read" | "find" | "assistance";

export default function App() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [appState, setAppState] = useState<AppStateName>("IDLE");
  const [statusLine, setStatusLine] = useState("Vision assistant ready.");
  const [lastSpokenAnswer, setLastSpokenAnswer] = useState<string>(
    "Vision assistant ready. How can I help?",
  );
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [assistanceOn, setAssistanceOn] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  const assistanceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const assistanceActiveRef = useRef(false);
  const isAssistanceLoopRunning = useRef(false);
  const isLookRunning = useRef(false);
  const hasGreetedRef = useRef(false);

  // Initial startup: check backend health and announce greeting
  useEffect(() => {
    healthCheck().then((up) => {
      setBackendUp(up);
      if (!hasGreetedRef.current) {
        hasGreetedRef.current = true;
        const greeting = up
          ? "Vision assistant ready. How can I help?"
          : "Vision assistant ready. Note: backend server is currently unreachable.";
        setLastSpokenAnswer(greeting);
        setStatusLine(greeting);
        void speak(greeting);
      }
    });
  }, []);

  // Announce state transitions for TalkBack / VoiceOver
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

  async function speakAndSave(text: string) {
    setLastSpokenAnswer(text);
    setStatusLine(text);
    setAppState("SPEAKING");
    await speak(text);
    setAppState(assistanceActiveRef.current ? "CONTINUOUS_ASSISTANCE" : "IDLE");
  }

  async function captureUri(): Promise<string | null> {
    if (!cameraRef.current || !cameraEnabled) {
      return null;
    }
    setAppState("CAPTURING");
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.6, skipProcessing: true });
      return photo?.uri ?? null;
    } catch {
      return null;
    }
  }

  async function runMode(mode: Mode, extra: Record<string, string> = {}) {
    if (mode === "look") {
      if (isLookRunning.current) return;
      isLookRunning.current = true;
    }
    try {
      if (!permission?.granted) {
        setAppState("PERMISSION_REQUIRED");
        await speakAndSave("Camera permission is required to analyze surroundings.");
        return;
      }
      if (!cameraEnabled) {
        setAppState("IDLE");
        await speakAndSave("Camera is currently closed. Say open camera or tap the camera button to enable it.");
        return;
      }

      const uri = await captureUri();
      if (!uri) {
        setAppState("ERROR");
        await speakAndSave("I couldn't capture a frame. Make sure the camera is open and try again.");
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
      const answer = result.answer || "I couldn't detect anything clearly in front of you.";
      await speakAndSave(answer);
    } catch {
      setAppState("ERROR");
      await speakAndSave("I couldn't reach the server. Please check your connection and try again.");
    } finally {
      if (mode === "look") {
        isLookRunning.current = false;
      }
    }
  }

  async function startListening() {
    // If speaking, silence first
    stopSpeaking();

    setAppState("LISTENING");
    setStatusLine("Listening. Speak your command now.");
    await speak("Listening.");

    const uri = await recordUtterance();
    if (!uri) {
      // Could be cancelled or permission denied
      if (appState === "LISTENING") {
        setAppState("IDLE");
        setStatusLine("Listening stopped.");
      }
      return;
    }

    setAppState("PROCESSING");
    setStatusLine("Transcribing voice command...");
    const rawText = await transcribe(uri);
    const command = parseVoiceCommand(rawText);

    switch (command.type) {
      case "STOP":
        handleStopSpeaking();
        break;

      case "REPEAT":
        handleRepeatResponse();
        break;

      case "HELP":
        setShowHelp(true);
        setShowSettings(false);
        await speakAndSave(
          "Help opened. You can say look, find my object, read this, repeat, or stop.",
        );
        break;

      case "SETTINGS":
        setShowSettings(true);
        setShowHelp(false);
        await speakAndSave("Settings opened.");
        break;

      case "BACK":
        setShowHelp(false);
        setShowSettings(false);
        await speakAndSave("Back to main vision screen.");
        break;

      case "OPEN_CAMERA":
        setCameraEnabled(true);
        await speakAndSave("Camera is now open.");
        break;

      case "CLOSE_CAMERA":
        setCameraEnabled(false);
        await speakAndSave("Camera is now closed.");
        break;

      case "START_ASSISTANCE":
        await toggleAssistance(true);
        break;

      case "STOP_ASSISTANCE":
        await toggleAssistance(false);
        break;

      case "LOOK":
        await runMode("look");
        break;

      case "FIND":
        await runMode("find", { target: command.target });
        break;

      case "READ":
        await runMode("read", { question: command.question || "Read this." });
        break;

      case "ASK":
        await runMode("ask", { question: command.question });
        break;

      case "UNKNOWN":
      default:
        await speakAndSave("I didn't catch that. Say look, find, read, repeat, or help.");
        break;
    }
  }

  async function stopListeningAction() {
    await cancelRecording();
    setAppState("IDLE");
    setStatusLine("Listening stopped.");
    await speak("Listening stopped.");
  }

  function handleStopSpeaking() {
    stopSpeaking();
    if (assistanceTimer.current) {
      clearTimeout(assistanceTimer.current);
      assistanceTimer.current = null;
    }
    setAppState(assistanceOn ? "CONTINUOUS_ASSISTANCE" : "IDLE");
    setStatusLine("Speech stopped.");
  }

  function handleRepeatResponse() {
    if (!lastSpokenAnswer) {
      void speak("No previous response to repeat.");
      return;
    }
    void speak(lastSpokenAnswer);
  }

  async function startAssistanceLoop() {
    if (isAssistanceLoopRunning.current) return;
    isAssistanceLoopRunning.current = true;
    try {
      while (assistanceActiveRef.current) {
        if (!permission?.granted || !cameraEnabled) break;
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
      await speakAndSave("Continuous assistance is off.");
      return;
    }
    await postSession("/api/assistance/start", SESSION_ID);
    setAppState("CONTINUOUS_ASSISTANCE");
    await speakAndSave("Continuous assistance is on. I will announce important scene changes.");
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
        <Text style={styles.status}>Camera permission is required to assist you.</Text>
        <Pressable
          style={styles.actionButton}
          accessibilityRole="button"
          accessibilityLabel="Grant camera permission"
          accessibilityHint="Allows the app to capture photos and identify objects"
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

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title} accessibilityRole="header">
          VISION COMPANION
        </Text>
      </View>

      {/* Status Live Region */}
      <View
        style={styles.statusBox}
        accessible={true}
        accessibilityRole="text"
        accessibilityLiveRegion="polite"
        accessibilityLabel={`Status: ${listeningLabel}. ${statusLine}`}
      >
        <Text style={styles.statusHeader}>STATUS: {listeningLabel}</Text>
        <Text style={styles.statusText} numberOfLines={3}>
          {statusLine}
        </Text>
      </View>

      {/* Main Content Area */}
      {showHelp ? (
        <HelpScreen onClose={() => setShowHelp(false)} />
      ) : showSettings ? (
        <SettingsScreen onClose={() => setShowSettings(false)} />
      ) : (
        <View style={styles.mainArea}>
          {cameraEnabled ? (
            <CameraView
              ref={cameraRef}
              style={styles.camera}
              facing="back"
              accessible={false}
              importantForAccessibility="no"
            />
          ) : (
            <View style={styles.cameraPlaceholder} accessible={false}>
              <Text style={styles.placeholderText}>Camera is closed</Text>
            </View>
          )}

          {/* Action Button Grid */}
          <ScrollView contentContainerStyle={styles.controlsScroll} showsVerticalScrollIndicator={false}>
            {/* 1. Voice Controls */}
            {appState === "LISTENING" ? (
              <AccessibleButton
                label="STOP LISTENING"
                hint="Cancel active voice listening"
                highlight
                onPress={() => void stopListeningAction()}
              />
            ) : (
              <AccessibleButton
                label="START LISTENING"
                hint="Start listening for voice commands such as look, find, or read"
                highlight
                onPress={() => void startListening()}
              />
            )}

            {/* 2. Direct Capture & Analyze */}
            <AccessibleButton
              label="CAPTURE & ANALYZE"
              hint="Take a picture now and announce objects and surroundings"
              onPress={() => void runMode("look")}
            />

            {/* 3. Repeat Response */}
            <AccessibleButton
              label="REPEAT RESPONSE"
              hint="Replay the last spoken description or answer"
              onPress={handleRepeatResponse}
            />

            {/* 4. Stop Speaking */}
            <AccessibleButton
              label="STOP SPEAKING"
              hint="Immediately silence the speech assistant"
              onPress={handleStopSpeaking}
            />

            {/* 5. Camera Toggle */}
            <AccessibleButton
              label={cameraEnabled ? "CLOSE CAMERA" : "OPEN CAMERA"}
              hint={cameraEnabled ? "Turn off camera preview to save power" : "Turn on camera preview"}
              onPress={() => {
                const next = !cameraEnabled;
                setCameraEnabled(next);
                void speak(next ? "Camera opened." : "Camera closed.");
              }}
            />

            {/* 6. Continuous Assistance Toggle */}
            <AccessibleButton
              label={assistanceOn ? "STOP ASSISTANCE" : "START ASSISTANCE"}
              hint="Toggle continuous scene monitoring"
              onPress={() => void toggleAssistance(!assistanceOn)}
            />

            {/* 7. Help */}
            <AccessibleButton
              label="HELP"
              hint="Open spoken guide and list of voice commands"
              onPress={() => {
                setShowHelp(true);
                setShowSettings(false);
              }}
            />

            {/* 8. Settings */}
            <AccessibleButton
              label="SETTINGS"
              hint="Adjust backend server and voice settings"
              onPress={() => {
                setShowSettings(true);
                setShowHelp(false);
              }}
            />
          </ScrollView>
        </View>
      )}
    </SafeAreaView>
  );
}

function AccessibleButton({
  label,
  hint,
  highlight = false,
  onPress,
}: {
  label: string;
  hint: string;
  highlight?: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      style={[styles.actionButton, highlight && styles.actionButtonHighlight]}
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityHint={hint}
    >
      <Text style={[styles.buttonText, highlight && styles.buttonTextHighlight]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.bg,
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 16,
  },
  header: {
    paddingVertical: 6,
  },
  title: {
    color: colors.accent,
    fontSize: 26,
    fontWeight: "900",
    letterSpacing: 1,
  },
  statusBox: {
    backgroundColor: colors.surface,
    padding: 12,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: colors.border,
    marginVertical: 6,
    minHeight: 68,
    justifyContent: "center",
  },
  statusHeader: {
    color: colors.accent,
    fontSize: 15,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  statusText: {
    color: colors.text,
    fontSize: 16,
    fontWeight: "600",
    marginTop: 2,
  },
  status: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "600",
    marginVertical: 12,
  },
  mainArea: {
    flex: 1,
    gap: 8,
  },
  camera: {
    height: 140,
    borderRadius: 10,
    overflow: "hidden",
  },
  cameraPlaceholder: {
    height: 80,
    borderRadius: 10,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: "center",
    justifyContent: "center",
  },
  placeholderText: {
    color: colors.muted,
    fontSize: 16,
    fontWeight: "700",
  },
  controlsScroll: {
    gap: 8,
    paddingVertical: 6,
  },
  actionButton: {
    backgroundColor: colors.surface,
    minHeight: 56,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 16,
  },
  actionButtonHighlight: {
    backgroundColor: colors.accent,
    borderColor: "#FFFFFF",
  },
  buttonText: {
    color: colors.text,
    fontSize: 19,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  buttonTextHighlight: {
    color: "#05070B",
  },
});
