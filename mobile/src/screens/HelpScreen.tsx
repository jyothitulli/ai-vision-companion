import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { colors } from "../theme";

type Props = {
  onClose: () => void;
};

export function HelpScreen({ onClose }: Props) {
  return (
    <View style={styles.container} accessibilityLabel="Help and instructions screen">
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={styles.heading} accessibilityRole="header">
          VOICE COMMANDS & HELP
        </Text>

        <Text style={styles.sectionHeader} accessibilityRole="header">
          Spoken Commands
        </Text>
        <Text style={styles.item}>
          • <Text style={styles.bold}>"Look"</Text> or <Text style={styles.bold}>"What is in front of me"</Text>: Describes objects and layout.
        </Text>
        <Text style={styles.item}>
          • <Text style={styles.bold}>"Find my [object]"</Text>: Locates item with left/right/center direction and distance.
        </Text>
        <Text style={styles.item}>
          • <Text style={styles.bold}>"Read this"</Text>: Reads text, labels, or documents in view.
        </Text>
        <Text style={styles.item}>
          • <Text style={styles.bold}>"Repeat"</Text>: Replays the last spoken answer immediately.
        </Text>
        <Text style={styles.item}>
          • <Text style={styles.bold}>"Stop"</Text>: Silences speech output right away.
        </Text>
        <Text style={styles.item}>
          • <Text style={styles.bold}>"Start assistance"</Text>: Continuous tracking mode for real-time changes.
        </Text>
        <Text style={styles.item}>
          • <Text style={styles.bold}>"Open camera"</Text> or <Text style={styles.bold}>"Close camera"</Text>: Controls camera feed.
        </Text>
        <Text style={styles.item}>
          • <Text style={styles.bold}>"Settings"</Text>: Opens configuration screen.
        </Text>

        <Text style={styles.sectionHeader} accessibilityRole="header">
          Accessible Touch Shortcuts
        </Text>
        <Text style={styles.item}>
          • Double-tap <Text style={styles.bold}>START LISTENING</Text> to speak any command hands-free.
        </Text>
        <Text style={styles.item}>
          • Double-tap <Text style={styles.bold}>CAPTURE & ANALYZE</Text> to take a photo directly.
        </Text>
        <Text style={styles.item}>
          • Double-tap <Text style={styles.bold}>REPEAT RESPONSE</Text> to hear the last description again.
        </Text>
        <Text style={styles.item}>
          • Double-tap <Text style={styles.bold}>STOP SPEAKING</Text> to mute speech immediately.
        </Text>
      </ScrollView>

      <Pressable
        style={styles.closeButton}
        onPress={onClose}
        accessibilityRole="button"
        accessibilityLabel="Close help"
        accessibilityHint="Returns to the main assistant screen"
      >
        <Text style={styles.closeButtonText}>CLOSE HELP</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
    paddingVertical: 12,
  },
  scroll: {
    paddingBottom: 20,
    gap: 10,
  },
  heading: {
    color: colors.accent,
    fontSize: 26,
    fontWeight: "800",
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  sectionHeader: {
    color: colors.accent,
    fontSize: 20,
    fontWeight: "700",
    marginTop: 12,
    marginBottom: 4,
  },
  item: {
    color: colors.text,
    fontSize: 17,
    lineHeight: 24,
  },
  bold: {
    fontWeight: "700",
    color: "#FFFFFF",
  },
  closeButton: {
    backgroundColor: colors.surface,
    minHeight: 56,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 8,
  },
  closeButtonText: {
    color: colors.text,
    fontSize: 20,
    fontWeight: "800",
  },
});
