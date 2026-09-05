import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { getApiBase } from "../api/client";
import { colors } from "../theme";

type Props = {
  onClose: () => void;
};

export function SettingsScreen({ onClose }: Props) {
  const [speechRate] = useState("0.92");
  return (
    <View style={styles.wrap} accessibilityLabel="Settings">
      <Text style={styles.heading} accessibilityRole="header">
        SETTINGS
      </Text>
      <Text style={styles.row} accessibilityLabel={`Backend ${getApiBase()}`}>
        Backend: {getApiBase()}
      </Text>
      <Text style={styles.row} accessibilityLabel={`Speech rate ${speechRate}`}>
        Speech rate: {speechRate} (platform TTS)
      </Text>
      <Text style={styles.row}>
        Camera frames are not stored. High contrast is always on.
      </Text>
      <Text style={styles.row}>
        This is an assistive perception aid, not guaranteed navigation.
      </Text>
      <Pressable
        style={styles.button}
        onPress={onClose}
        accessibilityRole="button"
        accessibilityLabel="Close settings"
      >
        <Text style={styles.buttonText}>CLOSE</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: 12, paddingVertical: 8 },
  heading: { color: colors.accent, fontSize: 24, fontWeight: "800" },
  row: { color: colors.text, fontSize: 18, lineHeight: 26 },
  button: {
    backgroundColor: colors.surface,
    minHeight: 56,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonText: { color: colors.text, fontSize: 20, fontWeight: "800" },
});
