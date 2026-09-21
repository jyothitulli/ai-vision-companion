import { ACCESSIBILITY_BUTTONS, PRIMARY_CONTROLS } from "../ui/modes";

describe("accessibility labels and controls", () => {
  it("exposes all 8 required accessibility controls with hints", () => {
    const ids = ACCESSIBILITY_BUTTONS.map((item) => item.id);
    expect(ids).toEqual(
      expect.arrayContaining([
        "start_listening",
        "stop_listening",
        "toggle_camera",
        "capture_analyze",
        "stop_speaking",
        "repeat_response",
        "help",
        "settings",
      ]),
    );

    for (const control of ACCESSIBILITY_BUTTONS) {
      expect(control.label.length).toBeGreaterThan(3);
      expect(control.hint.length).toBeGreaterThan(10);
      expect(control.role).toBe("button");
    }
  });

  it("maintains backward-compatible primary controls", () => {
    const labels = PRIMARY_CONTROLS.map((item) => item.label);
    expect(labels).toEqual(
      expect.arrayContaining(["LOOK", "ASK", "READ", "FIND OBJECT", "SETTINGS"]),
    );
  });
});
