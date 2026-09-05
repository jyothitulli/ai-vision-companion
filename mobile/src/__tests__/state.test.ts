import { stateAnnouncement } from "../state/appState";

describe("app states", () => {
  it("has an accessibility announcement for every state", () => {
    const states = [
      "IDLE",
      "LISTENING",
      "CAPTURING",
      "PROCESSING",
      "SPEAKING",
      "CONTINUOUS_ASSISTANCE",
      "ERROR",
      "PERMISSION_REQUIRED",
    ] as const;
    for (const state of states) {
      expect(stateAnnouncement[state].length).toBeGreaterThan(3);
    }
  });
});
