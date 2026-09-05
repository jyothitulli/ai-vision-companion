import { PRIMARY_CONTROLS } from "../ui/modes";

describe("accessibility labels", () => {
  it("exposes large primary controls with hints", () => {
    const labels = PRIMARY_CONTROLS.map((item) => item.label);
    expect(labels).toEqual(
      expect.arrayContaining(["LOOK", "ASK", "READ", "FIND OBJECT", "SETTINGS"]),
    );
    for (const control of PRIMARY_CONTROLS) {
      expect(control.hint.length).toBeGreaterThan(8);
    }
  });
});
