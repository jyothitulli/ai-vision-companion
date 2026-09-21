import { parseVoiceCommand } from "../speech/commands";

describe("Voice Command Parser", () => {
  it("parses LOOK commands and variations", () => {
    expect(parseVoiceCommand("look")).toEqual({ type: "LOOK" });
    expect(parseVoiceCommand("what is in front of me")).toEqual({ type: "LOOK" });
    expect(parseVoiceCommand("what's in front of me")).toEqual({ type: "LOOK" });
    expect(parseVoiceCommand("what is around me?")).toEqual({ type: "LOOK" });
    expect(parseVoiceCommand("describe surroundings")).toEqual({ type: "LOOK" });
    expect(parseVoiceCommand("describe my surroundings")).toEqual({ type: "LOOK" });
    expect(parseVoiceCommand("what do you see")).toEqual({ type: "LOOK" });
  });

  it("parses FIND commands with target extraction", () => {
    expect(parseVoiceCommand("find my keys")).toEqual({ type: "FIND", target: "keys" });
    expect(parseVoiceCommand("find the water bottle")).toEqual({ type: "FIND", target: "water bottle" });
    expect(parseVoiceCommand("where is my backpack?")).toEqual({ type: "FIND", target: "backpack" });
    expect(parseVoiceCommand("where's a chair")).toEqual({ type: "FIND", target: "chair" });
    expect(parseVoiceCommand("locate my glasses")).toEqual({ type: "FIND", target: "glasses" });
  });

  it("parses READ commands", () => {
    expect(parseVoiceCommand("read")).toEqual({ type: "READ", question: "read" });
    expect(parseVoiceCommand("read this")).toEqual({ type: "READ", question: "read this" });
    expect(parseVoiceCommand("read text")).toEqual({ type: "READ", question: "read text" });
    expect(parseVoiceCommand("what does this say")).toEqual({ type: "READ", question: "what does this say" });
  });

  it("parses STOP and REPEAT commands", () => {
    expect(parseVoiceCommand("stop")).toEqual({ type: "STOP" });
    expect(parseVoiceCommand("be quiet")).toEqual({ type: "STOP" });
    expect(parseVoiceCommand("cancel")).toEqual({ type: "STOP" });
    expect(parseVoiceCommand("repeat")).toEqual({ type: "REPEAT" });
    expect(parseVoiceCommand("repeat that")).toEqual({ type: "REPEAT" });
    expect(parseVoiceCommand("say that again")).toEqual({ type: "REPEAT" });
  });

  it("parses HELP and SETTINGS commands", () => {
    expect(parseVoiceCommand("help")).toEqual({ type: "HELP" });
    expect(parseVoiceCommand("what can you do")).toEqual({ type: "HELP" });
    expect(parseVoiceCommand("settings")).toEqual({ type: "SETTINGS" });
    expect(parseVoiceCommand("go back")).toEqual({ type: "BACK" });
  });

  it("parses CAMERA commands", () => {
    expect(parseVoiceCommand("open camera")).toEqual({ type: "OPEN_CAMERA" });
    expect(parseVoiceCommand("close camera")).toEqual({ type: "CLOSE_CAMERA" });
    expect(parseVoiceCommand("turn off camera")).toEqual({ type: "CLOSE_CAMERA" });
  });

  it("parses ASSISTANCE commands", () => {
    expect(parseVoiceCommand("start assistance")).toEqual({ type: "START_ASSISTANCE" });
    expect(parseVoiceCommand("stop assistance")).toEqual({ type: "STOP_ASSISTANCE" });
  });

  it("defaults unrecognized phrases to ASK with question payload", () => {
    expect(parseVoiceCommand("Is there an exit sign on the wall?")).toEqual({
      type: "ASK",
      question: "Is there an exit sign on the wall?",
    });
  });
});
