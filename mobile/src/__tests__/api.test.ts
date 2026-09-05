import { getApiBase } from "../api/client";

describe("api client", () => {
  it("defaults to local backend", () => {
    expect(getApiBase()).toContain("http");
  });
});
