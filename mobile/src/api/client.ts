const DEFAULT_API = "http://127.0.0.1:8000";

export function getApiBase(): string {
  return process.env.EXPO_PUBLIC_API_URL ?? DEFAULT_API;
}

export type AnalyzePayload = {
  success: boolean;
  mode: string;
  answer: string;
  confidence: number;
  processing_time_ms: number;
  warnings?: string[];
  error?: string;
};

async function postForm(path: string, form: FormData, timeoutMs = 45000): Promise<AnalyzePayload> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${getApiBase()}${path}`, {
      method: "POST",
      headers: {
        "X-API-Key": process.env.EXPO_PUBLIC_API_KEY ?? "dev-local-key",
      },
      body: form,
      signal: controller.signal,
    });
    const body = (await response.json()) as AnalyzePayload & { detail?: string };
    if (!response.ok) {
      return {
        success: false,
        mode: "error",
        answer: body.detail || body.error || "I couldn't reach the assistant. Please try again.",
        confidence: 0,
        processing_time_ms: 0,
      };
    }
    return body;
  } catch (error) {
    const aborted = error instanceof Error && error.name === "AbortError";
    return {
      success: false,
      mode: "error",
      answer: aborted
        ? "The request timed out. Please try again."
        : "I couldn't reach the server. Check that the backend is running.",
      confidence: 0,
      processing_time_ms: 0,
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function uploadFrame(
  uri: string,
  path: string,
  extra: Record<string, string> = {},
): Promise<AnalyzePayload> {
  const form = new FormData();
  form.append("file", {
    uri,
    name: "frame.jpg",
    type: "image/jpeg",
  } as unknown as Blob);
  for (const [key, value] of Object.entries(extra)) {
    form.append(key, value);
  }
  return postForm(path, form);
}

export async function postSession(path: string, sessionId: string): Promise<AnalyzePayload> {
  const form = new FormData();
  form.append("session_id", sessionId);
  return postForm(path, form, 15000);
}

export async function transcribe(uri: string): Promise<string> {
  const form = new FormData();
  form.append("file", {
    uri,
    name: "speech.wav",
    type: "audio/wav",
  } as unknown as Blob);
  const result = await postForm("/api/speech/transcribe", form, 60000);
  return "text" in result && typeof (result as { text?: string }).text === "string"
    ? (result as { text: string }).text
    : result.answer;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${getApiBase()}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
}
