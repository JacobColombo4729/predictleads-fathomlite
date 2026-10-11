import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceState = "idle" | "asking" | "recording" | "sending";

/** How long one spoken question can be. It is stopped automatically at this point. */
export const VOICE_MAX_SECONDS = 30;

/** True when this browser can record from a microphone (needs a secure page such as http://127.0.0.1 or https). */
export function voiceSupported(): boolean {
  return typeof navigator !== "undefined" && Boolean(navigator.mediaDevices?.getUserMedia) && typeof MediaRecorder !== "undefined";
}

function pickMimeType(): string | undefined {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return candidates.find((t) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t));
}

function explain(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "Microphone access was blocked. Allow it from the icon in your browser's address bar, then press Talk again.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No microphone was found. Plug one in or check your system settings.";
  if (name === "NotReadableError") return "The microphone is in use by another app.";
  return "The microphone could not be started.";
}

/**
 * Records one short spoken question. The microphone is opened only when the person presses Talk and is released as soon
 * as they press Stop (or the time limit is reached). Nothing is kept: the recording goes to the callback and is discarded.
 */
export function useVoiceRecorder(onRecording: (audio: Blob) => Promise<void>, onError: (message: string) => void) {
  const [state, setState] = useState<VoiceState>("idle");
  const [seconds, setSeconds] = useState(0);
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const chunks = useRef<Blob[]>([]);

  const release = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  }, []);

  useEffect(() => () => {
    release();
    if (recorder.current && recorder.current.state !== "inactive") recorder.current.stop();
  }, [release]);

  const stop = useCallback(() => {
    if (recorder.current && recorder.current.state !== "inactive") recorder.current.stop();
  }, []);

  const start = useCallback(async () => {
    if (state !== "idle") return;
    setState("asking");
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      setState("idle");
      onError(explain(e));
      return;
    }
    try {
      const mimeType = pickMimeType();
      const rec = new MediaRecorder(stream.current, mimeType ? { mimeType } : undefined);
      chunks.current = [];
      rec.ondataavailable = (ev) => { if (ev.data.size > 0) chunks.current.push(ev.data); };
      rec.onstop = () => {
        release();
        const blob = new Blob(chunks.current, { type: rec.mimeType || mimeType || "audio/webm" });
        chunks.current = [];
        if (blob.size < 1500) {
          setState("idle");
          onError("We didn't catch anything. Press Talk, speak your question, then press Stop.");
          return;
        }
        setState("sending");
        onRecording(blob).catch((e) => onError(e instanceof Error ? e.message : "That could not be turned into text.")).finally(() => setState("idle"));
      };
      recorder.current = rec;
      rec.start();
      setSeconds(0);
      setState("recording");
      timer.current = setInterval(() => {
        setSeconds((s) => {
          if (s + 1 >= VOICE_MAX_SECONDS) stop();
          return s + 1;
        });
      }, 1000);
    } catch {
      release();
      setState("idle");
      onError("This browser could not record audio.");
    }
  }, [state, onError, onRecording, release, stop]);

  return { state, seconds, start, stop };
}
