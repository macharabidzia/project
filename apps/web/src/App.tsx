import { useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";

import { api, type RuntimeReadinessStatus, type RuntimeTelemetryStatus, type SystemConfig } from "./lib/api";


function App() {
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [readiness, setReadiness] = useState<RuntimeReadinessStatus | null>(null);
  const [telemetry, setTelemetry] = useState<RuntimeTelemetryStatus | null>(null);
  const [connectionState, setConnectionState] = useState("disconnected");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [events, setEvents] = useState<string[]>([]);

  const roomRef = useRef<Room | null>(null);
  const attachedElementsRef = useRef<HTMLElement[]>([]);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => {
      void refresh();
    }, 2000);
    return () => {
      window.clearInterval(interval);
      void disconnect();
    };
  }, []);

  async function refresh() {
    try {
      const [configValue, readinessValue, telemetryValue] = await Promise.all([
        api.getSystemConfig(),
        api.getRuntimeReadiness(),
        api.getRuntimeTelemetry(),
      ]);
      setSystemConfig(configValue);
      setReadiness(readinessValue);
      setTelemetry(telemetryValue);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to refresh runtime status.";
      setErrorMessage(message);
    }
  }

  function pushEvent(message: string) {
    const stamp = new Date().toLocaleTimeString();
    setEvents((current) => [`[${stamp}] ${message}`, ...current].slice(0, 8));
  }

  function detachAllTracks() {
    for (const element of attachedElementsRef.current) {
      try {
        element.remove();
      } catch {
      }
    }
    attachedElementsRef.current = [];
  }

  async function connect() {
    if (roomRef.current || connectionState === "connecting") {
      return;
    }
    setErrorMessage("");
    setConnectionState("connecting");
    try {
      const identity = `voice-web-${Math.random().toString(16).slice(2, 10)}`;
      const tokenPayload = await api.getLiveKitToken(identity);
      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      room.on(RoomEvent.Connected, () => {
        setConnectionState("connected");
        pushEvent(`LiveKit connected: room ${tokenPayload.room_name}.`);
      });
      room.on(RoomEvent.Disconnected, () => {
        setConnectionState("disconnected");
        pushEvent("LiveKit disconnected.");
      });
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) {
          return;
        }
        const element = track.attach();
        if (element instanceof HTMLMediaElement) {
          element.autoplay = true;
          element.muted = false;
          element.style.display = "none";
          document.body.appendChild(element);
          attachedElementsRef.current.push(element);
        }
      });
      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        const elements = track.detach();
        for (const element of elements) {
          element.remove();
        }
      });

      await room.connect(tokenPayload.url, tokenPayload.token, { autoSubscribe: true });
      await room.localParticipant.setMicrophoneEnabled(true);
      roomRef.current = room;
      pushEvent("Microphone capture and WebRTC publish active via LiveKit.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to start microphone pipeline.";
      setErrorMessage(message);
      setConnectionState("failed");
      pushEvent(message);
      await disconnect();
    }
  }

  async function disconnect() {
    const room = roomRef.current;
    roomRef.current = null;
    if (room) {
      await room.disconnect();
    }
    detachAllTracks();
    setConnectionState("disconnected");
  }

  return (
    <div className="shell">
      <header className="hero">
        <div className="hero__copy">
          <p className="eyebrow">Voice OS</p>
          <h1>Single-process continuous voice runtime</h1>
          <p className="hero__lede">
            Browser audio publishes into a local LiveKit WebRTC room, then flows through CPU Vosk ASR, one KernelRuntime authority, GPU0 vLLM token streaming, GPU1 CosyVoice3 PCM streaming, and LiveKit audio egress.
          </p>
        </div>
        <div className="hero__badge-stack">
          <div className="hero-badge">
            <span>Connection</span>
            <strong>{connectionState.toUpperCase()}</strong>
          </div>
          <div className="hero-badge">
            <span>Backend</span>
            <strong>{systemConfig?.transport.livekit_url ?? "ws://127.0.0.1:7880"}</strong>
          </div>
        </div>
      </header>

      <section className="grid">
        <div className="panel panel--session">
          <div className="panel__heading">
            <h2>Runtime Status</h2>
          </div>
          <p>{readiness?.summary ?? "Waiting for runtime readiness."}</p>
          <div className="session-form__actions">
            <button onClick={() => void connect()} disabled={!readiness?.ready || connectionState === "connected"} type="button">
              Start Mic Stream
            </button>
            <button className="button-secondary" onClick={() => void disconnect()} type="button">
              Disconnect
            </button>
          </div>
          {errorMessage ? <p className="error-callout">{errorMessage}</p> : null}
          <div className="session-meta">
            <div>
              <span>ASR</span>
              <strong>{readiness?.checks.asr_cpu?.detail ?? "--"}</strong>
            </div>
            <div>
              <span>GPU0</span>
              <strong>{readiness?.checks.vllm_gpu0?.detail ?? "--"}</strong>
            </div>
            <div>
              <span>GPU1</span>
              <strong>{readiness?.checks.tts_gpu1?.detail ?? "--"}</strong>
            </div>
          </div>
        </div>

        <div className="status-panel">
          <div className="status-panel__eyebrow">Telemetry</div>
          <h3>Kernel</h3>
          <dl className="stats-grid">
            <div>
              <dt>Ingress queue</dt>
              <dd>{telemetry?.stats.kernel_queue_depth ?? 0}</dd>
            </div>
            <div>
              <dt>PCM queue</dt>
              <dd>{telemetry?.stats.pcm_queue_depth ?? 0}</dd>
            </div>
            <div>
              <dt>Ready</dt>
              <dd>{readiness?.ready ? "YES" : "NO"}</dd>
            </div>
            <div>
              <dt>Session</dt>
              <dd>{telemetry?.session_id ?? "--"}</dd>
            </div>
          </dl>
        </div>

        <div className="panel panel--layers">
          <div className="panel__heading">
            <h2>Pipeline</h2>
          </div>
          <ul className="flow-list">
            {systemConfig?.layers.map((layer) => (
              <li key={layer.name}>
                <strong>{layer.name}</strong> - {layer.backend} - {layer.purpose}
              </li>
            )) ?? <li>Loading pipeline layers...</li>}
          </ul>
        </div>

        <div className="panel panel--layers">
          <div className="panel__heading">
            <h2>Event Feed</h2>
          </div>
          <ul className="event-feed">
            {events.length ? events.map((item) => <li key={item}>{item}</li>) : <li>No live events yet.</li>}
          </ul>
        </div>
      </section>
    </div>
  );
}

export default App;
