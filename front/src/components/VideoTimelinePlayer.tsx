import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getJobVideoUrl } from "../api/client";
import { eventColor, eventEmoji, formatTime } from "../constants/events";
import type { AnalysisResult, DetectedEvent } from "../types";
import { getActiveEventIndices, highlightWindowFromResult } from "../utils/eventHighlight";
import { TimelineChart } from "./charts/TimelineChart";

function clampTime(t: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) return Math.max(0, t);
  return Math.max(0, Math.min(duration, t));
}

export function VideoTimelinePlayer({
  jobId,
  result,
}: {
  jobId: string;
  result: AnalysisResult;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const rafRef = useRef<number | null>(null);

  const [playheadSec, setPlayheadSec] = useState(0);
  const [durationSec, setDurationSec] = useState(result.meta?.durationSeconds ?? 0);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);

  const isExcerpt = result.meta?.isExcerpt ?? true;
  const analysisStartSec = result.meta?.analysisStartSec ?? 0;
  const metaDurationMin = result.meta?.durationMinutes ?? durationSec / 60;
  const durationMin = durationSec > 0 ? durationSec / 60 : metaDurationMin;

  const eventVideoTime = useCallback(
    (timestamp: number) => timestamp + analysisStartSec,
    [analysisStartSec]
  );

  const sortedEvents = useMemo(
    () => [...result.predictions].sort((a, b) => a.timestamp - b.timestamp),
    [result.predictions]
  );

  const highlightWindowSec = useMemo(() => highlightWindowFromResult(result), [result]);

  const activeIndices = useMemo(
    () =>
      getActiveEventIndices(
        sortedEvents,
        playheadSec,
        highlightWindowSec,
        analysisStartSec
      ),
    [sortedEvents, playheadSec, highlightWindowSec, analysisStartSec]
  );

  const syncFromVideo = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    setPlayheadSec(v.currentTime);
    if (Number.isFinite(v.duration) && v.duration > 0) {
      setDurationSec(v.duration);
    }
  }, []);

  const seekTo = useCallback(
    (seconds: number) => {
      const v = videoRef.current;
      if (!v) return;
      const t = clampTime(seconds, v.duration || durationSec);
      v.currentTime = t;
      setPlayheadSec(t);
    },
    [durationSec]
  );

  const seekToEvent = useCallback(
    (timestamp: number) => {
      seekTo(eventVideoTime(timestamp));
    },
    [eventVideoTime, seekTo]
  );

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    const onMeta = () => {
      if (Number.isFinite(v.duration) && v.duration > 0) {
        setDurationSec(v.duration);
      }
      syncFromVideo();
    };

    v.addEventListener("loadedmetadata", onMeta);
    v.addEventListener("durationchange", onMeta);
    v.addEventListener("timeupdate", syncFromVideo);
    v.addEventListener("seeked", syncFromVideo);
    v.addEventListener("seeking", syncFromVideo);
    v.addEventListener("play", () => setPlaying(true));
    v.addEventListener("pause", () => setPlaying(false));
    v.addEventListener("ended", () => setPlaying(false));

    return () => {
      v.removeEventListener("loadedmetadata", onMeta);
      v.removeEventListener("durationchange", onMeta);
      v.removeEventListener("timeupdate", syncFromVideo);
      v.removeEventListener("seeked", syncFromVideo);
      v.removeEventListener("seeking", syncFromVideo);
    };
  }, [syncFromVideo]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playing) {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }

    const tick = () => {
      if (v && !v.paused) setPlayheadSec(v.currentTime);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing]);

  return (
    <div className="timeline-layout">
      <div className="timeline-main">
        <div className="video-wrap card">
          {videoError ? (
            <p className="video-error">{videoError}</p>
          ) : (
            <video
              ref={videoRef}
              className="timeline-video"
              src={getJobVideoUrl(jobId)}
              preload="metadata"
              playsInline
              controls
              onError={() =>
                setVideoError(
                  "Impossible de lire la vidéo. Vérifiez que l'API tourne (les clips SoccerNet sont convertis au premier chargement)."
                )
              }
            />
          )}
        </div>

        <div className="card">
          <p className="timeline-hint">
            Cliquez sur la timeline ou un emoji pour sauter. Les événements en cours se surlignent à droite.
          </p>
          <TimelineChart
            predictions={result.predictions}
            groundTruth={result.groundTruth}
            durationMinutes={durationMin}
            isExcerpt={isExcerpt}
            analysisStartSec={analysisStartSec}
            playheadSeconds={playheadSec}
            highlightWindowSec={highlightWindowSec}
            onSeek={seekTo}
            onSeekEvent={seekToEvent}
          />
        </div>
      </div>

      {sortedEvents.length > 0 && (
        <EventSidebar
          events={sortedEvents}
          isExcerpt={isExcerpt}
          activeIndices={activeIndices}
          onJump={seekToEvent}
        />
      )}
    </div>
  );
}

function EventSidebar({
  events,
  isExcerpt,
  activeIndices,
  onJump,
}: {
  events: DetectedEvent[];
  isExcerpt: boolean;
  activeIndices: number[];
  onJump: (sec: number) => void;
}) {
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    if (activeIndices.length !== 1) return;
    const el = itemRefs.current[activeIndices[0]];
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeIndices]);

  return (
    <aside className="timeline-events-sidebar card">
      <h3 className="timeline-sidebar-title">Événements</h3>
      <p className="timeline-sidebar-sub">{events.length} détection{events.length > 1 ? "s" : ""}</p>
      <div className="event-sidebar-list">
        {events.map((ev, i) => {
          const active = activeIndices.includes(i);
          const color = eventColor(ev.label);
          return (
            <button
              key={`${ev.timestamp}-${ev.label}-${i}`}
              ref={(el) => {
                itemRefs.current[i] = el;
              }}
              type="button"
              className={`event-sidebar-item${active ? " event-sidebar-item--active" : ""}`}
              style={
                active
                  ? ({
                      borderLeftColor: color,
                      background: `color-mix(in srgb, ${color} 22%, var(--bg-elevated))`,
                    } as React.CSSProperties)
                  : ({ borderLeftColor: color } as React.CSSProperties)
              }
              onClick={() => onJump(ev.timestamp)}
            >
              <span className="event-sidebar-emoji">{eventEmoji(ev.label)}</span>
              <span className="event-sidebar-body">
                <span className="event-sidebar-label">{ev.label}</span>
                <span className="event-sidebar-time">{formatTime(ev.timestamp, ev.half, isExcerpt)}</span>
                <span className="event-sidebar-conf">{(ev.confidence * 100).toFixed(0)} %</span>
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
