import { useEffect, useRef, useState } from "react";
import type { Camera, RunResult, Track, VideoMeta } from "./types";

function Overlay({
  video,
  tracks,
  selected,
  meta,
}: {
  video: HTMLVideoElement | null;
  tracks: Track[];
  selected: string;
  meta: VideoMeta;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!video || !canvas.current) return;
    const ctx = canvas.current.getContext("2d");
    if (!ctx) return;
    const frames = new Map<
      number,
      { track: Track; box: Track["boxes"][number]["bbox"] }[]
    >();
    for (const track of tracks)
      for (const frame of track.boxes) {
        const list = frames.get(frame.frame_idx) ?? [];
        list.push({ track, box: frame.bbox });
        frames.set(frame.frame_idx, list);
      }
    function draw(time: number) {
      if (!ctx) return;
      ctx.clearRect(0, 0, meta.width, meta.height);
      for (const { track, box } of frames.get(Math.floor(time * meta.fps)) ??
        []) {
        const active = track.track_id === selected;
        ctx.strokeStyle = active ? "#fbc656" : "#70ddd0";
        ctx.lineWidth = active ? 4 : 2;
        ctx.strokeRect(
          box.x1 * meta.width,
          box.y1 * meta.height,
          (box.x2 - box.x1) * meta.width,
          (box.y2 - box.y1) * meta.height,
        );
      }
    }
    let id = 0;
    const v = video;
    const byFrame = typeof v.requestVideoFrameCallback === "function";
    function tick() {
      id = byFrame
        ? v.requestVideoFrameCallback((_, m) => {
            draw(m.mediaTime);
            tick();
          })
        : requestAnimationFrame(() => {
            draw(v.currentTime);
            tick();
          });
    }
    const paused = () => draw(v.currentTime);
    paused();
    tick();
    v.addEventListener("seeked", paused);
    return () => {
      if (byFrame) v.cancelVideoFrameCallback(id);
      else cancelAnimationFrame(id);
      v.removeEventListener("seeked", paused);
    };
  }, [video, tracks, selected, meta]);
  return (
    <canvas
      ref={canvas}
      width={meta.width}
      height={meta.height}
      aria-label="Tracked vehicle boxes"
    />
  );
}

export interface Jump {
  a: number;
  b: number;
  sequence: number;
}
export function Players({
  result,
  tracks,
  ids,
  jump,
}: {
  result: RunResult;
  tracks: Record<Camera, Track[]>;
  ids: Record<Camera, string>;
  jump: Jump | null;
}) {
  const [a, setA] = useState<HTMLVideoElement | null>(null),
    [b, setB] = useState<HTMLVideoElement | null>(null);
  const [playing, setPlaying] = useState(false),
    [at, setAt] = useState(0),
    [offset, setOffset] = useState(result.clock_offset.estimated_s);
  const [error, setError] = useState("");
  const pairShift = useRef(0);
  const va = result.videos.A!,
    vb = result.videos.B!;
  function seek(v: HTMLVideoElement | null, t: number) {
    if (v && Number.isFinite(v.duration))
      v.currentTime = Math.max(0, Math.min(v.duration, t));
  }
  useEffect(() => {
    if (!jump || !a || !b) return;
    const align = () => {
      seek(a, jump.a);
      seek(b, jump.b);
      pairShift.current = jump.b - jump.a;
    };
    align();
    a.addEventListener("loadedmetadata", align);
    b.addEventListener("loadedmetadata", align);
    return () => {
      a.removeEventListener("loadedmetadata", align);
      b.removeEventListener("loadedmetadata", align);
    };
  }, [jump, a, b]);
  const delta = offset - result.clock_offset.estimated_s;
  async function toggle() {
    if (!a || !b) return;
    if (playing) {
      a.pause();
      b.pause();
      return;
    }
    try {
      await Promise.all([a.play(), b.play()]);
      setError("");
    } catch {
      a.pause();
      b.pause();
      setError("Could not play both camera videos.");
    }
  }
  return (
    <section className="panel">
      <div className="videos">
        {(["A", "B"] as Camera[]).map((c) => (
          <div key={c}>
            <h3>
              Camera {c} <code>{ids[c]}</code>
            </h3>
            <div className="frame">
              <video
                ref={c === "A" ? setA : setB}
                preload="auto"
                muted
                playsInline
                src={
                  "/media/runs/" + result.run_id + "/" + result.videos[c]!.path
                }
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => {
                  a?.pause();
                  b?.pause();
                }}
                onTimeUpdate={
                  c === "A"
                    ? () => {
                        if (!a || !b) return;
                        setAt(a.currentTime);
                        const target =
                          a.currentTime + pairShift.current + delta;
                        if (!a.paused && Math.abs(b.currentTime - target) > 0.3)
                          seek(b, target);
                      }
                    : undefined
                }
              />
              <Overlay
                video={c === "A" ? a : b}
                meta={result.videos[c]!}
                tracks={tracks[c]}
                selected={ids[c]}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="transport">
        <button onClick={toggle}>{playing ? "Pause both" : "Play both"}</button>
        <label>
          Seek A{" "}
          <input
            type="range"
            min={0}
            max={va.duration_s}
            step={1 / va.fps}
            value={at}
            onChange={(e) => {
              const t = Number(e.target.value);
              seek(a, t);
              seek(b, t + pairShift.current + delta);
              setAt(t);
            }}
          />
        </label>
        <output>{at.toFixed(1)} s</output>
        <label>
          B offset{" "}
          <input
            aria-label="B offset"
            type="range"
            min={Math.min(-vb.duration_s, result.clock_offset.estimated_s)}
            max={Math.max(vb.duration_s, result.clock_offset.estimated_s)}
            value={offset}
            step={0.1}
            onChange={(e) => {
              const n = Number(e.target.value);
              setOffset(n);
              seek(
                b,
                at + pairShift.current + n - result.clock_offset.estimated_s,
              );
            }}
          />
        </label>
        <output>{offset.toFixed(1)} s</output>
      </div>
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
