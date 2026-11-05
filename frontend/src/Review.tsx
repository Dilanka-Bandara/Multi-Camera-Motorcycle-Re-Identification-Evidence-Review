import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { Players, Jump } from "./Players";
import type { Camera, Candidate, RunResult, RunStatus, Track } from "./types";

const fmt = (n: unknown) =>
  typeof n === "number" ? Number(n.toFixed(3)).toString() : "—";
const pretty = (s: string) => s.replaceAll("_", " ");
function Metrics({ run }: { run: RunResult }) {
  const metrics = run.metrics;
  const decisions = metrics?.decisions as
    | Record<string, Record<string, number | null>>
    | undefined;
  const falseMatches = metrics?.false_matches as Candidate[] | undefined;
  return (
    <>
      <h2>Evaluation metrics</h2>
      {metrics ? (
        <>
          <div className="cards">
            {[
              "rank1",
              "rank5",
              "correct_rejection_rate",
              "roc_auc",
              "expected_calibration_error",
            ].map((k) => (
              <div className="panel" key={k}>
                <small>{pretty(k)}</small>
                <strong>{fmt(metrics[k])}</strong>
              </div>
            ))}
          </div>
          <section className="panel">
            <h3>Match decisions</h3>
            <table>
              <thead>
                <tr>
                  {["Decision", "Precision", "Recall", "F1"].map((t) => (
                    <th key={t}>{t}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(decisions ?? {}).map(([key, row]) => (
                  <tr key={key}>
                    <td>{pretty(key)}</td>
                    {["precision", "recall", "f1"].map((k) => (
                      <td key={k}>{fmt(row[k])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <h3>False matches</h3>
            {falseMatches?.length ? (
              <table>
                <tbody>
                  {falseMatches.map((c) => (
                    <tr key={c.pair_id}>
                      <td>{c.pair_id}</td>
                      <td>{fmt(c.scores.combined)}</td>
                      <td>{c.verification?.summary}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p>No false accepted matches.</p>
            )}
          </section>
        </>
      ) : (
        <p>Ground truth is unavailable for this run.</p>
      )}
      <section className="panel">
        <h3>Processing time</h3>
        <table>
          <tbody>
            {Object.entries(run.timings).map(([k, v]) => (
              <tr key={k}>
                <td>{pretty(k)}</td>
                <td>{v.toFixed(2)} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
function Workspace({
  run,
  tracks,
  metrics,
}: {
  run: RunResult;
  tracks: Record<Camera, Track[]>;
  metrics: boolean;
}) {
  const [query, setQuery] = useState(run.queries[0]?.query_track_id ?? ""),
    [pair, setPair] = useState("");
  const [filter, setFilter] = useState("all"),
    [jump, setJump] = useState<Jump | null>(null);
  const all = [...tracks.A, ...tracks.B];
  const row = run.queries.find((q) => q.query_track_id === query),
    c = row?.candidates.find((c) => c.pair_id === pair) ?? row?.candidates[0];
  const lhs = all.find((t) => t.track_id === query),
    rhs = all.find((t) => t.track_id === c?.candidate_track_id);
  const ids = { A: "", B: "" };
  if (lhs) ids[lhs.camera] = lhs.track_id;
  if (rhs) ids[rhs.camera] = rhs.track_id;
  function select(id: string, selected?: Candidate) {
    setQuery(id);
    setPair(selected?.pair_id ?? "");
    const first = all.find((t) => t.track_id === id),
      second = all.find((t) => t.track_id === selected?.candidate_track_id);
    const t = { a: 0, b: 0 };
    for (const v of [first, second])
      if (v)
        t[v.camera === "A" ? "a" : "b"] =
          v.t_start_s - run.videos[v.camera]!.start_time_s;
    setJump((prev) => ({ ...t, sequence: (prev?.sequence ?? 0) + 1 }));
  }
  const verdicts = run.queries.flatMap((q) =>
    q.candidates.map((c) => c.verification?.status),
  );
  const media = "/media/runs/" + run.run_id + "/";
  return (
    <>
      <div className="title-row">
        <div>
          <h1>Run review</h1>
          <code>{run.run_id}</code>
        </div>
        <div className="links">
          {["export.json", "export.csv", "queries.csv"].map((name, i) => (
            <a
              key={name}
              download
              href={"/api/runs/" + run.run_id + "/" + name}
            >
              {["Results JSON", "Candidates CSV", "Queries CSV"][i]}
            </a>
          ))}
        </div>
      </div>
      <div className="summary">
        {["accepted", "uncertain", "rejected"].map((k) => (
          <span key={k} className={"badge " + k}>
            {verdicts.filter((s) => s === k).length} {k}
          </span>
        ))}
        <span>
          Clock offset {fmt(run.clock_offset.estimated_s)} s ·{" "}
          {run.clock_offset.reliable ? "reliable" : "uncertain"}
        </span>
        <span>{fmt(run.timings.total)} s processing</span>
      </div>
      <nav className="switch">
        <a
          aria-current={!metrics ? "page" : undefined}
          href={"#run/" + run.run_id}
        >
          Evidence review
        </a>
        <a
          aria-current={metrics ? "page" : undefined}
          href={"#metrics/" + run.run_id}
        >
          Metrics
        </a>
      </nav>
      {metrics ? (
        <Metrics run={run} />
      ) : (
        <>
          <Players result={run} tracks={tracks} ids={ids} jump={jump} />
          <section className="panel timeline">
            <h2>Presence timeline</h2>
            {(["A", "B"] as Camera[]).map((cam) => (
              <div key={cam}>
                <h3>Camera {cam}</h3>
                {tracks[cam].map((t) => {
                  const meta = run.videos[cam]!;
                  const time = t.t_start_s - meta.start_time_s;
                  return (
                    <button
                      key={t.track_id}
                      title={t.track_id + " · " + time.toFixed(1) + " s"}
                      aria-label={"Seek to " + t.track_id}
                      style={{
                        marginLeft: (100 * time) / meta.duration_s + "%",
                        width:
                          Math.max(
                            1,
                            (100 * (t.t_end_s - t.t_start_s)) / meta.duration_s,
                          ) + "%",
                      }}
                      onClick={() => {
                        const q = run.queries.find(
                          (q) => q.query_track_id === t.track_id,
                        );
                        if (q) select(q.query_track_id, q.candidates[0]);
                        else if (
                          row?.candidates.some(
                            (c) => c.candidate_track_id === t.track_id,
                          )
                        )
                          select(
                            row.query_track_id,
                            row.candidates.find(
                              (c) => c.candidate_track_id === t.track_id,
                            ),
                          );
                        else
                          setJump((prev) => ({
                            a: time,
                            b: time,
                            sequence: (prev?.sequence ?? 0) + 1,
                          }));
                      }}
                    />
                  );
                })}
              </div>
            ))}
          </section>
          <div className="review-grid">
            <section className="panel">
              <h2>Query tracks</h2>
              <label>
                Outcome
                <select
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                >
                  {["all", "matched", "uncertain", "no_reliable_match"].map(
                    (k) => (
                      <option value={k} key={k}>
                        {pretty(k)}
                      </option>
                    ),
                  )}
                </select>
              </label>
              {run.queries
                .filter((q) => filter === "all" || q.outcome === filter)
                .map((q) => (
                  <button
                    className="track"
                    aria-pressed={q.query_track_id === query}
                    key={q.query_track_id}
                    onClick={() => select(q.query_track_id, q.candidates[0])}
                  >
                    {q.query_track_id}
                    <small>{pretty(q.outcome)}</small>
                  </button>
                ))}
            </section>
            <section className="panel">
              <h2>Ranked candidates</h2>
              {!row?.candidates.length ? (
                <p>No candidates.</p>
              ) : (
                <table>
                  <thead>
                    <tr>
                      {["Track", "Score", "Δt (s)", "Verdict"].map((k) => (
                        <th key={k}>{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {row.candidates.map((item) => (
                      <tr
                        key={item.pair_id}
                        className={item === c ? "selected" : ""}
                      >
                        <td>
                          <button onClick={() => select(query, item)}>
                            {item.rank}. {item.candidate_track_id}
                          </button>
                        </td>
                        <td>{fmt(item.scores.combined)}</td>
                        <td>{fmt(item.delta_t_adjusted_s)}</td>
                        <td>
                          {item.verification?.status ?? "unreviewed"}
                          {item.ambiguous && <small>Ambiguous</small>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
            <section className="panel">
              <h2>Compare evidence</h2>
              {[lhs, rhs].map(
                (t) =>
                  t && (
                    <div key={t.track_id}>
                      <h3>{t.track_id}</h3>
                      <div className="strip">
                        {t.crops.map((c) => (
                          <img
                            key={c.frame_idx}
                            alt={"Frame " + c.frame_idx}
                            src={media + c.path_full}
                          />
                        ))}
                      </div>
                    </div>
                  ),
              )}
              {c?.verification ? (
                <>
                  <h3 className={"badge " + c.verification.status}>
                    {c.verification.status} · {fmt(c.verification.confidence)}
                  </h3>
                  <p>{c.verification.summary}</p>
                  <p>
                    Vehicle {fmt(c.verification.features.vehicle_part)} · Rider{" "}
                    {c.verification.evidence.some(
                      (e) =>
                        e.check === "rider_part_similarity" &&
                        e.polarity === "missing",
                    )
                      ? "unavailable"
                      : fmt(c.verification.features.rider_part)}
                  </p>
                  {c.verification.keypoint_viz_path && (
                    <img
                      className="geometry"
                      alt="Geometric inliers between vehicle crops"
                      src={media + c.verification.keypoint_viz_path}
                    />
                  )}
                  {c.verification.evidence.map((e, i) => (
                    <div className={"evidence " + e.polarity} key={i}>
                      <small>
                        {e.polarity} · {e.severity}
                      </small>
                      {e.message}
                    </div>
                  ))}
                </>
              ) : (
                <p>No independent review available.</p>
              )}
              {c?.competing_pair_ids.length ? (
                <>
                  <h3>Competing pairs</h3>
                  {c.competing_pair_ids.map((p) => (
                    <code key={p}>{p} </code>
                  ))}
                </>
              ) : null}
            </section>
          </div>
        </>
      )}
    </>
  );
}
export function Review({ id, metrics }: { id: string; metrics: boolean }) {
  const path = "/api/runs/" + id;
  const status = useQuery({
    queryKey: ["status", id],
    queryFn: () => api<RunStatus>(path + "/status"),
    refetchInterval: (q) =>
      ["done", "failed"].includes(q.state.data?.state ?? "") ? false : 1000,
  });
  const data = useQuery({
    queryKey: ["review", id],
    enabled: status.data?.state === "done",
    queryFn: async () => {
      const [run, A, B] = await Promise.all([
        api<RunResult>(path),
        api<Track[]>(path + "/tracks/A"),
        api<Track[]>(path + "/tracks/B"),
      ]);
      return { run, tracks: { A, B } };
    },
  });
  return (
    <main>
      {status.isError || data.isError ? (
        <p role="alert">Could not load this run.</p>
      ) : data.data ? (
        <Workspace key={id} {...data.data} metrics={metrics} />
      ) : (
        <section className="panel">
          <h1>Processing run</h1>
          <p>{status.data?.error ?? pretty(status.data?.stage ?? "Loading")}</p>
          <progress max={1} value={status.data?.progress ?? 0} />
        </section>
      )}
    </main>
  );
}
