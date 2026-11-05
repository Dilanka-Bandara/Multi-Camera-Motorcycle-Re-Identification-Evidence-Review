import { FormEvent, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { RunStatus } from "./types";

export function Home() {
  const presets = useQuery({
    queryKey: ["presets"],
    queryFn: () => api<string[]>("/api/presets"),
  });
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: () => api<RunStatus[]>("/api/runs"),
    refetchInterval: 5000,
  });
  const [source, setSource] = useState("preset");
  const [clock, setClock] = useState("estimate");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function start(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    setError("");
    setBusy(true);
    try {
      const overrides = {
        target: {
          classes:
            form.get("target") === "all"
              ? ["car", "bus", "truck", "motorcycle"]
              : ["motorcycle"],
        },
        topology: {
          direction: form.get("direction"),
          min_transit_s: Number(form.get("min")),
          max_transit_s: Number(form.get("max")),
          clock_offset: {
            mode: clock,
            ...(clock === "fixed"
              ? { fixed_s: Number(form.get("fixed")) }
              : {}),
          },
        },
      };
      let run: { run_id: string };
      if (source === "preset")
        run = await api("/api/runs/from-preset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            preset: form.get("preset"),
            config_overrides: overrides,
          }),
        });
      else {
        const payload = new FormData();
        for (const key of ["video_a", "video_b"])
          payload.set(key, form.get(key)!);
        payload.set("config_overrides", JSON.stringify(overrides));
        run = await api("/api/runs", { method: "POST", body: payload });
      }
      location.hash = "run/" + run.run_id;
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main>
      <h1>Cross-camera review</h1>
      <div className="home-grid">
        <section className="panel">
          <h2>New run</h2>
          <form onSubmit={start}>
            <fieldset>
              <legend>Video source</legend>
              <div className="switch">
                {["preset", "upload"].map((s) => (
                  <button
                    key={s}
                    type="button"
                    aria-pressed={source === s}
                    onClick={() => setSource(s)}
                  >
                    {s === "preset" ? "Use preset" : "Upload videos"}
                  </button>
                ))}
              </div>
              {source === "preset" ? (
                <label>
                  Preset
                  <select name="preset" required>
                    {(presets.data ?? []).map((p) => (
                      <option key={p} value={p}>
                        {p === "dev_fixture" ? "Development fixture" : p}
                      </option>
                    ))}
                  </select>
                  {presets.isError
                    ? "Could not load presets."
                    : presets.data?.length === 0
                      ? "No local preset data found."
                      : ""}
                </label>
              ) : (
                <div className="fields">
                  {["A", "B"].map((c) => (
                    <label key={c}>
                      Camera {c}
                      <input
                        required
                        type="file"
                        accept="video/*"
                        name={"video_" + c.toLowerCase()}
                      />
                    </label>
                  ))}
                </div>
              )}
            </fieldset>
            <fieldset>
              <legend>Matching settings</legend>
              <div className="fields">
                <label>
                  Target
                  <select name="target">
                    <option value="motorcycle">Motorcycles</option>
                    <option value="all">All vehicles</option>
                  </select>
                </label>
                <label>
                  Direction
                  <select name="direction">
                    <option value="A_to_B">A → B</option>
                    <option value="B_to_A">B → A</option>
                    <option value="both">Both</option>
                  </select>
                </label>
                <label>
                  Minimum transit (s)
                  <input
                    name="min"
                    required
                    type="number"
                    step="any"
                    min="0"
                    defaultValue={5}
                  />
                </label>
                <label>
                  Maximum transit (s)
                  <input
                    name="max"
                    required
                    type="number"
                    step="any"
                    min="0.01"
                    key={source}
                    defaultValue={source === "preset" ? 20 : 60}
                  />
                </label>
                <label>
                  Clock offset
                  <select
                    value={clock}
                    onChange={(e) => setClock(e.target.value)}
                  >
                    <option value="estimate">Estimate</option>
                    <option value="fixed">Fixed</option>
                    <option value="none">None</option>
                  </select>
                </label>
                {clock === "fixed" && (
                  <label>
                    Camera B offset (s)
                    <input
                      required
                      type="number"
                      step="any"
                      name="fixed"
                      defaultValue={0}
                    />
                  </label>
                )}
              </div>
            </fieldset>
            {error && <p role="alert">{error}</p>}
            <button
              className="primary"
              disabled={busy || (source === "preset" && !presets.data?.length)}
            >
              {busy ? "Submitting…" : "Start run"}
            </button>
          </form>
        </section>
        <section className="panel">
          <h2>Previous runs</h2>
          {runs.isError ? (
            <p role="alert">Could not load run history.</p>
          ) : runs.data?.length ? (
            runs.data.map((r) => (
              <a className="run-link" key={r.run_id} href={"#run/" + r.run_id}>
                <code>{r.run_id.slice(0, 12)}</code>
                <span className="badge">{r.state}</span>
              </a>
            ))
          ) : (
            <p>{runs.isPending ? "Loading…" : "No previous runs."}</p>
          )}
        </section>
      </div>
    </main>
  );
}
