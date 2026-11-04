import { useQuery } from '@tanstack/react-query';

interface Health { status: string; device: string; models_loaded: string[] }

async function fetchHealth(): Promise<Health> {
  const response = await fetch('/api/health');
  if (!response.ok) throw new Error('The API is unavailable.');
  return response.json();
}

export default function App() {
  const health = useQuery({ queryKey: ['health'], queryFn: fetchHealth, refetchInterval: 10000 });
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-5 md:px-12">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <span className="text-xl font-semibold tracking-tight">Moto<span className="text-teal-400">ReID</span></span>
          <span className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300">Evidence review workspace</span>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-20 md:px-12">
        <p className="text-sm font-medium uppercase tracking-widest text-teal-400">Multi-camera intelligence</p>
        <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">Hello MotoReID</h1>
        <p className="mt-6 max-w-xl text-lg leading-relaxed text-slate-400">
          Track motorcycles across cameras. Compare the evidence. Make uncertainty visible.
        </p>
        <section aria-label="System status" className="mt-12 max-w-2xl rounded-2xl border border-slate-800 bg-slate-900 p-7">
          <h2 className="text-lg font-semibold">Project foundation</h2>
          <p className="mt-2 text-sm leading-6 text-slate-400">The API and interface foundation are being validated. Video processing and evidence review will arrive in the following build phases.</p>
          <div role="status" className="mt-6 border-t border-slate-800 pt-5 text-sm">
            {health.isPending ? 'Connecting to API…' : health.isError ? (
              <span className="text-amber-300">API unavailable. Check that the backend is running.</span>
            ) : (
              <span className="text-teal-300">API connected · Models not initialized</span>
            )}
          </div>
          {health.isError && <button onClick={() => void health.refetch()} className="mt-4 rounded-lg border border-slate-600 px-4 py-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-300">Retry connection</button>}
        </section>
      </main>
    </div>
  );
}
