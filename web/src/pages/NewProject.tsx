import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Rocket } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, useApi } from "../api";
import { PageHeader } from "../components/Layout";
import { Card, ErrorBox, Seg } from "../components/ui";
import { StepPicker } from "./Pipeline";

const slug = (s: string) => s.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);

export function NewProject() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const { data: catalog } = useApi<any>("/pipeline/steps");
  const [name, setName] = useState("");
  const [id, setId] = useState("");
  const [idTouched, setIdTouched] = useState(false);
  const [dois, setDois] = useState("");
  const [description, setDescription] = useState("");
  const [budget, setBudget] = useState(8000);
  const [template, setTemplate] = useState<"map" | "full" | "custom">("map");
  const [steps, setSteps] = useState<string[]>([]);
  const [start, setStart] = useState(true);
  const [params, setParams] = useState("");
  useEffect(() => {
    if (!idTouched) setId(slug(name));
  }, [name, idTouched]);
  useEffect(() => {
    if (catalog && template !== "custom") setSteps(catalog.templates[template]);
  }, [catalog, template]);
  const create = useMutation({
    mutationFn: () => {
      let p: any = undefined;
      if (params.trim()) p = JSON.parse(params);
      return api<{ id: string }>("/projects", {
        method: "POST",
        json: {
          id, name: name || id, description: description || undefined, paper_budget: budget,
          seed_dois: dois.split(/[\s,;]+/).map((d) => d.trim()).filter(Boolean), steps, params: p, start,
        },
      });
    },
    onSuccess: (r) => {
      qc.invalidateQueries();
      nav(`/p/${r.id}/pipeline`);
    },
  });
  return (
    <div className="mx-auto max-w-[980px]">
      <PageHeader title="New project" sub="A project maps one field, grown from one or a few seed papers through their citation graph." />
      <div className="flex flex-col gap-4">
        <Card title="Field">
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="label">Name</label>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Wasserstein robust control" />
            </div>
            <div>
              <label className="label">Id (folder name, used by the CLI)</label>
              <input className="input font-mono" value={id} onChange={(e) => { setId(e.target.value); setIdTouched(true); }} placeholder="wasserstein-control" />
            </div>
            <div className="md:col-span-2">
              <label className="label">Seed DOIs (one per line)</label>
              <textarea className="input font-mono" rows={3} value={dois} onChange={(e) => setDois(e.target.value)} placeholder="10.1007/s10107-017-1172-1" />
            </div>
            <div className="md:col-span-2">
              <label className="label">Description (optional)</label>
              <input className="input" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What you want to learn about this field" />
            </div>
            <div>
              <label className="label">Paper budget</label>
              <input className="input" type="number" min={500} step={500} value={budget} onChange={(e) => setBudget(Number(e.target.value))} />
              <p className="muted mt-1 text-[12px]">The pilot used 8,000 (5,410 papers after deduplication). Larger maps take longer and cost more extraction.</p>
            </div>
            <div>
              <label className="label">Spend caps (defaults)</label>
              <p className="muted text-[12.5px] leading-relaxed">Extraction $5, labels $2, bridge judge $2, metadata $2, outside works about $1. The pilot spent about $0.95 in total.</p>
            </div>
          </div>
        </Card>
        <Card title="Pipeline">
          <Seg value={template} onChange={setTemplate} options={[
            { value: "map", label: "Core map (recommended)" }, { value: "full", label: "Full, with backtest and lab" }, { value: "custom", label: "Choose steps" },
          ]} />
          {catalog && (
            <div className="mt-3">
              <StepPicker wide catalog={catalog} selected={steps} setSelected={(s) => { setSteps(s); setTemplate("custom"); }} />
            </div>
          )}
          <label className="mt-3 flex items-center gap-2 text-[13px]">
            <input type="checkbox" checked={start} onChange={(e) => setStart(e.target.checked)} className="accent-[var(--color-brand-600)]" />
            Start as soon as the queue is free (otherwise it waits, held, until you resume it)
          </label>
          <details className="mt-3">
            <summary className="muted cursor-pointer text-[12.5px] font-medium">Advanced: parameter overrides (JSON)</summary>
            <textarea className="input mt-2 font-mono text-[12px]" rows={5} value={params} onChange={(e) => setParams(e.target.value)}
              placeholder='{"expansion": {"forward_years": 4}, "extraction": {"llm_spend_cap_usd": 3}}' />
          </details>
        </Card>
        {create.error && <ErrorBox error={create.error} />}
        <div className="flex justify-end gap-2">
          <button className="btn" onClick={() => nav(-1)}>Cancel</button>
          <button className="btn btn-primary" disabled={!id || !dois.trim() || !steps.length || create.isPending} onClick={() => create.mutate()}>
            <Rocket className="size-4" /> Create and queue
          </button>
        </div>
      </div>
    </div>
  );
}
