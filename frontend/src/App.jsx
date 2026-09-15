import { useEffect, useMemo, useState } from "react";

const API = "/api";

async function getJSON(path) {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

async function postJSON(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

function RiskGauge({ score, high }) {
  const pct = Math.round((score ?? 0) * 100);
  const color = high ? "var(--danger)" : pct >= 25 ? "var(--warn)" : "var(--ok)";
  return (
    <div className="gauge">
      <div className="gauge-ring" style={{ "--pct": pct, "--c": color }}>
        <div className="gauge-inner">
          <div className="gauge-score">{pct}%</div>
          <div className="gauge-label">30-day risk</div>
        </div>
      </div>
      <div className={`pill ${high ? "pill-danger" : "pill-ok"}`}>
        {high ? "Above threshold" : "Below threshold"}
      </div>
    </div>
  );
}

function FactorList({ factors }) {
  if (!factors?.length) return <p className="muted">Select a patient to see SHAP factors.</p>;
  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.contribution)), 1e-6);
  return (
    <ul className="factors">
      {factors.map((f) => {
        const w = (Math.abs(f.contribution) / maxAbs) * 100;
        const up = f.contribution >= 0;
        return (
          <li key={f.feature}>
            <div className="factor-head">
              <span className="mono">{f.feature}</span>
              <span className="muted">= {Number(f.value).toFixed(2)}</span>
              <span className={up ? "pos" : "neg"}>
                {up ? "+" : ""}
                {f.contribution.toFixed(3)}
              </span>
            </div>
            <div className="bar-track">
              <div
                className={`bar ${up ? "bar-pos" : "bar-neg"}`}
                style={{ width: `${w}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export default function App() {
  const [patients, setPatients] = useState([]);
  const [selected, setSelected] = useState(null);
  const [explain, setExplain] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [p, m] = await Promise.all([
          getJSON("/patients?limit=40"),
          getJSON("/metrics"),
        ]);
        setPatients(p.patients || []);
        setMetrics(m);
        if (p.patients?.length) setSelected(p.patients[0].patient_id);
      } catch (e) {
        setError(
          "API unreachable. Start the backend: uvicorn src.syncura.api.main:app --port 8000"
        );
      }
    })();
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    postJSON("/explain", { patient_id: selected })
      .then(setExplain)
      .catch((e) => setError(String(e.message || e)))
      .finally(() => setLoading(false));
  }, [selected]);

  const primaryName = metrics?.primary_model ?? "–";
  const testAuc = useMemo(() => {
    const t = metrics?.test?.[metrics?.primary_model];
    return t?.roc_auc != null ? t.roc_auc.toFixed(3) : "–";
  }, [metrics]);

  return (
    <div className="app">
      <header className="top">
        <div>
          <div className="brand">syncura-readmission</div>
          <div className="sub">Generated cohort</div>
        </div>
        <div className="meta-chips">
          <span className="chip">Synthetic EHR</span>
          <span className="chip">SHAP</span>
          <span className="chip">model: {primaryName}</span>
          <span className="chip">holdout AUC {testAuc}</span>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <main className="grid">
        <section className="card">
          <h2>Synthetic cohort</h2>
          <p className="muted">Demo slice of holdout patients (fabricated IDs).</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Age</th>
                  <th>LOS</th>
                  <th>Charlson</th>
                  <th>Label*</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p) => (
                  <tr
                    key={p.patient_id}
                    className={selected === p.patient_id ? "selected" : ""}
                    onClick={() => setSelected(p.patient_id)}
                  >
                    <td className="mono">{p.patient_id}</td>
                    <td>{p.age}</td>
                    <td>{p.length_of_stay}</td>
                    <td>{p.charlson_proxy}</td>
                    <td>{p.readmitted_30d}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fine">* Label shown for inspection only; not available at prediction time.</p>
        </section>

        <section className="card stack">
          <h2>Risk score</h2>
          {loading && <p className="muted">Computing SHAP…</p>}
          {explain && (
            <>
              <RiskGauge score={explain.risk_score} high={explain.high_risk} />
              <p className="muted center">
                Threshold {explain.threshold} · {explain.patient_id}
              </p>
            </>
          )}
        </section>

        <section className="card">
          <h2>Top contributing factors</h2>
          <p className="muted">SHAP contributions toward predicted readmission risk.</p>
          <FactorList factors={explain?.top_factors} />
        </section>
      </main>

      <footer>
        Karim Ladak · MIT
      </footer>
    </div>
  );
}
