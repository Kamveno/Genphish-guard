# STEPS — run it, host it, demo it

Everything you need to get the working system in front of the reviewer.
No theory in this file; that lives in `README.md` and `docs/REVIEW2.md`.

---

## 0. What you now have

| Thing | Where | What it is |
|---|---|---|
| **REST API** | `api/main.py` | 15 endpoints. OpenAPI docs at `/docs`. |
| **Web console** | `web/` | 6-tab operator console, served by the API at `/`. |
| **Inference service** | `api/service.py` | Loads/trains detectors, scores mail, runs drift, raises incidents. |
| **Model build** | `scripts/build_model.py` | Trains 16 detectors, writes `models/`, exports browser weights. |
| **Serve script** | `scripts/serve.sh` | Start/stop the server. |
| **Deploy configs** | `Dockerfile`, `render.yaml`, `Procfile`, `space.yaml` | One-click hosting. |
| **Research pipeline** | `src/genphish/` | The 20 modules behind all of it. |

---

## 1. Run it locally (3 commands, ~2 minutes)

```bash
cd genphish-guard

# 1. install
pip install -r requirements.txt -r requirements-api.txt

# 2. train the detectors once (~60 s) — writes models/ and web/model.json
python scripts/build_model.py

# 3. serve
./scripts/serve.sh
```

Open **http://localhost:8000**

That is the whole thing. If you skip step 2 the API trains on first boot
instead — the first request just takes ~20 s longer.

**Windows / no bash:**
```bash
python -m uvicorn api.main:app --reload --port 8000
```

**Background instead of foreground:**
```bash
./scripts/serve.sh --bg     # logs to server.log, pid in .server.pid
./scripts/serve.sh --stop
```

### Verify it came up

```bash
curl -s localhost:8000/api/health
```
Expect `"status":"ok"`, `"ready":true`, and `"data_mode":"SYNTHETIC"`.

---

## 2. The six tabs, and what each one proves

| Tab | What to do | What it demonstrates |
|---|---|---|
| **Detect** | Click a sample chip → **Analyse email** | The live detector. Confidence score, zero-trust tier, per-feature attribution, SOAR incident raised automatically. |
| **Transfer** | Just look | The core claim. AUC stays ~0.975 off the diagonal while F1 collapses ~29 pp. Signal survives, threshold dies. |
| **Calibration** | Just look | The contribution. Distribution matching recovers ~90% of the labelled gap using zero target labels. Also shows EM converging to the wrong prior (0.135 vs 0.249). |
| **Drift Monitor** | **Run stream** | The trigger. Martingale wealth decays pre-shift, climbs after the generator switch, crosses the Ville ceiling, raises an alarm. δ = 0.01. |
| **SOAR** | Expand an incident | The operational end. Incident schema, mapped playbook, dispatch to webhook or simulation. |
| **API** | **Send request** | That the console is a thin client, not a mock. Live 200 OK with latency. |

### Changing the calibration changes the policy

On the **Detect** tab, analyse the same email with each `Threshold calibration`
option. The operating threshold and the whole DELIVER/DEFUSE/BLOCK band
structure move with it:

| Calibration | Threshold (pooled XGBoost) |
|---|---|
| stale | 0.9684 |
| distribution matching | 0.0813 |
| prior-shift EM | 0.6820 |
| self-training | 0.4235 |
| labelled oracle | 0.5028 |

This is the single most useful thing to click in front of the panel: it shows
calibration is not a number in a table, it is the thing that decides what
happens to the mail.

### Two samples are deliberate failures

`False positive — legitimate IT security notice` and
`False negative — mild phishing, markers suppressed` are there on purpose,
with an explanation that appears when you select them. Show them. A panel
that finds a failure mode you did not mention assumes you did not know; a
panel shown the failure mode with its measurement assumes you did.

---

## 3. Use the API directly

Interactive docs: **http://localhost:8000/docs** (click *Try it out* on any endpoint).

```bash
# score one email
curl -s -X POST localhost:8000/api/detect \
  -H 'Content-Type: application/json' \
  -d '{"text":"URGENT verify your account at https://secure-verify.com",
       "detector":"xgboost","trained_on":"pooled","calibration":"stale"}'

# score a batch, raising an incident for each BLOCK/DEFUSE
curl -s -X POST localhost:8000/api/detect/batch \
  -H 'Content-Type: application/json' \
  -d '{"texts":["first email","second email"],"raise_incidents":true}'

# the transfer matrices
curl -s localhost:8000/api/transfer

# calibration comparison
curl -s localhost:8000/api/calibration

# drift stream: 1200 emails, generator switches at 300
curl -s 'localhost:8000/api/drift/simulate?n=1200&shift_at=300&source=deepseek3.2&target=gpt-4.1'

# incident log
curl -s localhost:8000/api/soar/incidents
```

### Endpoint list

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness, data mode, loaded detectors |
| GET | `/api/stats` | Corpus composition, feature groups, run summaries |
| GET | `/api/samples` | Built-in demo emails |
| POST | `/api/detect` | Score one email → tier, contributions, incident |
| POST | `/api/detect/batch` | Score up to 500 emails |
| GET | `/api/features` | Extract the 17 features without scoring |
| GET | `/api/transfer` | N×N AUC and F1 matrices |
| GET | `/api/calibration` | Per-pair thresholds and F1, all 5 methods |
| GET | `/api/drift/simulate` | Stream with a generator switch; wealth + alarm |
| POST | `/api/drift/session` | Open an online martingale session |
| POST | `/api/drift/step` | Feed one score into an open session |
| GET | `/api/soar/incidents` | Incident log with playbooks |
| DELETE | `/api/soar/incidents` | Clear the log |
| POST | `/api/soar/dispatch` | Send an incident to a webhook or simulation |
| GET | `/api/soar/playbook/{tier}` | Playbook for BLOCK / DEFUSE / DELIVER |

---

## 4. Host it so the reviewer can open a link

Pick one. **Render** is the least work.

### Option A — Render (recommended, free)

1. Push this repo to GitHub.
2. [render.com](https://render.com) → **New** → **Blueprint** → select the repo.
3. Render reads `render.yaml`. Click **Apply**. First build takes ~5 minutes.
4. You get `https://genphish-guard.onrender.com`.

> The free instance sleeps after ~15 minutes idle and takes ~40 s to wake.
> **Open the URL two minutes before the review starts** so it is warm.

### Option B — Hugging Face Spaces (free, does not sleep as aggressively)

1. [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Docker**.
2. Push this repo to the Space's git remote.
3. Copy the front-matter from `space.yaml` into the top of the Space's
   `README.md` (Spaces reads config from there).
4. It builds from the `Dockerfile` and serves on port 8000.

### Option C — Docker, anywhere

```bash
docker build -t genphish-guard .
docker run -p 8000:8000 genphish-guard
```

Works on any VPS, your lab machine, or a college server.

### Option D — Railway / Fly.io

Both read the `Procfile`. Push the repo and set the start command to
`python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT`.

### Option E — your own laptop on the lab network

```bash
HOST=0.0.0.0 ./scripts/serve.sh --bg
```
Then share `http://<your-laptop-ip>:8000`. Find the IP with `ip addr` or
`ipconfig`. Fine for a room demo, no hosting account needed.

---

## 5. The research pipeline (the numbers behind the console)

The console reads results the pipeline produces. Regenerate them any time:

```bash
# core: S1-S8 — transfer matrix, calibration, martingale, zero-trust, errors (~15 s)
python -m genphish.run_all

# extended: ablations, adversarial robustness, neural arm, 10-partition stats (~5 min)
python -m genphish.run_extended --stability --logreg --control-perms 19

# the invariants
python -m pytest -q      # 22 tests
```

Outputs land in `results/` (CSVs + `summary.json` + 13 figures in
`results/figures/`). The API picks up `results/transfer_*.csv` automatically;
if they are absent it computes the matrices live instead.

---

## 6. Switching from SYNTHETIC to REAL data

Right now every output is stamped **SYNTHETIC**. To switch:

1. Download the corpus: Zenodo DOI **10.5281/zenodo.20250116**
   (`cross-model-phishing.zip`).
2. Extract the feature table and save it as `data/corpus_features.csv`.
3. Re-run:
   ```bash
   python scripts/build_model.py
   python -m genphish.run_all
   python -m genphish.run_extended --stability
   ./scripts/serve.sh --bg
   ```

Nothing else changes. `corpus.load_corpus()` prefers the real file when it
exists, and every figure, the API's `data_mode` field and the console's
header badge all restamp to **REAL** on their own.

**Re-run the de-quantization probe first.** Real counts are integers too, so
the 9.27 pp AUC brittleness may be real rather than an artefact of the
synthetic corpus. That is a finding either way — say which it turned out to be.

---

## 7. Demo running order (8 minutes)

1. **Open the console.** Point at the header: `API live`, `data: SYNTHETIC`.
   Say the data status out loud before anyone asks.
2. **Detect → credential-harvest sample → Analyse.**
   BLOCK, confidence 0.999, `urgency_term_count` dominating the attribution,
   SOAR incident raised with its playbook.
3. **Detect → false-positive sample.** Read the note. "This is a real IT
   notice and we flag it. Here is why, and here is why the DEFUSE tier exists."
4. **Detect → change calibration** through all five options on the same email.
   Watch the threshold and the tier bands move. "Calibration is the policy."
5. **Transfer tab.** "AUC 0.975 off-diagonal, F1 down 29 points. The signal
   survives; the threshold dies."
6. **Calibration tab.** "90% of the labelled gap, zero target labels. And it
   is significantly *below* the oracle — it is not free."
7. **Drift Monitor → Run stream.** "No labels. The band rate rises, the
   martingale crosses the Ville ceiling, the alarm fires, and that alarm is
   what triggers recalibration."
8. **API tab → Send request.** "Every panel is this API. Nothing is mocked."
9. If asked about code quality: `python -m pytest -q` → 22 passed.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: fastapi` | `pip install -r requirements-api.txt` |
| `/api/health` says `ready:false` | Read the `error` field. Usually a missing dep. `python scripts/build_model.py` surfaces it clearly. |
| Console loads but every panel is empty | API unreachable. The header pill will say so. Check the port, and that you opened `localhost:8000` not `file://`. |
| Drift alarm never fires | Raise **Stream length** to 1200+. Detection needs a few hundred post-shift emails. Some generator pairs shift less than others — try `llama-3.3-70b → deepseek3.2`, which fires fastest. |
| Fonts look wrong | Google Fonts blocked by the network. Harmless; the fallback stack is monospace + system sans. |
| `xgboost` fails to import on Linux | `apt-get install libgomp1` |
| Port 8000 busy | `PORT=8080 ./scripts/serve.sh` |
| Render build times out | It installs xgboost from a wheel; if the free builder is slow, retry once. |
| Render app returns 502 on first hit | It is waking from sleep. Wait ~40 s and reload. |

---

## 9. What is still open, honestly

Say these before the panel finds them.

| Gap | Status |
|---|---|
| Real Zenodo corpus | Pipeline ready, data not dropped in. Section 6 above is the whole switch. |
| Raw email bodies / text-encoder arm | Blocked on data, not code. `neural.text_encoder_readiness()` reports this in code rather than asserting it in prose. |
| Extra generators via Ollama | Planned; `features.extract` already ingests raw text so the matrix widens without pipeline changes. |
| De-quantization mitigation | Measured, not yet fixed. Monotone constraints or integer bucketisation. |
| Cross-lingual subset | Not started. |
