# CN Edge–Fog Results

## Key findings

- The 500-image eval split gives always-local accuracy **0.848**, always-fog
  **0.914**, learned policy **0.870**, and oracle **0.926** on WiFi/5G.
- On WiFi/5G the learned policy costs **160.2/160.4 ms** per image versus
  **165.2 ms** for always-local and about **577/569 ms** for always-fog.
- On `poor_wifi` the learned policy keeps **0.870** accuracy at **173.6 ms**;
  always-fog costs **902.9 ms**.
- On `lpwan` the learned policy stays local at **165.2 ms** and **0.848**
  accuracy; always-fog costs about **65.1 s**.
- Policy validation action agreement with the oracle is **95.6%** on 800
  held-out image/profile rows.
- Real TCP validation passed **20/20** requests (5 images x 4 profiles) with
  **100% SHA-256 integrity** and **100% agreement** with direct fog inference.
- Mean total-latency error against the analytic model was **2.68%** (WiFi),
  **2.62%** (5G), **1.82%** (poor WiFi), and **3.43%** (LPWAN); maximum
  absolute error was **4.50%**.
- The demo shows local routing for a confident `SNE` image, offload on
  WiFi/5G/poor WiFi for a low-confidence image, and local routing on LPWAN.

## Outputs

| File | Contents |
| --- | --- |
| `summary.csv` | Per-profile/policy accuracy, latency, bytes, offload rate, cost |
| `raw_results.csv` | Per-image/profile/policy decisions |
| `policy_tree.joblib`, `policy_tree.txt` | Learned depth-4 policy |
| `policy_validation.csv` | Oracle agreement on the held-out policy split |
| `tcp_validation_summary.csv` | Real TCP and model-error summary |
| `tcp_validation_raw.csv` | Real TCP request timings and integrity checks |
| `figures/*.png` | Report figures |

## Exact commands

Run from `CN/`:

```bash
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.train_policy \
  --policy-size 1000 --eval-size 500

PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.run_policy_experiments

PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.validate_tcp \
  --images 5 --profiles wifi,5g,poor_wifi,lpwan

PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.demo \
  --image ../DATA/pristine/phase2_eval/00002785.jpg \
          ../DATA/degraded/phase2_eval/01441804.jpg

PYTHONDONTWRITEBYTECODE=1 ../DIP/training/.venv/bin/python -m pytest -q \
  -p no:cacheprovider
```

The full matrix uses cached predictions. The validation command uses the real
TCP path and a profile-delay proxy; LPWAN takes about four minutes.

## Caveats

- Main-matrix network conditions are analytic simulation assumptions.
- The validation proxy emulates bandwidth/RTT but not packet loss.
- `phase2_eval` is used, not the independent `phase2_test` split.
- Historical teammate CSVs, old figures, and caches are archived under
  `../../../DUMP/CN/`.
