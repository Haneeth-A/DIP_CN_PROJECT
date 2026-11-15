# CN Edge–Fog Offloading — 4 Slides

## Slide 1 — Motivation and architecture

**Title:** Network-aware offloading for WBC classification

- Local INT8 EfficientNet-B0 is fast; the fog ensemble is more accurate.
- Decision per image/profile: stay local or offload over the existing TCP protocol.
- Inputs: 7 image-quality features, local confidence/entropy/margin, RTT, bandwidth, loss.
- Output: one policy action; the transport and result schema are unchanged.

```text
quality + local prediction + network state
                    |
                    v
             learned policy
              /          \
          local          offload
            |               |
       edge result     TCP -> fog -> result
```

**Speaker note:** The contribution is the decision layer, not a new protocol.
LPWAN is protected by a local-only bandwidth guard.

---

## Slide 2 — Profiles and learned policy

**Title:** Profiles, features, and decision tree

| Profile | RTT | Bandwidth | Loss |
| --- | ---: | ---: | ---: |
| WiFi | 10 ms | 50,000 kbps | 0.5% |
| 5G | 20 ms | 100,000 kbps | 0.5% |
| Poor WiFi | 200 ms | 2,000 kbps | 3% |
| LPWAN | 5,000 ms | 5 kbps | 5% |

- Depth-4 decision tree trained on policy-train oracle actions.
- Oracle cost: `latency + 1000 ms * (1 - correct)`.
- Validation action agreement: **95.6%**.
- LPWAN guard: never offload below 1,000 kbps.

**Speaker note:** The tree is deliberately shallow for interpretability and to
avoid overfitting the 1,000-image policy split. The true label is never an
input at decision time.

---

## Slide 3 — Accuracy, latency, and offload behavior

**Title:** Learned policy approaches the oracle without fog-level cost

| Profile | Learned acc | Learned cost | Local cost | Fog cost |
| --- | ---: | ---: | ---: | ---: |
| WiFi | 0.870 | 160.2 ms | 165.2 ms | 578.0 ms |
| 5G | 0.870 | 160.4 ms | 165.2 ms | 575.0 ms |
| Poor WiFi | 0.870 | 173.6 ms | 165.2 ms | 902.9 ms |
| LPWAN | 0.848 | 165.2 ms | 165.2 ms | 65.1 s |

- Local accuracy: **0.848**; fog accuracy: **0.914**; oracle: **0.926/0.916**.
- Learned offload rate: about **3.6%** on WiFi/5G/poor WiFi, **0%** on LPWAN.
- Figures: `accuracy_vs_latency.png`, `offload_rate_by_profile.png`,
  `latency_breakdown.png`, `policy_vs_oracle.png`.

**Speaker note:** The policy gains accuracy over always-local on fast networks
while avoiding the much larger fog latency. On LPWAN, local is the clear win.

---

## Slide 4 — Real TCP validation, demo, and limits

**Title:** End-to-end validation and handover

- Real TCP: 5 images x 4 profiles = **20/20 requests** passed.
- **100%** SHA-256 integrity, **100%** match with direct fog inference.
- Mean total-latency error vs analytic model: **1.82–3.43%**; maximum **4.50%**.
- Demo: confident image stays local; low-confidence image offloads on
  WiFi/5G/poor WiFi, stays local on LPWAN.
- Handover: `policy_tree.joblib`, `summary.csv`, validation CSVs, figures,
  demo, report, and tests are in place.

**Speaker note:** Validation uses a local proxy that emulates bandwidth and
RTT. It does not emulate packet loss. `phase2_test` remains untouched for the
independent final evaluation.

---

## Suggested figure placement

1. Architecture slide: no figure required.
2. Profiles/policy slide: `results/cn_edge_fog/figures/offload_rate_by_profile.png`.
3. Results slide: `results/cn_edge_fog/figures/accuracy_vs_latency.png` and
   `results/cn_edge_fog/figures/latency_breakdown.png`.
4. Validation slide: `results/cn_edge_fog/figures/policy_vs_oracle.png` or the
   validation summary table.
