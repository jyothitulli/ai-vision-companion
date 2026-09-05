# Evaluation

Script: `evaluation/evaluate.py`

Measures (when provided real prediction JSON):

- precision, recall, F1, map_proxy
- latency summaries if samples are included

System metrics to record after profiling, not to invent:

- detection / depth / end-to-end latency
- STT and spoken-response latency
- false-alert and missed-event rates for continuous assistance
- per-class precision/recall for specialized detectors

Demo plan: `docs/demo-scenarios.md`.
