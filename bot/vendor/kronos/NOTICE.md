# Vendored: Kronos (MIT licence)

Source: https://github.com/shiyu-coder/Kronos (`model/` folder, master branch, fetched 2026-11-08).
Paper: "Kronos: A Foundation Model for the Language of Financial Markets" (arXiv:2508.02739).
Weights are downloaded at runtime from Hugging Face (NeoQuasar/Kronos-small, NeoQuasar/Kronos-Tokenizer-base).

Used ONLY in shadow mode by `bot/kronos_worker.py`: forecasts are recorded and scored,
never used to place, size or exit trades (AGENTS.md guardrail 9 and 15).
Unmodified copy; keep it that way so upstream fixes can be re-vendored cleanly.
