# Phase 7 - local confirmations passed

The user reported the confirmations passed on 2026-11-02. The Cache/VRAM UI
clarity concern is addressed in Phase 8. The original checklist follows for reference.

Live GPU execution and desktop access were deliberately left to you. Start
`build/current/gluj-bench-ui.exe`; keep the worker beside it. Shorter coverage is
sufficient for initial functional checks. Please report a pass/failure (and the
saved result or error for failures) for each applicable item:

- **Memory controls/results:** choose a GPU memory family, Single 8 MiB, then Sweep
  to 32 MiB. Check separate read/write/copy values by size, spread/sample counts,
  allocation and path labels. Try sequential native copy and shuffled shader copy;
  native irregular combinations should be unavailable/rejected. Curated preview
  should contain separate equally covered sequential/shuffled conditions.
- **Cache discovery:** run Automatic cache once. Check fresh device/driver/time
  provenance and either transition intervals with heuristic drop strength or an
  explicit no-tier finding with measured results. No probability-confidence claim
  or recommendation to rerun should appear.
- **Host transfer:** sweep to 32 MiB with Both directions. Check separate H2D/D2H
  rates and labeled mean request costs. Exercise a 3-slot batched configuration
  and an 8-slot serial configuration; serial uses one copy per submission. Mean
  cost must not be labeled an individual-request latency percentile.
- **Matrix, if supported:** test buffered FP16 and INT8 where available. Select an
  advertised native tile, reuse 8 then 16, and shuffled traversal. Check successful
  output validation, actual tile/input/accumulation labels and tile-streaming
  wording. Register mode should hide incompatible traversal controls.
- **Sustained/Stop and persistence:** try a 15-second supported condition and stop
  a longer one partway through. Check responsive Stop and truthful partial status.
  Restart the app and confirm existing settings/history plus new results survive.

Unsupported GPU features should remain unavailable rather than fail as advertised
supported choices. Cross-vendor/low-memory qualification remains a separate Phase 9
requirement. This checklist does not ask you to rerun because measurements vary.
