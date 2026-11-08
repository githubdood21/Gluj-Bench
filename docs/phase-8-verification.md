# Phase 8 - verification record

Local Windows checks performed with user-authorized PC access on 2026-11-02 and
2026-11-03. The application used an isolated qualification profile; original user
settings and history were not edited.

- Cache/VRAM naming: local-memory family and explicit Cache-oriented shader /
  Streaming / VRAM modes inspected. Separate results retain mode/path context.
- Results: actual 21-size GPU FP32 sweep opened; largest 256 MiB measurement was
  selected initially, with median, observed range, six observations, reference
  drift, chart and per-size text alternatives. Result and queue catalogues scroll.
- History: saved name/notes/pin survived restart; archive/restore and actual native
  JSON save/import retained six raw observations and ImportedComparison identity.
  Permanent deletion was covered by core regression and its UI confirmation guard;
  no permanent-delete click was performed during desktop verification.
- PNG: native Save As dialogs produced selected-ui.png (one embedded summary) and
  all-ui.png (13 embedded summaries with explicit Include all). PNG iTXt JSON was
  independently decoded to verify scope. Import/round-trip coverage also passes
  automated tests, including an original legacy PNG fixture.
- Plans: pending Skip persisted while other work continued; global Stop cancelled
  the active condition and skipped pending work. Cancel-one exposed a status bug
  fixed in Phase 9: continuing conditions must stay Pending/Running, not Skipped.
  The regression covers both continuation and global-stop semantics.
- Failure: terminating this test app's child worker during a run persisted Failed
  status without a successful score; Reconnect restored the engine.
- Layout: inspected standard 1280x850 and maximized 1920x1032 windows on the local
  display. Dialogs and standard selectors were exercised. Other DPI/scales, full
  keyboard traversal and assistive technologies were not exhaustively qualified.

See [Phase 9](phase-9-qualification.md) for final local candidate status and physical
hardware gaps. Measurements with variable performance never trigger rerun advice.

Final rebuilt executable verified interactively: cancelling one condition preserved
Cancelled/Running/Pending statuses; global Stop left Cancelled/Skipped statuses.
