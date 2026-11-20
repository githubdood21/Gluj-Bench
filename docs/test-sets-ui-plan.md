# Test sets UI plan

Agreed on 11 November 2026. This records the next redesign for implementation after context compaction.

## Model and layout

- A **test set** contains individual tests, each with its own workload, precision, and execution settings. A vector set can contain FP16, FP32, and FP64 tests.
- The middle **Test sets** panel lists curated sets and saved user-created sets in the selected category. Each entry has a Run action.
- The right panel shows the selected set's name and included tests. Measured output belongs on **Benchmark Results**.
- Retain the measurement-console styling, full-row selection, clear group headings, shared hardware selector, and stable panel geometry.

## Curated sets

- Curated sets are strictly read-only and carry a small lock icon.
- Users can run them and inspect their included tests and settings.
- Do not offer editing, renaming, adding, removing, Customize set, Save copy, or another cloning path for curated sets.
- Users must configure their own set to modify anything.

## User-created sets

1. **New set** in the middle panel starts a blank user-created set.
2. The right panel contains an editable set name, the included-test list, and **Add test**.
3. **Add test** opens the settings form within the same right panel. Choose workload, precision, placement, dataset, and other supported settings. Finish with **Add to set** or **Cancel**.
4. Each included test has **Edit** and **Remove** actions. Edit opens that test's settings and finishes with **Apply changes** or **Cancel**.
5. **Save set** persists the complete collection in the appropriate category. Subsequent saves update the same user-created set. Users can rename and delete their own sets.
6. Run from the middle panel executes the saved set. Preserve each included test's configuration and sampling policy.

## Simplification

- Remove the always-visible configuration form and precision dropdown from the set overview. Precision belongs to the individual test being added or edited.
- Replace the separate Planned tests list with the set's included-test list.
- Remove Build a batch, Add current test, and standalone Run custom test. The set is the runnable collection.
- Keep secondary technical settings inside the individual test editor. Use short labels and avoid duplicate summaries.

## Implementation order

1. Adapt persistence and selection to explicit user-created sets with independent included-test configurations. Keep existing saved user configurations usable.
2. Build the set overview and explicit add/edit/cancel flows. Enforce curated read-only behavior in callbacks as well as visible controls.
3. Replace obsolete batch/template-copy controls and update the user guide.
4. Verify category and GPU selection, correct vector/matrix loading, per-test precision, add/edit/remove, save/reload/rename/delete, cancelled edits, storage failures, and curated immutability. Check minimum-size layout and row selection versus Run.

## Implementation status

Implemented on 11 November 2026.

- [x] Preserve existing saved configurations in `benchmark-settings.json`.
- [x] Middle-panel Test sets library, New set, category filtering, and saved-set Run actions.
- [x] Strictly read-only curated selection and settings inspection, protected in callbacks; no copy route.
- [x] Explicit Add test / Add to set and Edit / Apply changes flows, with Cancel leaving included tests unchanged.
- [x] Set naming, save/update under the same identity, test removal, and set deletion.
- [x] Preserve each included test's configuration and sampling policy when saving and running.
- [x] Replace visible batch controls and the standalone custom-run flow with the set overview.
- [x] Retain shared GPU selection, full-row selection, group headings, and fixed right-panel width.
- [x] Update README and user guide.
- [x] Validate malformed saved settings before resolving; invalid entries remain removable and replaceable.

Verification: 168 enabled UI tests passed; the separate offscreen interaction/layout check passed at 1280x850 and 980x680. Tests cover persistence, rename/update/delete, cancelled and invalid edits, mixed sampling policies, category filtering, unsupported entries, storage failures, and curated immutability. Strict UI Clippy, formatting, and whitespace checks also passed. Desktop and hardware-run verification remain with the user.

The implementation retains the existing saved-file schema. Results remain on Benchmark Results. No release has been published or ZIP replaced, and these changes have not been committed or pushed.

Optimized workspace build passed. `build/current/` now contains matching release executables and provenance (`c66a42c-source-d4377fb2a0f8`). All 80 saved-data files were hash-verified unchanged. The existing release ZIP is untouched.

Follow-up polish: unsupported operations use a compact `?` popup beside Precision (Workload where no Precision selector is present). User-created sets show workload names and the same deduplicated datatype/test-count summary as curated sets.

17 November: Current local build refreshed to `c66a42c-source-0b92a74bd1da`, including the simplified hardware-only unsupported lists. Current UI validation: 177 tests and both offscreen checks passed. Saved data was hash-verified unchanged.
