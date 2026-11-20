# Hover explanation audit

Implemented on 11 November 2026. The tables record the copy direction; the shared implementation lives in `ui/tooltips.slint` and `ui/help-copy.slint`.

## Copy direction

Copy polish using Unslop by Damir Sokolovsky (CC BY 4.0), retaining the approved measurement-console design.

- Product: help people see what their hardware can actually do.
- Character: direct, calm, approachable.
- World: a hardware workbench with quiet controls and readable measurements.
- Moments: the measured numbers and graphs remain the focus.
- Signature: small circular hints explain the effect of a choice without moving the interface.

The audit found glossary-style definitions, internal terms, and descriptions that repeat their labels. Replace them with what the user will get, what will change, and any useful trade-off. Write for someone who has never used a benchmarking tool.

## Interaction and copy

- Replace the rectangular precision `?` button with a quiet, outlined 16px circle. Keep a larger invisible hit area, without changing label alignment or panel geometry.
- Show the explanation on hover after about 400ms, and on keyboard focus. Do not require a click; allow click/tap as a fallback.
- Use one shared tooltip treatment for buttons, labels, badges, and technical readouts. Overlay the page; never expand a row or move another control.
- Explain the outcome in everyday language: what happens when I use this, or what does this number tell me? Include the trade-off when it helps the decision.
- Prefer one natural sentence, or two very short sentences. Keep the rendered tooltip to one or two lines; adjust wording and width together rather than squeezing it into keywords.
- Avoid unexplained terms such as observations, realizations, execution metadata, inherited configuration, coalescing, and sampling policy. Technical control names can remain in the interface; the tooltip translates their effect.
- Do not promise that a setting will always make a test faster, more accurate, or more representative. Describe what changes without guessing the result.
- Tooltips dismiss on pointer departure, focus loss, Escape, scrolling, or navigation. Keep them within window bounds and prevent stale explanations after a hardware/selection change.
- Technical labels receive a circular hint where necessary. Buttons and obvious interactive segments explain themselves directly on hover/focus; do not add a circle to every control.
- Essential validation errors, destructive-action confirmations, selected settings, and unavailable-state reasons stay visible. Tooltips supplement those messages.
- Expose equivalent descriptions to assistive technology. Tooltip hit areas must not intercept clicks meant for a button, checkbox, slider, or chart.

## First pass: scope and interpretation

| Location / target | Proposed hover copy |
| --- | --- |
| Precision hint | Formats your hardware cannot run are left out of this list. |
| Curated lock | These tests stay fixed so results are comparable. Create your own set to change the settings. |
| Test-set row | See which tests are included and how they will run. |
| Run on a user-created set | Runs the settings you last saved. Save any changes first to include them. |
| Add test | Choose another test and its settings before adding it to this set. |
| Add to set | Include this test in your set, ready to save with the others. |
| Apply changes | Keep these changes in your set. Save the set to use them next time. |
| Save set | Keep this collection of tests and settings so you can run it again later. |
| Remove | Take this test out of the set. Save the set to keep the change. |
| Delete set | Remove this saved set. Results from tests you already ran will stay available. |
| Reset | Undo this test's custom settings and return to the settings it starts with. |
| Advanced count | Shows how many extra settings have been changed from their starting values. |
| Shorter run | Finish sooner, but collect fewer data points to compare. |
| Clear view | Empty this results page. You can still find saved runs in history. |
| Latest header | Shows the newest result recorded with these settings. |
| Typical header | Shows the middle result from runs with the same settings. |
| Best header | Shows the fastest result recorded with these settings. |
| Change | See how much the result changed from the run you are comparing against. |
| New badge | This result belongs to the current or most recent group of tests you ran. |
| Observations tab | See the individual readings to check how performance changed during the test. |
| Technical tab | See the exact settings and hardware information recorded with this result. |
| Standard deviation percentage | Shows how much the readings varied. A smaller percentage means steadier results. |
| Observed range | See the lowest and highest readings recorded during this test. |
| Dataset-loss percentage | See how much speed dropped with the largest dataset compared with the small reference test. |

For the precision hint, name the unavailable choice and practical limitation when known: "FP64 cannot run on this GPU." If only one data placement is unsupported, say that instead. A general fallback may say that unsupported formats are left out. Do not paste the current category-wide diagnostic paragraphs into a tooltip. Distinguish unsupported precision from an unsupported memory or execution path. Retain access to longer diagnostics in technical details rather than silently discarding them.

## Second pass: technical test settings

| Target | Proposed hover copy |
| --- | --- |
| Precision selector | Changes how numbers are represented, which can affect speed and accuracy. |
| Data placement | Choose where the data lives to see how memory affects performance. |
| Registers | Keep the data close to the calculations to focus on computing speed. |
| GPU memory | Use data stored in graphics memory to see how memory access affects computing speed. |
| Dataset sizing / Data preset | Choose a data size yourself, or let the app size it for your hardware. |
| One dataset / Sweep to target | Test one data size, or try increasingly larger sizes to see where performance changes. |
| Dataset size | Larger datasets can reveal slowdowns that small tests miss. The test may need extra memory too. |
| Physical CPU cores | Choose how many CPU cores work on this test at the same time. |
| Access pattern | Change the order data is read to see how well your hardware handles less predictable access. |
| Scattered GPU lanes | Make nearby GPU workers read from different places, which can make memory access less efficient. |
| Latency working set | Choose how much memory the test reads through. Leave empty to let the app choose. |
| Workload seed | Use the same number to recreate the same randomized data and access order. |
| Fixed / Fresh seed | Repeat the same randomized test each time, or generate a different version for each run. |
| Realizations | Try different versions of randomized data to see how much the chosen pattern affects results. |
| Access block target | Choose how much nearby data is handled together before the test moves elsewhere. |
| Measurement duration | Run each version of the test for longer to measure performance over more time. |
| CPU / GPU activity | Give the hardware brief pauses, or let it work continuously at 100%. |
| RAM / VRAM budget | Limit how much memory the test may allocate. Your hardware may allow less. |
| DEFLATE corpus | Choose data that is easier or harder to compress and see how that changes speed. |
| Compression block | Split the input into smaller or larger pieces that are compressed separately. |
| Compression level | Spend more effort compressing the data, which can produce smaller files but take longer. |
| Sieve range | Search for primes up to this number. Higher limits give each CPU worker more work. |
| CPU matrix shape | Change the dimensions of the number grids to compare different calculation shapes. |
| CPU matrix reuse | Repeat the full calculation more times before moving on to the next pass. |
| K block | Break the calculation into smaller or larger chunks to see how that affects speed. |
| GPU matrix tile | Change the size of the small number grids processed together by the GPU. |
| GPU matrix reuse | Do more calculations on each small grid before moving to another one. |
| GPU matrix traversal | Change which small grid the GPU visits next to test less predictable memory access. |
| Copy implementation | Compare copying with GPU code against the graphics driver's built-in copy operation. |
| Transfer staging buffers | Use more temporary buffers to give transfers more room to overlap. |
| Transfer submission | Send several copies together, or wait for each copy to finish before continuing. |
| Copies per submission | Send more copies together to spread the cost of starting a transfer across more work. |
| System RAM share | Put more of the test's input and output in system memory instead of graphics memory. |
| Traffic rate | Push the hardware as fast as possible, or see how it handles a chosen level of demand. |

Use option-specific text only while that option is relevant. Do not imply that randomized access guarantees cache eviction, registers eliminate all overhead, or a memory budget equals total application memory use.

## Third pass: results, history, sharing, and hardware

| Target | Proposed hover copy |
| --- | --- |
| Memory: Latest / Typical / Change | Switch between your newest speeds, the middle results across runs, and how much speeds changed. |
| Show best matching run | Show the fastest saved result measured with the same settings. |
| Compare with | Pick an earlier run to see how your current results have changed. |
| Comparison differences | Check what changed between these runs before judging the difference in performance. |
| Runs column | Shows how many runs are grouped under these test settings. |
| Result timestamp | Shows when this result was recorded. Times are shown in UTC. |
| TOPS label | Shows how much calculation work finishes each second. Higher means more work done in this test. |
| GB/s label | Shows how much data moves each second. Higher means faster transfer for this operation. |
| ns per access label | Shows how long the hardware waits for each memory read. Lower means a shorter wait. |
| Reference line on dataset graph | Shows the speed measured with a small dataset, so you can compare larger ones against it. |
| Shaded observed range | See how far readings spread at each data size, from the lowest to the highest. |
| Observation offset | Shows how far into the test this reading was taken. |
| Active / elapsed | Compare time spent doing measured work with the total time that passed. |
| Realization picker | See the readings from a different version of the randomized test. |
| Partial samples | See the readings collected before this run stopped. |
| History export checkbox | Include this run when saving the selected results to a file. |
| Compare with this run | Use this run as the starting point for comparing your current results. |
| Pin as baseline | Mark this run as a reference for future comparisons. |
| Archive run | Move this run out of the usual history list. You can bring it back later. |
| Native JSON import/export | Save or load complete test records, including every recorded data point. |
| Import snapshot | Load saved comparison results from an original Gluj-Bench image. |
| Share results | Create an image of your chosen results that others can view and use for comparisons. |
| Select highlights | Let the app pick which results to show in the image. |
| Embed all completed results | Let others compare every completed result, even those not shown in the image. |
| Hardware Refresh | Check your hardware again and update which tests are available. |
| Hardware Technical properties | Look up the extra hardware details reported by your system. |
| RAM channel/bus readout | A wider memory channel can carry more data at once. This width is per channel. |
| Cancel test | Stop this test and continue with the remaining tests. |
| Stop all tests | Stop the current test and cancel all tests still waiting to run. |

Existing chart-point hover already shows the series, value, and copy action. Preserve it. Add explanations to chart labels or legends where useful; do not place a second tooltip over the data-point tooltip.

## Keep simple labels simple

No additional explanations for Back, Close, Previous, Next, ordinary search fields, or navigation page names. Do not explain every whole panel with an oversized hover area. Prefer clearer labels before adding a tooltip to a vague label.

## Behavior to resolve before writing promises

- Reset currently removes the workload override and restores inherited/automatic behavior; it is not necessarily a factory reset of every editor field.
- New-set Add test currently starts from `AppSettings::default()`. Settings-page values must not be described as defaults for all new tests without reconciling that data flow.
- Change is a signed numeric ratio. A negative latency change is an improvement; avoid universal green-up/red-down explanations.
- Typical is a median across matching run metrics, distinct from observations within one run. Best also respects metric direction.
- Unsupported-operation text currently describes the category. Build concise, relevant wording from actual capability information before attaching it to Precision.
- Verify baseline pinning's exact listing/persistence behavior before promising it locks or automatically selects a comparison. Pinning does not imply an immutable measurement.

## Implementation sequence

1. Shared hover/focus overlay and circular hint, with fixed geometry and concise copy limits.
2. Precision hint, curated lock, set-edit/save scope, Reset, Shorter run, and Clear view.
3. Result definitions, comparisons, native files versus PNG snapshots, and queue actions.
4. Technical editor labels, then hardware/settings where clarification still helps.
5. Verify pointer and keyboard behavior, disabled-control explanations, clipping at 980x680 and 1280x850, scrolling dismissal, and no interference with chart-point copying or normal clicks.

Source scan: `ui/app-window.slint`, `ui/theme.slint`, `src/benchmark_setup.rs`, `src/saved_tests.rs`, `src/main.rs`, `src/presentation.rs`, `src/phase_eight.rs`, and `src/snapshot.rs` under `crates/gluj-bench-ui`.

## Copy review before implementation

Read each tooltip without its technical label. Could someone unfamiliar with benchmarking explain what will change? If not, rewrite it. Do not replace one specialist word with another, repeat the button label, or make the reader work out the trade-off. Keep exact units and formulas in detailed results where they are needed.

Reset wording, capability reasons, and the RAM bus explanation remain conditional on the behavior checks above. A friendly sentence must still be technically true.

## Implementation notes

- Shared overlay with 400ms hover delay, keyboard focus, Escape dismissal, and window-edge clamping. Native buttons and dropdowns keep their normal input behavior; help tracks their containing areas.
- Circular hints cover test settings and result statistics. Plain-language action help covers set management, history menus, sharing, comparison controls, and stopping/skipping tests. Obvious navigation stays quiet.
- Popup menu hints appear beside the menu. Scrolling, resizing, changing panels/hardware, or activating controls dismisses stale help.
- Existing chart-point readouts and click-to-copy behavior remain separate. Saved results and measurement behavior are unchanged.
- Full unavailable-test reasons remain in hardware Technical properties for the selected device/category. Reset restores the existing workload defaults; no new promise about global Settings applying to every new test is made.
- Slint's build-time UI compilation uses a 16 MiB worker-thread stack to accommodate the additional shared control wrappers on Windows.

Verification: 168 enabled UI tests passed, plus separate offscreen hover/focus/menu/click checks and the supported-size gallery check. Strict UI Clippy, formatting, and whitespace checks passed. Desktop verification remains with the user.

Optimized workspace build passed. `build/current/` contains the verified executables and provenance `c66a42c-source-d4377fb2a0f8`. All 80 saved-data files were hash-verified unchanged. Release ZIPs, publishing, and GitHub were untouched.

12 November follow-up: Precision now lists unavailable formats for the selected GPU, grouped by a shared reason. CPU Variant lists unavailable instruction modes, including AVX2/FMA, and remains visible when a single scalar variant is left. Workload hints identify fully unavailable workload families. Reasons distinguish missing hardware/driver features, timing or memory-access limits, and test paths the app cannot measure yet. No unsupported option is made runnable.

12 November verification: 171 enabled UI tests, both separate offscreen checks, strict UI Clippy, formatting, whitespace, and the optimized workspace build passed. Current binary hashes match the release artifacts; all 80 saved-data files are unchanged. Revision: `c66a42c-source-d4377fb2a0f8`. No release ZIP, publishing, commit, or push was performed.

12 November clarification: Support hints now show only `Unsupported:` and one name per line, without reasons. GPU matrix lists use the selected device's reported format capabilities; FP8/sparse app placeholders and storage/timing restrictions are not labelled as unsupported GPU formats. Missing capability data is not treated as proof of no hardware support.

17 November resumed verification: 177 enabled UI tests, both separate offscreen checks, strict UI Clippy, formatting, whitespace, and the optimized workspace build passed. The gallery test was adjusted to the current row width; a display-text loop counter was changed to enumeration to satisfy Clippy without changing truncation behavior. `build/current/` now matches source revision `c66a42c-source-0b92a74bd1da`. All 80 saved-data files are unchanged. Nothing was published, committed, or pushed.

## 20 November: dropdown and chart-label polish

Focused Unslop pass using the existing measurement-console direction. Product: measured hardware throughput. Character: precise, calm, approachable. World: graphite instruments. Moments: measured numbers and charts. Signature: a small brass selection rule. Unslop by Damir Sokolovsky ([@heydamir](https://x.com/heydamir)), CC BY 4.0.

Audit:
- The dataset comparison was constrained to a 132px field, clipping its reference text.
- Dropdowns used stock Fluent surfaces, padding, and selection styling instead of the app's shared theme.
- Hoverable text did not explicitly fill its hit area, allowing short labels to drift toward the center.

Changes:
- The dataset comparison gets a full-width, wrapping line below the chart title; the hover label grows with its text.
- Every shared selector uses 36px option rows, 12px text insets, graphite surfaces, quiet hairlines, and a brass marker on the current value.
- Menus show up to eight rows with scrolling, open above controls near the window bottom, and close if their anchor moves or width changes.
- Arrow/Home/End keys browse options, Return confirms, and Escape cancels. Hovering or browsing does not change the saved choice. Full option rows accept clicks, and controls retain accessible names and selection state.
- Copy, typography, charts, test options, and benchmark measurements remain within the existing product direction. No new visual assets or decorative motion.

Verification: 177 enabled UI tests, both separate offscreen checks (including dropdown keyboard confirmation/cancellation and full-row selection), strict UI Clippy, formatting, whitespace checks, and the optimized workspace build passed. The final narrow-window preview shows the comparison text in full and aligned with the chart title. `build/current/` matches revision `c66a42c-source-56610732e53c`; both executable hashes match the release artifacts, and all 80 saved-data files are unchanged. No new app build folder, release ZIP, publication, commit, or push was created.

## 20 November: expanded test-settings clipping

Shared fact rows now allocate a consistent label column and measure each wrapped value at its actual available width. Each row grows to the taller of its label and value, with a 20px minimum, so long workload names remain visible and cannot overlap Dataset or Sampling. The supported-size offscreen gallery now opens an included test's settings in the narrow sidebar. The original application logo is unchanged; logo studies remain separate preview assets.

Verification: 177 enabled UI tests and both separate offscreen checks passed. The 980px-wide rendered preview shows the full wrapped workload value and clearly separated labels. Formatting and whitespace checks passed. Strict UI Clippy and the optimized workspace build also passed. `build/current/` now matches revision `c66a42c-source-83e258f6f706`; both executable hashes match the release artifacts and all 80 saved-data files are unchanged. No release ZIP, publication, commit, or push was made.
