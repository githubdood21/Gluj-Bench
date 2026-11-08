# Gluj-Bench release playtest

8 November 2026 · Windows production build 0.5.0-rc.1 · Revision `55d7a87-source-0beae1af0e60`

Small desktop pass on a Ryzen 9 5950X and Radeon RX 7900 XTX at the default window size and 100% UI scale. I used an isolated copy of saved CPU and GPU results, plus a live CPU memory run. The playtest covered startup, navigation, the curated plan and queue, results and history, a malformed import, Settings validation, and the snapshot preview. Saved GPU sweeps covered the results UI. Live GPU throughput, other display scales, long sessions, and export round trips remain outside this pass.

## Fix before release

| Finding | Reproduction and expected behavior |
| --- | --- |
| **Opening a GPU run leaves the component picker on the CPU.** | From a CPU result, open **Run history**, select a saved GPU FP32 run, and click **Open result**. The GPU table and TOPS graph load, but **Component** still shows the Ryzen CPU and highlights it in the dropdown. Selecting the Radeon manually corrects the picker. Keep the picker, table, and selected result synchronized when opening from history. |
| **Opening a legacy result counts it twice for the session.** | With the same legacy FP32 result already present in Results, open its entry in Run History. The completed count goes from 4 to 5 and the FP32 row shows 2 runs. After closing the app, the saved results file still contains 4 results and one FP32 entry. Resolve the history entry to the existing result when they represent the same measurement. Add a regression check that opening a result preserves counts and typical values. |
| **History search misses the displayed test name.** | With **L1 cache bandwidth** visible under **All runs**, enter that exact title in **Search tests, names or notes**. The list returns zero matches. Clearing the query brings it back. Index the same friendly title the row uses, alongside custom names, notes, and IDs. |

## UI and style polish

| Area | Observed | Suggested change |
| --- | --- | --- |
| **Plan and test details** | The configuration header prints `Available | bytes | bytes/s`. A planned row exposes `cpu.bandwidth.cache.l1`, `access block bytes`, and a raw access label. The collapsed curated row says `sequential · realizations 1`. | Use **Ready**, **Read/write/copy bandwidth**, and **Sequential** as applicable. Show the size, core count, and observation target in Details. Keep IDs and raw option keys in Technical or native export, and omit the default single-realization count. |
| **Observations and import errors** | Partial samples starts with `Cache l2:Copy:payload-1677312:streaming-false` and a raw seed. A single-point series repeats the same value on the y-axis. A truncated JSON import shows `EOF while parsing a list` below the inspector and is out of sight with the chart open. | Use a label like **L2 copy · 1.60 MiB**, and expose the seed under Technical or for explicit custom seed inspection. Give single-point plots one readable value or usable axis bounds. Put **Couldn’t import this run file** beside **Files**, with parser details available on demand. |
| **Results copy** | A first run offers **Comparison differences**, which opens to **No matching workload**. A saved result reports **+0.0% higher throughput**. The footer can show `Full native run saved in run-history.` | Use **No matching baseline** as a quiet state and suppress the differences button in that case. Use **No change** for a rounded-zero delta between distinct runs. Put routine storage details in Run History or Technical and keep errors prominent. |
| **Run controls** | During a curated run, the queue row shows **Cancel test**, the footer shows **Cancel benchmark**, and the sidebar shows **Stop all tests**. | Give the active-test action one consistent name and one visible location. Keep **Stop all tests** distinct and reachable from any page. The existing run/cancel behavior worked. |
| **Spacing and hierarchy** | System Information repeats readiness in the scan line, hardware card, and sidebar. The processor card has a large empty tail. The snapshot checklist floats vertically in its panel, and the preview has large empty bands. The snapshot header refers to native JSON while its actions are for images. | Use the compact Settings spacing as the reference. Size hardware cards to content, reduce repeated readiness signals, top-align the snapshot checklist and preview, and describe the PNG’s embedded comparison data next to its checkbox. |
| **Small text and keyboard labels** | Result dates, row metadata, and sidebar status are substantially quieter and smaller than their surrounding content. Windows accessibility identified the Dataset size control as a numeric edit with a value and no name. | Use a consistent secondary type size and contrast for dates and statuses. Give settings inputs accessible names that match the visible labels. |

The brass accent already makes the main Run action easy to find. The compact Settings groups, row selection in Run History, and the TOPS graph with its single reference-gap number are good patterns for the remaining polish.

## What held up

- The first native window handle appeared after **0.43 s** in one launch with up to 50 ms polling. This measures window availability. Navigation and result inspection felt responsive during the pass.
- The app and worker showed **no measurable CPU-time increase** during one roughly 3-second idle home sample and one roughly 4-second idle Results sample. UI working set was about **101 MiB** at home and **158 MiB** after the playtest on Results. Worker working set stayed around **21 MiB**. These are spot checks on this machine.
- The live L1 test completed in **6.97 s**. **Stop all tests** kept the completed result, cancelled the active L2 test, and cleared the queued remainder. History showed the cancelled run, and a separate saved cancelled run opened its partial samples.
- Truncated native JSON was rejected with the six history entries intact. A dataset size of `0` showed a validation message and left the saved valid size at `256 MiB`. The saved GPU sweep displayed its TOPS chart and **-5.7% at 256 MiB**. The snapshot preview rendered.
- No crash or hang occurred during these checks. The isolated processes exited and all 80 existing settings, history, and results files checked matched their pre-playtest hashes.
