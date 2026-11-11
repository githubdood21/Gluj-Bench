# Gluj-Bench user guide

## Run your first benchmark

1. Extract the Windows x64 ZIP into a writable folder. Open `gluj-bench-ui.exe` with `gluj-bench-worker.exe` alongside it.
2. Wait for **Engine ready**. **System Information** shows the hardware and capabilities the app detected.
3. Open **Benchmarks**, choose a category, and select the GPU adapter where applicable. The adapter selector applies to both templates and custom tests.
4. Choose a row in **Experiments** and click **Run**. Each template supplies its own workload, precision coverage, dataset policy, and extensive measurement settings. Rows show the minimum measurement time. Preparation, calibration, and sweeps can add time.
5. Open **Benchmark Results** and click a result row. Use the component picker to switch between CPU and GPU results.

During a run, **Cancel test** stops the active test and leaves the remaining queue in place. **Stop all tests** also clears the pending queue. Available observations from a cancelled test can be inspected in Run History.

## Configure a custom test

1. Choose the category and hardware at the top of **Benchmarks**. Use the **Custom test** panel on the right.
2. Select a **Workload**. For GPU performance, choose **Vector** or **Matrix**, then choose a supported **Precision** and **Data placement**. Registers measure compute throughput. GPU memory and supported system-RAM paths expose dataset controls.
3. Choose **One dataset** or **Sweep to target**, then set the size. Workload-specific options, such as matrix tiles, appear where applicable.
4. Open **Advanced** for measurement duration, seeds, activity targets, allocation budgets, and execution settings. Its count shows fields that differ from defaults. **Back** returns to setup, and **Reset** restores the selected test's defaults.
5. Review the buffer estimate and click **Run custom test**. **Shorter run** reduces observations for custom tests.

An experiment's **Customize** action copies its settings into the builder. Templates covering several configurations populate a batch. Template **Run** uses the template's fixed definition, independently of the custom form and its **Shorter run** setting. Saved template results include a template ID and version.

## Build a batch

Enable **Build a batch**, configure a test, and click **Add current test**. Change the workload, precision, or settings and add the next test. Each added entry keeps its own configuration while you edit. Review **Planned tests**, uncheck entries to skip, and click **Run batch**.

**List actions** contains comparison and clearing. Turning off **Build a batch** returns to a single test and retains the list for later. Selecting another GPU adapter applies that adapter to GPU entries in the batch, with compatibility checked before running.

**Unsupported operations** lists unavailable modes for the selected category and device. For supported CPU workloads, **Compare CPU core counts** opens the fixed-work comparison.

Sequential access is the standard choice. Custom workloads offer randomized access where supported. A fixed seed in **Advanced** reproduces the access order.

## Read the results

| Readout | Meaning |
| --- | --- |
| **TOPS** | Trillions of operations per second for the selected workload. A fused multiply-add counts as two operations. |
| **GB/s** | Billions of bytes per second. Memory read, write, and copy are separate measurements. |
| **ns per access** | Time per dependent access. Lower values indicate lower latency. |
| **Median and observed range** | The middle measurement and the lowest-to-highest values recorded across observations. |

On a compute sweep, the horizontal axis shows dataset size. Hover a point to see its exact value and click it to copy the value. A dashed line marks the compute reference when available. A percentage beside the graph shows the measured drop at the largest dataset when that comparison applies.

Compare matching workloads, precision, sizes, and settings. Use **Compare with** to choose a saved baseline. For per-dataset data and individual observations, open **Details & measurements**.

## Keep and share measurements

- **Run history** opens saved and cancelled runs. Select a row to open that measurement. Its **Actions** menu lets you edit notes, pin a baseline, or archive the run.
- **Share results** creates a PNG preview. Check the results you want in the image, then use **Save PNG**. The original PNG also carries comparison summaries that can be loaded with **Import snapshot**.
- For full raw observations, check runs in **Run history** and choose **Files → Export native JSON**. Use **Import native JSON** from the same menu to load records for comparison.
- When updating or moving the app, keep `run-history/`, `results.json`, `settings.json`, and `benchmark-settings.json` beside the executables.
