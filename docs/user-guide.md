# Gluj-Bench user guide

## Run your first benchmark

1. Extract the Windows x64 ZIP into a writable folder. Open `gluj-bench-ui.exe` with `gluj-bench-worker.exe` alongside it.
2. Wait for **Engine ready**. **System Information** shows the hardware and capabilities the app detected.
3. Open **Benchmarks**, choose a category and family, then select **Curated**. Review **Tests to run** and uncheck any tests you want to skip.
4. Click **Run curated**. The plan shows minimum measurement time and estimated test buffers. Preparation, calibration, and dataset sweeps can add time.
5. Open **Benchmark Results** and click a result row. Use the component picker to switch between CPU and GPU results.

During a run, **Cancel test** stops the active test and leaves the remaining queue in place. **Stop all tests** also clears the pending queue. Available observations from a cancelled test can be inspected in Run History.

## Configure a custom sweep

1. Choose **Custom** and pick the workload variant you want to measure. For GPU workloads, select the adapter under **Target hardware**.
2. Enable **Customize settings** or the per-test override where offered. Controls depend on the workload and can include CPU cores, precision, access order, memory placement, and activity target.
3. For a supported compute workload, choose a buffered preset and **Sweep to target** to measure a range of dataset sizes. Use **One dataset** to focus on a single size. The plan summary helps you check buffer requirements.
4. Click **Add to list** to queue that configuration with others. Review the checklist, then click **Run custom**. An empty custom list runs the currently configured test.

Sequential access is the standard choice. Custom workloads can use randomized access where supported. A fixed seed lets you reproduce the same access order. **Shorter run** reduces the number and duration of observations.

## Read the results

| Readout | Meaning |
| --- | --- |
| **TOPS** | Trillions of operations per second for the selected workload. A fused multiply-add counts as two operations. |
| **GB/s** | Billions of bytes per second. Memory read, write, and copy are separate measurements. |
| **ns per access** | Time per dependent access. Lower values indicate lower latency. |
| **Median and observed range** | The middle measurement and the lowest-to-highest values recorded across observations. |

On a compute sweep, the horizontal axis shows dataset size. Hover a point to see its exact value and click it to copy the value. A dashed line marks the compute reference when available. A percentage beside the graph shows the measured drop at the largest dataset when that comparison applies.

Compare matching workloads, precision, sizes, and settings. Use **Compare with** to choose a saved baseline. For per-dataset data and individual observations, open **Benchmark details & measurements**.

## Keep and share measurements

- **Run history** opens saved and cancelled runs. Select a row to open that measurement. Its **Actions** menu lets you edit notes, pin a baseline, or archive the run.
- **Share results** creates a PNG preview. Check the results you want in the image, then use **Save PNG**. The original PNG also carries comparison summaries that can be loaded with **Import snapshot**.
- For full raw observations, check runs in **Run history** and choose **Files → Export native JSON**. Use **Import native JSON** from the same menu to load records for comparison.
- When updating or moving the app, keep `run-history/`, `results.json`, `settings.json`, and `benchmark-settings.json` beside the executables.
