# Product requirements

This document states what `dicom-stream-triage` has to demonstrate. The repository is a synthetic benchmark. A service built on the same rules would still not be a clinical device. Measured values are the seed-0 run in [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md).

## 1. Problem

When a chest study arrives, decode and windowing sit in front of inference. A sensitive nodule score draws boxes on vessel crossings and ribs. Most of a screening list is normal, so an arrival-order queue leaves a pneumothorax in the middle of the list.

Those are the operational reasons a triage product exists. They are not, by themselves, evidence that this synthetic queue shortens a hospital turnaround time.

## 2. Users of this benchmark

- A local viewer that opens one study, a heatmap, and boxes.
- A batch path that reports decode time, box counts, and queue order.

## 3. Functional requirements

| Item | Requirement |
| --- | --- |
| Input | Synthetic computed radiography, uncompressed Explicit VR Little Endian, 16-bit, one frame |
| Decode | Compare sequential pydicom with `np.frombuffer` plus a threaded batch, under the same window |
| Boxes | Lung residual peaks. Confidence $`\geq 0.5`$ is the baseline. Spatial entropy $`\leq \tau`$ is the filter |
| Matching | Greedy box IoU $`\geq 0.3`$ |
| Study score | Pneumothorax from the ConvNeXt head. Nodule evidence from the residual peak, shifted so scores at or below 0.46 contribute 0 |
| Auto-pass | Pneumothorax < 0.25, nodule < 0.35, fused-distribution entropy < 0.50 nat |
| Order | Remaining studies by $`5 p_{\mathrm{ptx}} + 2 p_{\mathrm{nodule}} + 1.2 H`$ |
| Wait | One read is a 45-second slot. An auto-passed study does not consume a slot |

$`\tau`$ is chosen on the tuning split as the cut with the largest false-positive drop that does not lower precision and keeps at least 85% of baseline recall. Reported box counts freeze that cut on the held-out split.

## 4. Out of scope

- A cleared device, a sensitivity claim for a regulator, or a hospital policy.
- JPEG 2000, RLE, multi-frame objects, a live PACS association, and vendor weights.
- A 4k–8k mammography matrix. It would be the same decoder on a larger input. This run measures 2048 px.
- A claim that the nodule heatmap classifies a 5 px nodule. The head separates pneumothorax. Nodule evidence is the residual peak.
- Memory-mapped I/O. The fast path is a buffer view of the pixel element.

## 5. Acceptance

| Criterion | Target | Seed 0 |
| --- | --- | --- |
| Batch throughput | Threaded FPS above pydicom FPS | 69.73 > 36.65 (1.90×) |
| Single-study latency | Report sequential zero-copy | 24.93 ms (pydicom 27.28 ms) |
| Pixel agreement | Max absolute window error 0 | 0 |
| False-positive reduction | $`\geq 40\%`$ | 85.7% (147 → 21) |
| Precision | Filtered $`\geq`$ baseline | $`0.580 \geq 0.169`$ |
| Emergency wait | $`\geq 80\%`$ and zero auto-passed emergencies | 93.7% (3587.7 s → 225.0 s), 0/11 |
| Normal auto-pass | Report | 100% (132/132) |
| Nodule review | Report | 77.3% (17/22). Five nodules auto-passed |

Latency moves if the run is repeated. Box counts and queue counts are fixed for seed 0.

## 6. Auto-pass and review

An auto-passed study leaves the radiologist queue. An auto-passed pneumothorax is a miss. Seed 0 has zero misses.

Five nodules had residual peaks in the normal band ($`\leq 0.46`$), so their nodule evidence was 0 and they left with the normals. Lowering the nodule cap does not separate those five from normals on this score. All 15 vessel crossings stayed in review.

## 7. What a later service would still have to add

This repository does not implement the rows below. They are the boundary a deployment would have to state.

| Boundary | Contract |
| --- | --- |
| Ingest | One frame received by C-STORE. Zero-copy applies only when the transfer syntax is uncompressed Explicit VR |
| Score | Pneumothorax probability, nodule evidence, entropy, and an auto-pass or review flag |
| Compression | JPEG 2000 and RLE need their own decoders and are absent from this timing |

## 8. Documents

| File | Role |
| --- | --- |
| [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) | The measured run |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | What the code implements |
| [PAPER.md](PAPER.md) | The same run as a technical note |
