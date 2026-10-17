# Technical report

Seed 0 of `dicom-stream-triage`, one CPU run. The cohort is synthetic. The numbers are that run, not a measurement of a clinical device.

## 1. Studies

Each image has two lung ellipses, a mediastinum, ribs, thin vessels, and one finding. The finding is a nodule, a vessel cross, or an apical pneumothorax in one lung. A vessel cross is not a nodule and does not receive a ground-truth box.

Stored pixels are \(200 + 3500 \cdot x\) with \(x \in [0, 1]\). The window is center 1950 and width 3500, which recovers \(x\). The DICOM file is Secondary Capture, Explicit VR Little Endian (`1.2.840.10008.1.2.1`), MONOCHROME2, 16-bit. The patient name is `SYNTHETIC`.

The box experiment draws equal thirds of nodule, vessel, and normal: 30 each. Tuning uses 90 studies at seed 11. Evaluation uses a different 90 at seed 29. The worklist of 180 studies at seed 41 was drawn from a mix near 72% normal, 12% nodule, 8% vessel, and 10% pneumothorax. The draw was 132 / 22 / 15 / 11. Training uses 160 studies with the four kinds balanced.

## 2. Decode

The baseline is `pydicom.dcmread`, `pixel_array`, then the window. The fast path scans the file bytes for the Rows `(0028,0010)`, Columns `(0028,0011)`, and Pixel Data `(7FE0,0010)` tags and views the frame with `np.frombuffer(..., dtype="<u2")`. That array aliases the file buffer. The float window allocates. The threaded path runs the same function on 8 workers.

Before the timed section, one study is windowed both ways and the maximum absolute difference is checked. At seed 0 it was 0.

| Path | Mean | FPS |
| --- | --- | --- |
| pydicom sequential | 27.28 ms | 36.65 |
| zero-copy sequential | 24.93 ms | 40.10 |
| zero-copy, 8 threads | 14.34 ms (wall / 8) | 69.73 |

The sequential gain is 1.09×. On a 2048×2048 frame the window and the file read dominate the parser. pydicom 3 already exposes an uncompressed frame as a NumPy array. The 1.90× threaded gain is that window running concurrently. Do not quote 14.34 ms as the latency of one click. That figure divides a batch.

JPEG 2000 and RLE are not in the timing. A pure-Python RLE decoder was slower than pydicom's RLE on this machine, so it was not shipped. The fast path is not `mmap`.

## 3. Boxes

A proposal is a local maximum of the residual after subtracting a blur with \(\sigma = 6\), inside the lung mask. The mask is the generator ellipse, not a segmentation model. Confidence is \(s = r / (r + 0.35)\). A score of 0.5 is a residual peak of 0.35. On this generator the strongest normal peak is about 0.39, and nodule and cross peaks are about 0.61, so 0.5 sits between them.

Spatial entropy is the Shannon entropy, in nats, of a 36×36 residual window summed onto a 6×6 grid. Per-pixel entropy did not separate nodules from crosses: both contain a similar number of bright pixels. After pooling, a nodule sits in the central cells and a cross spreads along a row and a column. `self_check` reports 1.389 nats for a compact Gaussian and 2.881 nats for a cross.

τ = 2.578 is the entropy quantile, among 19 on the 90 tuning studies, with the largest false-positive drop among cuts that keep precision at or above baseline and recall at or above 85% of baseline. On the 90 held-out studies, false positives fall from 147 to 21, precision rises from 0.169 to 0.580, and recall falls from 1.000 to 0.967. One of 30 true boxes is dropped. The IoU threshold is 0.3. A detection box is 24 px on a side. A ground-truth box is 2.6 times the nodule σ.

The ConvNeXt head does not produce these boxes. It supplies the pneumothorax probability used by the queue and the heatmap drawn on the board.

## 4. Head and queue

`PatchDetector` uses a stride-2 stem and one more stride-2 stage, so the output stride is 4. Widths are 16 then 32. Each block is a 7×7 depthwise convolution, GroupNorm, GELU, and a pointwise expansion of 2. Heatmaps are nodule and pneumothorax logits. The classifier is linear on concatenated global-max features and heatmap peaks. Training is 160 studies, 16 epochs, AdamW. The final training loss is 0.075.

The head separates pneumothorax. A nodule about 5 px wide is not separable from ribs in stride-4 features, so nodule mass is not taken from the three-class softmax.

Fused distribution:

\[
e = \mathrm{clip}((s - 0.46) / 0.25,\, 0,\, 1)
\]

\[
p = \mathrm{normalize}\big(\mathrm{clip}(1 - p_{\mathrm{ptx}} - e,\, 0),\; e,\; p_{\mathrm{ptx}}\big)
\]

\(p_{\mathrm{ptx}}\) is the third class of the head. \(s\) is the strongest residual peak on that study. When \(s \le 0.46\), \(e = 0\), so a normal peak near 0.39 contributes no nodule evidence.

Auto-pass requires \(p_{\mathrm{ptx}} < 0.25\), nodule mass \(< 0.35\), and \(H(p) < 0.50\) nat. Priority is \(5 p_{\mathrm{ptx}} + 2 p_{\mathrm{nodule}} + 1.2 H(p)\). Ties keep arrival order.

At seed 0 the 132 normals have nodule mass 0, pneumothorax mass near 0, and all 132 auto-pass. The 15 vessel crossings have mean nodule mass 0.674 and none auto-pass. The 11 pneumothorax studies have mean fused pneumothorax mass 0.530. Their residual peaks are also high, so they share mass with a nodule component of 0.470. Mean priority is 4.42 for pneumothorax, 2.13 for crossings, 1.04 for nodules, and about 0 for normals, so pneumothorax is first. Emergency auto-pass count is 0.

FIFO emergency wait is 3587.7 s. Priority wait is 225.0 s. The reduction is 93.7%. The 45 s slot is not a measured reading time. Auto-passed studies are absent from the radiologist queue. FIFO still reads all 180 studies in arrival order.

Five of 22 nodules (22.7%) had \(s \le 0.46\), so their evidence was 0 and they auto-passed. On this score they sit with the normals. Nodule review is 77.3%.

## 5. Board

`app.py` reads the recorded JSON and `results/dashboard.npz`. Sliders re-count the saved evaluation boxes and the saved worklist probabilities. They do not retrain. The gallery heatmap is the sigmoid of the nodule channel. White is ground truth, teal passes the entropy cap, and orange clears confidence only.

The entropy slider runs from 1.50 to 6.00 nat in steps of 0.01. The default is τ rounded to 2.58. The "recorded cuts" label allows 0.011 nat of slider rounding and an exact confidence of 0.5.
