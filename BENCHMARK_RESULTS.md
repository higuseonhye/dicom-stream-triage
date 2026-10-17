# DICOM stream triage — benchmark results

One CPU run on synthetic chest studies. No hospital archive, no vendor weights, and no JPEG 2000 or RLE file is read.

- Seed 0. Pipeline frame 2048 px, 8 studies, 8 threads.
- Box counts use 90 held-out studies, separate from the 90 used to choose $\tau$. IoU threshold 0.3.
- Worklist of 180 studies. One read is a simulated slot of 45 seconds.

## 1. DICOM decode

Uncompressed explicit VR. Both paths read the file, decode the frame, and window it. np.frombuffer aliases the uint16 bytes. The float window is a new allocation. JPEG 2000 and RLE are not in this timing.

| Path | Mean ms/study | P50 ms | FPS |
| --- | --- | --- | --- |
| pydicom sequential | 27.28 | 26.49 | 36.65 |
| zero-copy sequential | 24.93 | 24.72 | 40.10 |
| zero-copy threads 8 | 14.34 | n/a | 69.73 |

Threaded FPS / pydicom FPS = 1.90. Max absolute window error 0.000000. Throughput target (threads faster than sequential pydicom): met.
The threaded milliseconds per study are the batch wall time divided by the study count. A single clicked study is the sequential zero-copy row.

## 2. False-positive boxes

The baseline keeps lung residual peaks with confidence $`\geq 0.5`$. The filter also requires spatial entropy $`H \leq \tau`$. Entropy is the Shannon entropy, in nats, of the residual mass inside a 36 px window after pooling onto a 6×6 grid. $\tau$ is chosen on the tuning split as the cut with the largest false-positive drop that does not lower precision and keeps at least 85% of baseline recall. The table freezes that cut on the held-out split.

| Policy | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| confidence $`\geq 0.5`$ | 30 | 147 | 0 | 0.169 | 1.000 |
| confidence $`\geq 0.5`$ and $`H \leq 2.578`$ | 29 | 21 | 1 | 0.580 | 0.967 |

False-positive reduction 85.7% (40% target met). Precision defense met.

## 3. Worklist

The study distribution is fused. Pneumothorax mass is the third class of the ConvNeXt study head. Nodule mass is the residual-peak score shifted by 0.46 and divided by 0.25, and scores at or below 0.46 contribute 0. The remainder is normal mass, and the three values are renormalized. Auto-pass requires pneumothorax probability, nodule probability, and the entropy of that distribution all below their cuts. Everything else is ordered by $`5 p_{\mathrm{ptx}} + 2 p_{\mathrm{nodule}} + 1.2 H`$. Emergency wait is the time until a study whose label is pneumothorax is opened. An auto-passed emergency is a miss and is left out of the wait average.

| Kind | Studies | Mean nodule | Mean pneumothorax | Mean entropy | Auto-pass |
| --- | --- | --- | --- | --- | --- |
| normal | 132 | 0.000 | 0.000 | 0.000 | 100.0% |
| nodule | 22 | 0.205 | 0.002 | 0.517 | 22.7% |
| vessel | 15 | 0.674 | 0.003 | 0.640 | 0.0% |
| pneumothorax | 11 | 0.470 | 0.530 | 0.691 | 0.0% |

FIFO emergency wait 3587.7 s, priority wait 225.0 s, reduction 93.7% (80% target met).
Normal auto-pass 100.0%. Nodule review 77.3%. Emergency misses 0 / 11.
