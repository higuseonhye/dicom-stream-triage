# A synthetic benchmark for DICOM windowing, spatial-entropy box filtering, and chest-worklist priority

This note rewrites one repository run in technical form. It is not a paper and it was not peer reviewed. It does not claim a hospital result.

## Abstract

Eight uncompressed 2048 px chest DICOM studies were read as Explicit VR and windowed on one CPU. Sequential pydicom averaged 27.28 ms (36.65 fps). A uint16 `frombuffer` view averaged 24.93 ms (40.10 fps). The same path on 8 threads reached 69.73 fps. Pixel difference was 0. On a separate 90-study evaluation set, residual peaks at confidence 0.5 produced 147 false positives, precision 0.169, and recall 1.000. Adding a 6×6 spatial-entropy cap of 2.578 nats left 21 false positives, precision 0.580, and recall 0.967, an 85.7% drop. On a 180-study worklist the mean wait for 11 pneumothorax studies was 3587.7 s under first-in, first-out and 225.0 s under priority (93.7%), with zero emergency auto-passes. All 132 normals auto-passed. Five of 22 nodules also auto-passed. JPEG 2000, RLE, a hospital PACS, and external detector weights were not run.

## 1. Background

### 1.1 What a DICOM object actually stores

A consumer image and a DICOM object do not share a layout. A DICOM file is a sequence of tagged elements. The header carries identifiers, age, study time, manufacturer, technique such as kVp and mA, and the display window (center and width). The pixel element is often 12-bit or 16-bit. Computed-tomography pixels are stored as Hounsfield units. A radiograph stores raw attenuation, rescaled into the integer range. Neither is an 8-bit JPEG.

Two costs follow. First, archives do not agree on a codec. Uncompressed Explicit VR, JPEG Lossless, JPEG 2000, and RLE can all appear, and the CPU time is dominated by whichever codec the archive chose. Second, a network wants a float tensor, commonly on \([0, 1]\). Windowing a 16-bit integer into float32 copies and casts even when the integer frame itself was viewed without a copy.

The high-speed path people describe has two parts: do not parse every metadata element when only the pixel offset is required, and decode on several threads, ideally with a memory map so the pixel bytes are not copied. Section 3 measures a narrower version of the first idea, plus threads. It does not memory-map the file.

### 1.2 Why a high score is not a nodule

A frontal chest radiograph is a projection. Ribs cross, and a vessel seen along its axis is a small round density. Both can match the appearance of a nodule. A network can assign a high confidence to that overlap.

A compact lesion, if the score map is honest, is peaked: probability is high at the center and falls off. An overlap or an artifact can be high on a few pixels and unstable across neighbors. Filtering a local patch by both the height of the score and the entropy of its spatial distribution is a way to drop that second pattern.

The filter in this repository is that idea applied to the residual image, not to a model heatmap. Per-pixel entropy did not separate the two shapes. Pooling onto a 6×6 grid did, because a Gaussian collapses into the center cells and a cross occupies a row and a column.

### 1.3 Reordering a worklist

The use measured here is not a replacement read. It is a change of order.

Under first-in, first-out, a screening list of mostly normal studies holds an acute pneumothorax until the studies in front of it are opened. A priority queue puts the pneumothorax first, holds ambiguous nodules for review, and lets studies that are quiet on every score leave the radiologist queue.

Computer-assisted detection (CADe) and diagnosis (CADx) carry a high clinical-trial bar because they participate in the finding or the diagnosis. Computer-assisted triage (CADt) is classified around notification and worklist order. Cleared pneumothorax triage devices have been described as binary presence outputs rather than localization maps. That distinction explains why a queue product is a different regulatory object from a detection product. It does not clear this repository. Nothing here is a notified device.

Published studies of real lists are a separate evidence base. Annarumma and colleagues trained a network to triage adult chest radiographs. Baltruschat and colleagues simulated a hospital worklist and reported mean pneumothorax report turnaround of 35.6 minutes under priority against 80.1 minutes under first-in, first-out, while also showing that a false negative can move a critical study to the end of the list. Those minutes are their simulation. The seconds in this note are a 45-second slot applied to synthetic labels.

## 2. Methods

Lungs are two ellipses. Ribs and vessels are drawn only inside the lungs. A nodule is a Gaussian with \(\sigma = 0.020 \times 256\) px. A vessel cross is a thin horizontal and vertical ridge. A pneumothorax darkens the apex of one lung and brightens a pleural line. Pixels are stored as \(200 + 3500x\) and recovered with window center 1950 and width 3500.

Both decode paths read the file, open the frame, and window it. The fast path views Pixel Data with `np.frombuffer`. The integer frame is not copied. The float window is. Eight threads are used. Files are read once as a warm-up before timing.

The residual is \(r = \mathrm{clip}(x - G_{\sigma=6}(x), 0)\). Peaks outside the lung ellipse are dropped. \(s = r / (r + 0.35)\). The baseline is \(s \ge 0.5\). The filter adds \(H \le \tau\). \(H\) is the entropy of a 36 px window summed onto 6×6 cells. Matching is greedy IoU in descending score, threshold 0.3. τ is chosen on 90 studies that are not the evaluation set.

Two ConvNeXt-style blocks form a stride-4 head, trained for 16 epochs on 160 studies. Only the pneumothorax probability is taken from that head. Nodule evidence is \(\mathrm{clip}((s - 0.46) / 0.25, 0, 1)\). The entropy of the renormalized normal, nodule, and pneumothorax masses is the study uncertainty. Auto-pass cuts are pneumothorax 0.25, nodule 0.35, and entropy 0.50 nat. Priority is \(5 p_{\mathrm{ptx}} + 2 p_{\mathrm{nodule}} + 1.2 H\). One wait slot is 45 seconds.

## 3. Results

Threaded throughput is 1.90× sequential pydicom. Sequential zero-copy is 1.09×. On a 2048 px frame, skipping a full element walk matters less than the window itself. A repeated run will move the milliseconds. Pixel agreement should stay at 0 regardless of seed.

On the evaluation set the baseline is 30 true positives, 147 false positives, and 0 false negatives. The filter is 29, 21, and 1. False positives fall 85.7%. Precision rises from 0.169 to 0.580. Recall is 0.967. The tuning rule required at least 85% of baseline recall, and the evaluation recall is above that line. Precision of 0.580 is still low. The remaining 21 false positives are a mix of crossings and other peaks.

All 132 normals have nodule evidence 0 and auto-pass. None of the 15 crossings auto-pass. None of the 11 pneumothorax studies auto-pass, and their mean priority is the highest of the four kinds. The 93.7% wait reduction is the result of removing normals from the slot queue and placing pneumothorax at the front of what remains. Five nodules had peaks at or below 0.46, received the same evidence as a normal, and auto-passed. Nodule review is 77.3%.

## 4. What this run does not show

Speed on JPEG 2000 or RLE. A false-positive reduction for an external detector. A mammography matrix. A measured radiologist reading time. A feature that pulls the five auto-passed nodules back into review. A claim that the pneumothorax head also classifies nodules. That head separated pneumothorax only.

Public corpora such as NIH ChestX-ray14 and MIMIC-CXR are the usual place to test whether a chest model survives a change of hospital. They are not loaded here. MONAI's intensity transforms document the same window-and-cast step this pipeline writes by hand. This code does not call MONAI.

## References

1. NEMA. DICOM Standard, PS3.5: Data Structures and Encoding. https://www.dicomstandard.org/
2. pydicom documentation. https://pydicom.github.io/
3. MONAI Core transforms. https://docs.monai.io/en/stable/transforms.html
4. Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR 2017. The expanded NIH release is ChestX-ray14.
5. Johnson AEW, Pollard TJ, Berkowitz SJ, et al. MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. Scientific Data. 2019;6:317.
6. Annarumma M, Withey SJ, Bakewell RJ, Pesce E, Goh V, Montana G. Automated triaging of adult chest radiographs with deep artificial neural networks. Radiology. 2019;291(1):196–202.
7. Baltruschat IM, Steinmeister L, Nickisch H, Saalbach A, Grass M, Adam G, Knopp T, Ittrich H. Smart chest X-ray worklist prioritization using artificial intelligence: a clinical workflow simulation. PMC8128725.
