# SmartTraffic AI — Motorcycle Helmet Violation Detection

Custom YOLO11 helmet/no-helmet detection with ByteTrack motorcycle tracking, evidence capture, CSV reporting, and annotated traffic-video output. Developed as a Google Colab prototype.

## Features

- Fine-tune YOLO11n on a motorcycle helmet dataset from Roboflow
- Evaluate precision, recall, mAP@50, and mAP@50–95
- Detect and track motorcycles using YOLO11n + ByteTrack
- Associate helmet/no-helmet detections with motorcycle tracks using a spatial heuristic
- Confirm candidate no-helmet violations across frames
- Export annotated video, evidence images, and CSV events

## Repository structure

```text
SmartTrafficAI_GitHub/
├── notebooks/SmartTrafficAI_Colab.ipynb
├── src/smarttraffic_v2.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start (Google Colab)

1. Open `notebooks/SmartTrafficAI_Colab.ipynb` in Google Colab and enable a GPU runtime.
2. Run the installation cell and download a Roboflow dataset using **your own API key**. Verify the dataset license, version, and class names.
3. Run training and validation to generate `best.pt`.
4. Upload a **raw** traffic MP4 (not an already annotated output).
5. Upload `src/smarttraffic_v2.py` to `/content/`, or clone this repository to `/content/SmartTrafficAI_GitHub`.
6. Run the V2 tracking cell, inspect the video and CSV, and export the results.

For inference without retraining, set `BEST` to your existing `best.pt` path and skip the dataset/training cells; then upload a video and run the V2 section.

## Example command

```bash
python src/smarttraffic_v2.py \
  --video traffic.mp4 \
  --helmet-model best.pt \
  --output outputs \
  --bike-conf 0.22 \
  --helmet-conf 0.30 \
  --confirm-frames 4 \
  --imgsz 960
```

## Validation snapshot

A previous experiment reported **mAP@50 0.809**, **mAP@50–95 0.376**, **precision 0.834**, and **recall 0.740** on 105 validation images. These are historical notebook-run results, **not reproduced by this repository export** and not proof of performance on real CCTV.

## Limitations

- Rider-to-bike matching uses geometry rather than ground-truth rider identities.
- ByteTrack may change IDs; event counts may contain duplicates or misses.
- AI-generated traffic video is useful for visualization, but not a reliable real-world benchmark.
- This prototype does **not** issue official traffic citations or verify legal violations.

## Data and licensing

Dataset, model weights, and sample video are **not included**. Check the Roboflow dataset's applicable license and obtain permission before distributing footage. Do not commit API keys, private videos, or generated model weights without appropriate rights.

## Tech stack

Python · Ultralytics YOLO11 · ByteTrack · OpenCV · PyTorch · Roboflow · Google Colab
