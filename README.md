# FalseNine

**From broadcast football video to tactical insight.** A computer-vision, data-science and tactical-analysis project: detect players and the ball, track them, split the teams, place everyone on the pitch in metres, then measure what teams actually do using real tracking data.

<!-- TODO: add a screenshot of the dashboard: docs/dashboard.png -->
![Dashboard](docs/dashboard.png)

> Built as a portfolio project, one notebook at a time, with an honest evaluation at every step (including the places where it does not work well; see [Limitations](#limitations)).

---

## Results at a glance

| Stage | What was measured | Result |
|---|---|---|
| Detection | mAP50 / mAP50-95 on the held-out test split (19 images) | **0.926 / 0.607** |
| Detection | False person boxes, pretrained vs fine-tuned | **61 → 2** |
| Detection | Ball found / extra ball boxes | 13 of 17 found; extra boxes **8 → 2** |
| Tracking | Track IDs created for ~25 real people in 150 frames (default vs tuned) | **43 → 33** |
| Tracking | Ball detected / after gap-filling | **74% → 94%** of frames |
| Pitch mapping | Median position error, known camera calibration | **0.31 m** (90th percentile 0.85 m) |
| Pitch mapping | Calibrate once, then track the camera | under 1 m for ~270 frames (about 11 s), then 2–3 m |
| Match analysis | Share of the Metrica game with the ball in play | **58%** |

The numbers come from the notebooks' saved outputs; each notebook ends with a Findings cell that explains them.

---

## The pipeline

```
 video frame ──► 1. Detect ──► 2. Track ──► 3. Teams ──► 4. Pitch (metres) ──► 5. Metrics ──► 6. Tactics
                 YOLO11m      ByteTrack    shirt colour   homography +           distance,      shape, space
                 (fine-tuned)              k-means        camera tracking        speed zones    control, turnovers
```

1. **Detection:** YOLO11m fine-tuned on football frames (classes: Ball, Player, Ref) at 1280 px, because the ball is only ~20 px wide.
2. **Tracking:** ByteTrack (via `supervision`) gives each person an ID; the settings were tuned with a measured comparison, not by feel.
3. **Teams:** each track's shirt colour (grass pixels masked out) is clustered with k-means in Lab colour space. The model's own Ref label is *not* used, because it did not transfer to a new match.
4. **Pitch mapping:** a homography converts pixels to pitch metres; camera motion is chained between frames with SIFT features + RANSAC.
5. **Metrics:** distance, speed zones and sprints from Metrica's 25 Hz tracking data.
6. **Tactics:** team shape with and without the ball, space control (nearest-player map), and team shape after losing the ball.

---

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_exploration` | Looks at the detection dataset: class balance, box sizes, label quality |
| 02 | `02_detection_baseline` | Pretrained YOLO baseline, and what it gets wrong on football footage |
| 03 | `03_finetune` | Fine-tunes YOLO11m (on Colab), evaluates on the test split, before/after comparison |
| 04 | `04_tracking_teams` | ByteTrack tuning, shirt-colour team assignment, ball cleaning, tracked video |
| 05 | `05_pitch_mapping` | Homography from image to pitch, error against SoccerNet ground truth |
| 05b | `05b_camera_tracking` | Calibrate once, track camera motion, measure how the error grows |
| 06 | `06_tracking_metrics` | Metrica data: workload, speed zones, team shape, ball-in-play |
| 07 | `07_tactics` | Shape by possession, space control, shape after a turnover |
| 08 | `08_dashboard_assets` | Collects small outputs and builds the dashboard data files |

---

## Dashboard

A single scrolling page (plain HTML/JS, no server needed). The side menu highlights the section you are in, and the colour theme changes with it. It includes a match replay, a per-player workload explorer and interactive team-shape charts.

```bash
# after running notebook 08:
python -m http.server --directory app 8000
# then open http://localhost:8000/site/
```

---

## Repository layout

```
FalseNine/
├── notebooks/        01 … 08
├── src/              football_eval.py (IoU, evaluation), metrica_io.py (Metrica loader and event rules)
├── app/
│   ├── site/         index.html, data.js, replay.js   (the dashboard)
│   └── assets/       small figures, CSVs, key_results.json, H.264 video
├── outputs/figures/  figures used in this README
├── data/             NOT in the repo (see below)
└── models/           NOT in the repo (see below)
```

---

## Reproduce it

```bash
git clone <this repo> && cd FalseNine
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Data (not redistributed, download from the original sources):**

| Data | Source | Used for |
|---|---|---|
| Player / ball detection images | Roboflow Universe, *soccer-players-ckbru* (CC BY 4.0) | Notebooks 01–03 |
| Video clips + pitch annotations (SNGS-021 to 023) | SoccerNet Game State Reconstruction | Notebooks 04–05b |
| Tracking + event data (Games 1–3) | Metrica Sports sample data | Notebooks 06–07 |

Put them under `data/` as the notebooks expect (`data/clips`, `data/videos`, `data/metrica`).

**Model:** `models/falsenine_det.pt` is not committed. Train it with notebook 03 (a free Colab GPU is enough) and download the weights.

The notebooks run on a CPU laptop except training. Everything after training uses the saved weights.

---

## Methods in brief

- **Detection metrics:** precision, recall, AP50, mAP50 and mAP50-95 (IoU thresholds explained in the notebook).
- **Tracking:** ByteTrack with a tuned `minimum_matching_threshold`; judged by the number of IDs created, the number of tracks under one second, and mean track length.
- **Teams:** per-track median shirt colour → Lab space → k-means with K = 4 (two teams, plus "other" for referees and keepers, plus "no colour").
- **Ball:** rolling-median jump removal (150 px), then interpolation of gaps up to 10 frames, only *between* real detections.
- **Pitch error:** distance in metres between a tracked player's foot point and the SoccerNet ground-truth position, matched by IoU.
- **Speed:** position differences at 25 Hz, median then mean smoothing, values above 10.5 m/s treated as tracking glitches; top speed reported as the 99.9th percentile.
- **Sprint:** at least 1 second above 7 m/s. Speed zones: walk <2, jog 2–4, run 4–5.5, high-speed 5.5–7, sprint >7 m/s.
- **Team shape:** length, width, defensive-line height (mean depth of the four deepest outfielders), block depth and spread, computed on frames with at least 8 outfield players.
- **Space control:** every cell of a 2 m grid goes to the nearest player's team.
- **Turnover study:** team shape from 2 s before to 10 s after the ball is lost, averaged over all turnovers (77 Home, 81 Away).

---

## Findings

<!-- TODO: paste 2–3 sentences from the Findings cell of notebook 07 (shape by possession, space control, after a turnover). Keep the numbers identical to the notebook. -->

- Detection: fine-tuning mostly removes false boxes (61 → 2 on people) rather than finding more people.
- Tracking: tuning the matching threshold cut track IDs from 43 to 33 on the same 150 frames; breaks mostly happen when the camera pans.
- Pitch mapping: with a known calibration, tracked players land 0.31 m (median) from the ground truth; a single calibration decays to 2–3 m after about 11 seconds of camera motion.
- Tactics: see notebook 07 for how team shape changes with and without the ball and in the 10 seconds after a turnover.

---

## Limitations

- **Optimistic detection scores.** The 19-image test split comes from the same match as the training images. A cross-match check showed the model's *Ref* label does not transfer (it labelled blue-shirt players as referees), which is why team assignment uses shirt colour.
- **Tracking works in image space.** IDs break when the camera pans quickly. ID accuracy was *not* measured against ground-truth IDs (only ID counts and track lengths were).
- **Calibration of frame 1 uses ground-truth labels.** After that the camera is tracked automatically; lens distortion and automatic pitch-line detection are not implemented.
- **Goalkeepers** are assigned to "other", not to their team.
- **Tactical results come from one sample match** and are descriptive: no significance tests, and the difference between teams is small next to the variation within each state.
- **Space control** is a nearest-player map. It ignores player speed and the ball.
- **Possession and ball-in-play** are derived from the events file with simple rules.
- The vision pipeline (SoccerNet clips) and the tactics analysis (Metrica data) use *different matches*. They are not yet connected end to end.

## Future work

- Line-based pitch calibration (detect pitch lines to remove the dependency on ground-truth frame-1 calibration).
- Measure ID accuracy against ground-truth IDs (HOTA / IDF1).
- Assign goalkeepers to teams from position.
- Replace nearest-player space control with a speed-aware model; add confidence intervals (bootstrap) to the turnover study.
- Analyse Games 2 and 3.

---

## Credits and licences

- Detection images: Roboflow Universe, *soccer-players-ckbru*, CC BY 4.0.
- Video clips and pitch annotations: SoccerNet Game State Reconstruction (check the SoccerNet terms before redistributing any data).
- Tracking and event data: Metrica Sports sample data, see the licence in their repository.
- Libraries: Ultralytics YOLO11, supervision (ByteTrack), OpenCV, NumPy, pandas, scikit-learn, Matplotlib.

Code in this repository: MIT <!-- TODO: add a LICENSE file if you choose MIT -->.

Author: **Sereen**
