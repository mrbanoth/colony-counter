# Colony Counter

Counts bacterial and fungal colonies on Petri dish photos, from pin-point colonies a few pixels wide to
colonies covering half the plate. A web page shows every colony as a circle; one click removes a false
detection, another adds a missed colony, and the corrected labels can be exported for the next training round.

## How it works

Colonies on these plates range from 4 px to 2,700 px wide (a 600x range). No single view of the photo can
handle that: shrunk to fit the detector, a whole plate loses its smallest colonies, while a full-resolution
crop cannot contain the biggest. So one detector (YOLO26, single class "colony") looks twice:

| Pass | Input | Finds |
|------|-------|-------|
| Global | whole plate, resized | large colonies |
| Tiles | overlapping crops at full resolution (25 tiles on a 3434 px photo) | small colonies |

The two are then fused: small boxes come from the tiles and large boxes from the global pass. Colonies cut
by a tile edge are merged, duplicates are removed, and the plate rim can optionally limit the count. The
thresholds of this fusion are tuned on held-out plates to minimise the count error, not just the detection
score.

The detector is trained on the same two views (whole plates and full-resolution tiles), with the plates
that contain large colonies repeated so the model sees enough of them.

## Results

The training notebook evaluates the model on held-out test plates (never used for training or tuning) and
writes `reports/metrics.json` with charts:

- detection precision, recall, F1, mAP50 and mAP50-95, plus a confusion matrix of found, missed and false colonies;
- recall per colony size, from tiny (< 32 px) to huge (> 1000 px), which shows whether big colonies are missed;
- count accuracy per plate: mean and largest error, and the share of plates counted exactly, within ±1 and within 5%;
- the same count metrics for the global pass alone and the tile pass alone, to show what the fusion adds.

## Repository

```
app/            inference and web app (CPU, ONNX Runtime; no PyTorch needed)
  counter.py    two-pass detection and fusion
  geometry.py   tiling and box maths shared with training
  server.py     web server (standard library only)
  web/          single-page interface
training/
  prepare_dataset.py  labelled zips -> train/val/test splits, whole-plate and tile views
  train.py            YOLO26 training (Colab/GPU and CPU presets)
  colab_train.ipynb   the whole training pipeline on a free Google Colab GPU
  export.py           trained weights -> ONNX
  tune.py             fusion thresholds tuned on the validation plates
  evaluate.py         metrics, confusion matrix and plots on the test plates
catalyst/       Zoho Catalyst project link (AppSail deployment)
Dockerfile      inference image (linux/amd64)
```

Plate photos, labels and model files are not in the repository. The trained model is attached to the
[latest release](https://github.com/mrbanoth/colony-counter/releases/latest).

## Run it

Download `colony.onnx` and `colony.json` from the latest release into `weights/`, then either:

```bash
pip install -r app/requirements.txt
python app/server.py            # http://127.0.0.1:7860
```

or with Docker:

```bash
docker build --platform linux/amd64 -t colony-counter .
docker run -p 9000:9000 colony-counter   # http://127.0.0.1:9000
```

Counting from the command line: `python app/counter.py photo1.jpg photo2.jpg`

## Train it

The labelled data is a set of zips, each holding per plate a photo (`<id>.jpg`) and its labels (`<id>.json`):

```json
{"colonies_number": 2, "labels": [{"id": 1, "x": 736, "y": 600, "width": 190, "height": 188}, ...]}
```

`x`, `y` are the top-left corner of a colony's box, in pixels. The web app's "Export corrected labels" writes
this same format, so corrected photos can be added as one more folder of training data.

Training runs in Google Colab, so nothing heavy runs on your own computer:

1. Upload the zips to a Google Drive folder named `colony-data` (in *My Drive*).
2. Open [training/colab_train.ipynb in Colab](https://colab.research.google.com/github/mrbanoth/colony-counter/blob/main/training/colab_train.ipynb),
   choose *Runtime > Change runtime type > T4 GPU*, then *Runtime > Run all*.
3. About 5 hours later the model and its evaluation report are in *My Drive/colony-counter/model*.
   Checkpoints are saved to Drive after every epoch; after a disconnect, *Run all* resumes.

The same steps on any machine with a GPU:

```bash
pip install -r training/requirements.txt
python training/prepare_dataset.py data1.zip data2.zip corrected_photos/   # zips or folders
python training/train.py --preset colab
python training/export.py runs/colony/colab/weights/best.pt
python training/tune.py
python training/evaluate.py
```

## Deploy on Zoho Catalyst AppSail

```bash
docker build --platform linux/amd64 -t colony-counter:latest .
docker save colony-counter:latest -o catalyst/colony-counter.tar
cd catalyst
catalyst deploy appsail --name colonycounter --source docker-archive://colony-counter.tar --port 9000
```

The app opens its port within a second and loads the model in the background, answers each photo in well
under AppSail's 30 s limit and uses about 350 MB of memory (set the AppSail memory to 1024 MB or more).
