# Colony Counter

Counts bacterial and fungal colonies on Petri dish photos, from pin-point colonies a few pixels wide to
colonies covering half the plate. It finds the dish rim and counts only inside it. A web page shows every colony
as a circle; one click removes a false detection, another adds a missed colony, a missed colony of any shape
can be traced by hand, the rim can be moved or resized, and the corrected labels can be exported for the next
training round.

## How it works

Colonies on these plates range from 4 px to 2,700 px wide (a 600x range). No single view of the photo can
handle that: shrunk to fit the detector, a whole plate loses its smallest colonies, while a full-resolution
crop cannot contain the biggest. So one detector (YOLO26, single class "colony") looks twice:

| Pass | Input | Finds |
|------|-------|-------|
| Global | whole plate, resized | large colonies |
| Tiles | overlapping crops at full resolution (25 tiles on a 3434 px photo) | small colonies |

The two are then fused: small boxes come from the tiles and large boxes from the global pass. Colonies cut
by a tile edge are merged, duplicates are removed, and only colonies inside the dish rim are counted. The
thresholds of this fusion (including how far past the rim a colony centre may lie) are tuned on held-out plates
to minimise the count error, not just the detection score.

The rim is found with a circle search on a reduced photo: every candidate circle is scored by how much of
its outline lies on an edge and how much brighter the agar is just inside than just outside, then the best one
is refitted to the edge pixels. It finds the rim on all 263 validation and test plates, including dishes that
sit off-centre or partly outside the photo. If a rim is ever wrong, *Adjust rim* in the web page fixes it.

The detector is trained on the same two views (whole plates and full-resolution tiles), with the plates
that contain large colonies repeated so the model sees enough of them.

Fuzzy colonies (spreading, diffuse or faint ones whose border fades into the agar) are what detectors miss
most. `training/fuzzy.py` measures how soft each labelled colony's border is, from brightness profiles in 90
directions; about a third of the labelled colonies are fuzzy. Every training view that contains one gets an
extra blurred, lower-contrast copy, and the evaluation reports recall for fuzzy and crisp colonies separately.

## Results

The trainer evaluates the model on held-out test plates (never used for training or tuning) and writes
`reports/metrics.json` with charts:

- detection precision, recall, F1, mAP50 and mAP50-95, plus a confusion matrix of found, missed and false colonies;
- recall per colony size, from tiny (< 32 px) to huge (> 1000 px), which shows whether big colonies are missed;
- recall for fuzzy and for crisp colonies;
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
  fuzzy.py            how fuzzy each colony is (soft or faint border)
  pack_dataset.py     prepared views -> small shards for the cloud trainer, extra copies of fuzzy colonies
  cloud_trainer.py    the whole training pipeline as a Zoho Catalyst AppSail service
  train.py            YOLO26 training on one machine (GPU and CPU presets)
  export.py           trained weights -> ONNX
  tune.py             fusion thresholds tuned on the validation plates
  evaluate.py         metrics, confusion matrix and plots on the test plates
catalyst/       Zoho Catalyst project link, build script for the two AppSail services, dataset uploader
Dockerfile      inference image, for container hosts (linux/amd64)
```

Plate photos, labels and model files are not in the repository. The trainer publishes the model to the
Stratus bucket (`model/colony.onnx`, `model/colony.json`); the running trainer also serves them at
`/model/colony.onnx` and `/model/colony.json`.

## Run it

Put `colony.onnx` and `colony.json` into `weights/`, then either:

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

In the web page:

| Action | How |
|--------|-----|
| remove a false detection | click it |
| add a missed round colony | click the agar |
| add a missed colony of any shape (spreading, fuzzy, irregular) | *Draw colony*, then trace its outline (or Shift+drag) |
| fix the dish rim | *Adjust rim*: drag inside the circle to move it, drag its edge to resize it |
| zoom / move | mouse wheel / drag |

Traced colonies are exported with their outline (`polygon`) next to the usual box, and the rim as `dish_rim`.

## Train it

The labelled data is a set of zips, each holding per plate a photo (`<id>.jpg`) and its labels (`<id>.json`):

```json
{"colonies_number": 2, "labels": [{"id": 1, "x": 736, "y": 600, "width": 190, "height": 188}, ...]}
```

`x`, `y` are the top-left corner of a colony's box, in pixels. The web app's "Export corrected labels" writes
this same format, so corrected photos can be added as one more folder of training data.

### In Zoho Catalyst (no GPU, nothing heavy on your computer)

Training runs in the Catalyst cloud as an AppSail service (`training/cloud_trainer.py`, CPU only), with the
data and every checkpoint in a Stratus bucket:

1. Prepare and pack the data (a few minutes, no training):
   ```bash
   pip install -r training/requirements.txt
   python training/prepare_dataset.py data1.zip data2.zip corrected_photos/   # zips or folders
   python training/pack_dataset.py                                            # -> data/cloud/
   ```
2. In the Catalyst console (*Cloud Scale > Stratus*) create a bucket named `colony-counter-ml`, then upload
   the data with the signed-in Catalyst CLI (`catalyst login`); `--init` optionally gives starting weights:
   ```bash
   python catalyst/upload_dataset.py [--init runs/colony/<run>/weights/best.pt]
   ```
3. Deploy the trainer (a few KB of code; it installs a trimmed PyTorch on the instance):
   ```bash
   python catalyst/stage.py trainer
   cd catalyst
   catalyst deploy appsail --name colonytrainer --build-path <repo>/.build/trainer --stack python_3_11 --command "python3 training/cloud_trainer.py"
   ```
4. In the console, open the AppSail service *colonytrainer > Configuration > App Execution Settings* and set
   the memory to 2048 MB and the disk to 1024 MB. If the bucket has another name, add the environment
   variable `BUCKET`.
5. Open the service URL once; the page shows the progress. The trainer then keeps itself going: rounds of
   two shards (400 views, downloaded and deleted again, as the instance has only about 1 GB of disk), a
   checkpoint to Stratus after each, and after every pass over the data a new ONNX model with tuned counting
   settings in `model/` of the bucket. After the last pass it writes the test report to `reports/`. A
   restarted instance resumes from the last checkpoint.

CPU training is slow: count on a few hours per pass over the data.

### On a machine with a GPU

```bash
pip install -r training/requirements.txt
python training/prepare_dataset.py data1.zip data2.zip corrected_photos/
python training/train.py --preset gpu-n
python training/export.py runs/colony/gpu-n/weights/best.pt
python training/tune.py
python training/evaluate.py
```

## Deploy on Zoho Catalyst AppSail

The web app runs as a Catalyst-managed Python service; its Linux packages are bundled at build time:

```bash
python catalyst/stage.py counter --arch x86_64 --python 3.11 --bucket colony-counter-ml
cd catalyst
catalyst deploy appsail --name colonycounter --build-path <repo>/.build/counter --stack python_3_11 --command "python3 main.py"
```

The app opens its port within a second and loads the model in the background, answers each photo in about
6-10 s (AppSail stops requests at 30 s) and peaks at about 150 MB of memory. Every ten minutes it checks the
bucket and switches to a newer model published by the trainer, without a redeploy.

The `Dockerfile` builds the same app as a container image for other hosts.
