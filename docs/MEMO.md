# Memo: PPE compliance detection with RT-DETR

**Repo:** https://github.com/Chan-dev12/rtdetr-vision-api · **Weights:** release `v1.0`, `best.pt` (sha256 `52019c226834`)

## 1. Domain and data

**Problem.** Answer the question a site supervisor actually asks of a photo: *is anyone missing a
helmet, gloves or a vest?* Helmet, safety vest, gloves, head and hands are not COCO classes.

**Dataset.** SH17 v1 (Kaggle, Ahmad & Rahimi 2024, CC BY-NC-SA 4.0): 8,093 Pexels photos of
industrial and construction work with 17 classes. I kept the 6 needed for compliance and dropped
the other 11. Compliance needs each PPE item *and* the body part it protects, so a bare head is
visible as a head without a matching helmet. Tiny items such as ear-muffs and glasses would add
classes the questions don't use.

**Labelling.** No self-labelling. The official `sh17.yaml` class order differs from the SH17
README, so I mapped ids from the yaml and checked 12 drawn previews (`reports/label_preview/`).
Cleaning clipped 1,836 boxes that ran past the image edge. There were no exact duplicates.

## 2. Split strategy

The **test split is SH17's official `val_files.txt`** (1,620 images), unchanged, so results are
comparable with the dataset authors'. The rest is split 85/15 into train (5,502) and val (971)
**by near-duplicate group** (perceptual hash within 6 of 64 bits), stratified on each group's rarest
class, seed 42. Grouping matters because a near-copy in train and test measures memory, not
generalisation. 6 groups spanned the official split, so I dropped those 6 training images instead
of moving test images. Class balance is very skewed: helmet is 2.1% of training boxes and safety vest 1.0%.

What the test set is **not**: every split comes from the same Pexels photographers and style.
It is stock photography, not fixed site-camera footage.

## 3. Evaluation

| Test split (1,620 img) | mAP50 | mAP50-95 | P @0.65 | R @0.65 |
|---|---|---|---|---|
| all classes | 0.781 | 0.560 | 0.922 | 0.832 |
| helmet / safety-vest / gloves | 0.761 / 0.529 / 0.667 | 0.555 / 0.323 / 0.400 | 0.78 / 0.75 / 0.84 | 0.68 / **0.37** / **0.51** |

- **Per class.** Person, head and hands have recall of 0.82 to 0.89. The PPE classes that matter
  most are the weakest, in line with their share of training data.
- **Threshold 0.65** is the F1-optimal point of the sweep. Moving from 0.5 to 0.65 traded recall
  (0.865 to 0.832) for precision (0.872 to 0.922).
- **By size.** Recall is 0.94 for large boxes, 0.80 for medium, 0.29 for small. Small boxes cause
  694 of the 1,541 misses.
- **Confusion.** Almost all errors are against background: 1,345 missed boxes and 366 false
  positives. Of 40 true class swaps, 24 are gloves and hands, and 8 are helmet and head.

**What these numbers don't tell you.** (1) The threshold was picked on the test split, so
P and R at 0.65 are slightly optimistic; validation should have been used. (2) Test shares
source and style with train, so this is an in-distribution upper bound. I did not build an
out-of-source set. (3) The micro-averaged P and R are dominated by person, head and hands (91% of
boxes), which hides the 0.37 vest recall. (4) mAP says nothing about whether a *compliance answer*
is right; one missed helmet flips an image-level answer. (5) mAP50-95 penalises loose boxes that
don't affect counting.

## 4. Five failure cases

Images are in `reports/test/failures/`: ground truth green, predictions red, misses yellow. The
counts come from `errors.csv`.

| # | Image | What went wrong | Root cause and evidence | Fix |
|---|---|---|---|---|
| 1 | `01_…efc3b6dea9f6` | 54 boxes missed on a truck full of workers | **Crowding + top-down viewpoint.** Drone view, bodies overlap. 9 misses typed `missed_crowded`; 29 missed hands and 10 missed heads are small. | Aerial/overhead images; tiled inference |
| 2 | `03_…7fb5a31071bf` | All 34 people and heads on a ferry deck missed, zero predictions | **Small objects.** All 34 boxes are under 32x32 px at 1280 px, then shrunk again to 640 for the model. Small-person recall is 0.14 overall. | Train/infer at 960 to 1280 px, or tiling |
| 3 | `15_…9bbbb41ddd08` | A glove predicted as `hands` (0.78) | **Class confusion, gloves vs hands.** The glove box is small and low-contrast. This is the largest swap: 16 gloves labelled hands, 8 the reverse. Gloves recall 0.51. | More close-up glove data; a crop classifier on hand boxes |
| 4 | `19_…644ccc946e33` | Night street: a cyclist missed, a dark helmet boxed as `head` (0.72) | **Low light.** Brightness about 0.20, the darkest of the 30 worst images. The dark helmet has little contrast with the scene. For `/ask` this produces a *false "no helmet"*. The R3 dark flag (below 0.18) does not fire. | Brightness augmentation; raise the dark threshold to about 0.25 |
| 5 | `27_…6e7d81ee5a3f` | 4 `head` false positives (0.76 to 0.87) on women winnowing rice | **Label noise.** The heads are clearly visible, covered by scarves, but ground truth has no head box for them. The model is right. Reported head precision is therefore pessimistic. | Relabel; audit false positives above 0.75 |

**Pattern.** Missing objects dominates, and size is the strongest single cause. With two more
days I would train at 960 px and build a small out-of-source set of site-camera frames to measure
the real gap.

## 5. Part B: when the detector is called

The router maps each question to one intent. `meta` (what can you detect), `general` (unrelated)
and `unsupported` (colours, identities, text, unknown objects) are answered **without** RT-DETR.
Everything else runs the detector. Python then turns boxes into counts, 3x3 regions,
image-quality flags, and one-to-one body-part to PPE matches, so the LLM never counts. Seven
deterministic rules (`policy.py`) decide whether those facts support an answer. The key one, R3,
says the absence of a detection is evidence only if the image is usable and the class's measured
test recall is at least 0.60. An LLM rewording that introduces a number not in the facts is
rejected.

**"Insufficient information" example** (real output, `docs/samples/ask_hard_image.json`):

> Q: "Are there any safety vests?" on a darkened copy of `docs/samples/bus.jpg`
> Facts: 4 people and 0 vests detected; image flagged "very dark"; vest test recall 0.37.
> A: "I didn't detect any safety vests, but I can't be confident there are none: the image looks
> very dark, which is a known cause of missed detections; the detector only finds 37% of safety
> vests on its test set." (`status: insufficient_information`, rule R3)

Answering "no" would be a guess. The detector misses 63% of vests even on clean test images.
