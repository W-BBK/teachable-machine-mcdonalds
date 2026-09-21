# 🍟 McHire Detector

A webcam classifier that sits on your desk and watches you study.

Pick up your phone, and it opens the McDonald's careers page in your browser.

Motivation through consequences.

---

## What this actually is

A [Teachable Machine](https://teachablemachine.withgoogle.com/) image model exported to Keras, running live against your webcam. Every frame gets classified into one of ten desk-activity categories. If the model is more than **90% confident** that the current class is `using phone`, it opens [jobs.mchire.com](https://jobs.mchire.com) — a gentle reminder of the alternative career path.

There's a 10-second cooldown, so it only ruins your day once every ten seconds instead of sixty times a minute.

## The pipeline

```
webcam frame
  → resize to 224×224
  → normalize to [-1, 1]
  → keras_model.h5 (Teachable Machine export)
  → argmax over 10 classes
  → if class == "using phone" and confidence > 0.90 and 10s since last trigger
      → open the McDonald's careers page
```

## What it can see

The model in `labels.txt` is trained on ten classes:

| # | Class | | # | Class |
|---|-------|---|---|-------|
| 0 | working | | 5 | nothing in frame |
| 1 | calculator | | 6 | Folder |
| 2 | **using phone** ← the trigger | | 7 | reading book |
| 3 | pencil pouch | | 8 | notebook |
| 4 | bottle | | 9 | keys |

Only class 2 does anything. The rest are there so the model can tell the difference between a phone and, say, a calculator — which matters more than you'd think, since a calculator in your hand looks an awful lot like a phone to a small CNN.

## Setup

Requires **Python 3.9**.

```bash
git clone https://github.com/W-BBK/teachable-machine-mcdonalds.git
cd teachable-machine-mcdonalds

python3.9 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running it

```bash
python main.py
```

A window opens showing your webcam feed, and the terminal prints the current class and confidence score every frame. Press **Esc** with the webcam window focused to quit.

Then go do your homework.

## Configuration

Everything tunable lives at the top of `main.py`:

| What | Where | Default |
|------|-------|---------|
| Destination URL | `url` | `https://jobs.mchire.com` |
| Camera index | `cv2.VideoCapture(0)` | `0` — try `1` if you get the wrong camera |
| Confidence threshold | `confidence_score > 0.90` | 90% |
| Cooldown between triggers | `time.time()-last_fired>10` | 10 seconds |

## Platform notes

**The trigger is macOS-only as written.** It shells out to:

```python
os.system(f"open -a Safari {url}")
```

`open` is a macOS command. On other platforms, swap that line for:

```python
# Windows
os.system(f"start {url}")

# Linux
os.system(f"xdg-open {url}")

# Cross-platform (recommended)
import webbrowser
webbrowser.open(url)
```

**If you're on a Mac with an iPhone nearby**, turn off Continuity Camera first, or macOS will helpfully hand your iPhone's camera to OpenCV instead of your webcam:

> Settings → General → AirPlay & Continuity → Continuity Camera → off

## Training your own model

The included `keras_model.h5` is trained on one specific desk, in one specific room, with one specific set of lighting and one specific phone. It will be noticeably worse at recognizing yours.

To fix that:

1. Go to [Teachable Machine](https://teachablemachine.withgoogle.com/train/image) and start an **Image Project**.
2. Create a class per activity you care about. Keep `using phone` spelled exactly that way, or update the comparison in `main.py` to match.
3. Record a few hundred webcam samples per class — vary your posture, the lighting, and where you hold things. Include a `nothing in frame` class; it keeps the model from forcing every empty frame into some other category.
4. Train, then **Export Model → Tensorflow → Keras**.
5. Drop the downloaded `keras_model.h5` and `labels.txt` into the project root, replacing the existing ones.

## Repo contents

```
main.py            # the whole thing, ~60 lines
keras_model.h5     # exported Teachable Machine model
labels.txt         # class index → name, in model output order
requirements.txt   # pinned deps
```

## Known quirks

- **It runs a full model inference per frame**, with no frame skipping, so expect meaningful CPU usage and a sluggish preview window. Classifying every third frame would be plenty.
- **The webcam preview shows the downscaled 224×224 image**, not your full-resolution feed, because the resize happens before `cv2.imshow`. It looks bad on purpose-ish.
- **Confidence is printed rounded to whole percent** via some string-slicing that assumes a `.0` suffix — cosmetic only, the actual threshold check uses the raw float.
