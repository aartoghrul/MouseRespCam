# MouseRespCam

Webcam respiratory-rate monitoring for anaesthetised mice during stereotaxic surgery.

There are no markers and no setup. The tool finds the breathing region by itself, keeps it
tracked while the camera or the surgeon moves, shows a breath-by-breath waveform you can count
yourself, and reports the breathing rate only when it can be trusted.

> **Status: research prototype.** Checked for internal consistency on five recordings, but not
> yet against a ground-truth reference. Do not use it as the sole basis for anaesthetic
> decisions.

## Quick start

```bash
uv venv -p python3.12 .venv && uv pip install -p .venv/bin/python -e .

# GUI: replay a recording in simulated real time (speed adjustable in the toolbar)
.venv/bin/python -m surgeryvitals videos/Recording_3.mp4

# Headless: process as fast as possible, write results/<name>/ and a summary figure
.venv/bin/python -m surgeryvitals videos/Recording_3.mp4 --headless --plot exploration/summary.png
```

The window sizes itself to the screen it opens on (fonts and panels scale with the screen).

## Reading the GUI

| Panel | What it shows |
|---|---|
| **Camera** | The video with the region used for the estimate. Green = tracking, amber = motion artifact (samples ignored), red dashed = region lost (re-acquiring). *Breathing map* overlays where rhythmic motion was found. |
| **Respiratory rate** | The current rate (15 s window) with a confidence label. When the signal cannot be trusted the number is withheld (shown dimmed for up to 15 s, then "—"). Below it: breath-to-breath rate, signal quality, last-60 s mean and range, and how often a rate was available. |
| **Breathing motion** | The last 20 s of the motion waveform. Each upward peak is one breath (orange marker), with vertical lines every second so you can count by hand. Grey segments are artifacts. |
| **Respiratory rate since start** | The full trend, breath-to-breath points, and a confidence strip (good / fair / poor / searching). |

Keys: `Space` play/pause, `A` re-run auto-detection, `R` draw the region by hand, `Esc` cancel
drawing, `⌘O` open a recording.

## How it works (summary)

1. **Find the mouse.** Whole-frame optical flow in ~32 px blocks. Each block is scored for
   rhythmic motion at 30–210 breaths/min, and the best compact group becomes the region. The
   30/min floor excludes the surgeon's own breathing.
2. **Keep it in place.** Feature matching (ORB + RANSAC) on the rigid scene *around* the mouse
   (stereotax, platform). Tracking the mouse itself would cancel its breathing motion.
3. **Breathing waveform.** Optical flow inside the region is projected onto the breathing
   direction (steered to the breathing frequency) and integrated. Large non-breathing motion is
   rejected. The polarity is chosen so each breath is an upward peak.
4. **Rate.** A harmonic-weighted spectrum tracked over time with an HMM is compared with
   breath-by-breath counting. A rate is shown only when the two agree and the breathing is
   regular.

Full details, parameters, design decisions, rejected alternatives and references:
**[docs/ALGORITHM.md](docs/ALGORITHM.md)**.

## Outputs (`results/<recording>/`)

| File | Content |
|---|---|
| `rate.csv` | Per second: reported rate, confidence, signal quality, breath-to-breath rate, raw tracker value, monitor state |
| `breaths.csv` | Time of every detected breath, for comparison with ground truth |
| `roi.csv` | Tracked region over time |
| `signal.npz` | Displayed waveform at frame rate, plus artifact flags |

## Project layout

```
src/surgeryvitals/
  config.py     all tunable parameters (documented defaults)
  localizer.py  automatic breathing-region detection (periodicity map)
  tracker.py    anchor tracking of the region (ORB + RANSAC, re-keying)
  monitor.py    per-frame pipeline: signal, artifacts, breaths, rate, confidence, states
  dsp.py        spectra, harmonic salience, causal filter, HMM rate tracker
  source.py     video input (recorded video; a live camera goes here)
  recorder.py   result accumulation and export
  gui.py        PySide6 / pyqtgraph monitor
  report.py     static summary figure
docs/ALGORITHM.md   method description and references
docs/figures/       figures for ALGORITHM.md and make_figures.py, which regenerates them
exploration/        analysis figures and scripts from the design phase
```

## Limitations

- A true rate below 25 breaths/min is shown as withheld rather than as a number. An
  apnoea/bradypnoea alarm is planned.
- It needs about 15 s of visible breathing to find the region, at start-up and after the camera moves.
- Live-camera input is not wired up yet.
- Accuracy has not been measured against a reference.

## Acknowledgements

We thank Deniz Kılınç Bülbül for preparing the surgery recordings that made it possible to develop
and test the algorithm.

This tool builds on established methods. The main ones are
[Farnebäck optical flow](https://doi.org/10.1007/3-540-45103-X_50),
[ORB features](https://doi.org/10.1109/ICCV.2011.6126544) with
[RANSAC](https://doi.org/10.1145/358669.358692), harmonic-sum pitch estimation (Noll, print only;
[Hermes](https://doi.org/10.1121/1.396427)), HMM pitch tracking
([pYIN](https://doi.org/10.1109/ICASSP.2014.6853678)), and agreement-based fusion of
respiratory-rate estimates ([Karlen et al.](https://doi.org/10.1109/TBME.2013.2246160)). It is
related to earlier camera-based monitoring of rodents
([Kunczik et al. 2019](https://doi.org/10.1364/BOE.10.004422)) and automatic respiratory
region-of-interest detection in humans
([Janssen et al. 2016](https://doi.org/10.1088/0967-3334/37/1/100)). Facemap's motion SVD
([Syeda et al. 2024](https://doi.org/10.1038/s41593-023-01490-6)) informed the exploration. It is
built with OpenCV, [NumPy](https://doi.org/10.1038/s41586-020-2649-2),
[SciPy](https://doi.org/10.1038/s41592-019-0686-2), PySide6, pyqtgraph and matplotlib. Full
citations are in [docs/ALGORITHM.md](docs/ALGORITHM.md#references).
