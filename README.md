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

# GUI: opens ready to record; press the red Record button to start the live camera
.venv/bin/python -m surgeryvitals

# GUI: replay a recording in simulated real time (speed adjustable in the toolbar)
.venv/bin/python -m surgeryvitals videos/Recording_3.mp4

# Headless: process as fast as possible, write results/<name>/ and a summary figure
.venv/bin/python -m surgeryvitals videos/Recording_3.mp4 --headless --plot exploration/summary.png
```

On Windows use `.venv\Scripts\python` instead of `.venv/bin/python`.

The window sizes itself to the screen it opens on (fonts and panels scale with the screen).

## Live camera

```bash
.venv/bin/python -m surgeryvitals --list-cameras                  # index and name of each camera
.venv/bin/python -m surgeryvitals --camera 1                      # GUI with camera 1 selected
.venv/bin/python -m surgeryvitals --camera --headless --name M12  # print the rate every 5 s; Ctrl+C stops and saves
.venv/bin/python -m surgeryvitals --camera --camera-size 1280x720 --max-fps 15   # lighter load
```

Any USB or built-in webcam should work on macOS, Windows and Linux, with no configuration:

- **Frame rate is measured, not trusted.** Drivers often report the wrong rate (the MacBook camera
  reports 15 fps and delivers 30), and webcams slow down in dim light. The rate in breaths/min
  depends directly on the frame rate, so the camera is resampled onto an even time grid at its
  measured rate. Tested against the recorded videos with jitter, a misreported rate and a
  15 fps dim-light drop: live rates match the offline ones (median difference 0.0 /min).
- **Processing rate fits the machine.** Start-up measures the cost per frame and lowers the
  processing rate (30 → 15 fps) if needed. If processing still falls behind, the backlog is
  dropped and the waveform restarts. The *Processing* row shows the load and these gaps.
- **Camera choice.** The capture backend is picked per platform (AVFoundation on macOS; DirectShow,
  then Media Foundation, on Windows; V4L2 on Linux). Full HD is requested as MJPG, and the camera
  falls back to its nearest mode.
- **macOS permission.** The first run asks for camera access for the app you started from
  (Terminal, iTerm, VS Code…). If it was denied, allow it in *System Settings → Privacy & Security
  → Camera* and restart that app.
- Nothing is recorded until you press **Record**. A live session is saved to
  `results/<session name or camera N>_<date>-<time>/` when you press *Stop* or close the window.
  Footage is written while recording.

**Settings** (sliders icon, `⌘,`): camera (choose one when several are attached), resolution,
processing rate, results folder, a session name such as an animal ID, and which outputs to save
(see [Outputs](#outputs)). *Use for this session* keeps the choice until the program closes;
*Save as default* loads it every time the program starts; *Restore built-in defaults* resets the
form. Settings cannot be changed while recording (stop first). Headless runs use `--out`, `--name`
and `--footage off|half|full` instead.

## Reading the GUI

| Panel | What it shows |
|---|---|
| **Camera** | The video with the region used for the estimate. Green = tracking, amber = motion artifact (samples ignored), red dashed = region lost (re-acquiring). *Breathing map* overlays where rhythmic motion was found. |
| **Respiratory rate** | The current rate (15 s window) with a confidence label. When the signal cannot be trusted the number is withheld (shown dimmed for up to 15 s, then "—"). Below it: breath-to-breath rate, signal quality, last-60 s mean and range, and how often a rate was available. |
| **Breathing motion** | The last 20 s of the motion waveform. Each upward peak is one breath (orange marker), with vertical lines every second so you can count by hand. Grey segments are artifacts. |
| **Respiratory rate since start** | The full trend, breath-to-breath points, and a confidence strip (good / fair / poor / searching). |

Toolbar (hover for a description): open a recording · **record / stop** the live camera |
play/pause/replay and playback speed (only shown for a recording) | auto-detect region · draw
region · breathing map | open the results folder · settings. After each session the saved folder
is also linked under the rate.

Keys: `Space` stops a live recording, or plays/pauses a recording, `A` re-run auto-detection, `R` draw the region by hand, `Esc` cancel
drawing, `⌘O` (`Ctrl+O` on Windows) open a recording, `⌘,` settings.

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

## Outputs

One folder per session: `results/<recording name>/`, or `results/<session name>_<date>-<time>/` for a
live camera. Each file can be switched off in Settings except `session.json`.

| File | Content |
|---|---|
| `rate.csv` | Per second: reported rate, confidence, signal quality, breath-to-breath rate, raw tracker value, monitor state |
| `breaths.csv` | Time of every detected breath, for comparison with ground truth |
| `motion.csv` | Per frame: displayed waveform, raw displacement along the breathing axis, raw mean optical flow of the region (full-resolution px/frame), artifact flag, state |
| `roi.csv` | Tracked region over time |
| `summary.png` | Summary figure of the session |
| `footage.mp4` | Live sessions: the camera footage (half resolution by default, ≈ 1.3 GB/hour; full ≈ 5 GB/hour). These are the evenly resampled frames that were analysed, so opening the file gives the same analysis. At half resolution, rates matched full resolution within 0.5 /min |
| `session.json` | Start time, camera, frame rates, gaps, save options and all algorithm parameters |

## Project layout

```
src/surgeryvitals/
  config.py     all tunable parameters (documented defaults)
  localizer.py  automatic breathing-region detection (periodicity map)
  tracker.py    anchor tracking of the region (ORB + RANSAC, re-keying)
  monitor.py    per-frame pipeline: signal, artifacts, breaths, rate, confidence, states
  dsp.py        spectra, harmonic salience, causal filter, HMM rate tracker
  source.py     video input: recorded video, and live cameras (resampled to an even time grid)
  recorder.py   result accumulation, saving, and footage writing
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
- Live capture has been tested on macOS (built-in camera) and with simulated cameras. It has not
  yet been run on Windows hardware.
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
built with [OpenCV](https://opencv.org/), [NumPy](https://doi.org/10.1038/s41586-020-2649-2),
[SciPy](https://doi.org/10.1038/s41592-019-0686-2), [PySide6](https://pypi.org/project/PySide6/), pyqtgraph and matplotlib. Full
citations are in [docs/ALGORITHM.md](docs/ALGORITHM.md#references).
