---
title: ClassLens - Class Quality Evaluator
emoji: 🎓
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Rates recorded live classes Poor/Average/Good/Excellent
---

# ClassLens · Class Quality Evaluator

ClassLens reviews a **recorded live online class** (built for group chess classes for children) and rates it
**Poor / Average / Good / Excellent**. Every score comes with an explanation, and teaching scores cite
**timestamped quotes** from the class so a reviewer can jump straight to the moment in the recording.

- Upload a video **or** paste a Google Drive link. Recordings of 45 minutes and longer are supported.
- Live progress for every step, with timings.
- Results are cached, so re-checking the same video is instant and costs nothing.
- Optional username/password login, so the public link can't be used to spend the OpenAI credit.

---

## Contents

1. [How a class is scored](#how-a-class-is-scored)
2. [Teaching quality (80 %)](#1-teaching-quality-80-)
3. [Audio quality (10 %)](#2-audio-quality-10-)
4. [Video quality (10 %)](#3-video-quality-10-)
5. [How it works (pipeline)](#how-it-works-pipeline)
6. [Privacy and data handling](#privacy-and-data-handling)
7. [Speed and cost](#speed-and-cost)
8. [Running locally](#running-locally)
9. [Deploying to Hugging Face Spaces](#deploying-to-hugging-face-spaces)
10. [Configuration reference](#configuration-reference)
11. [Project structure](#project-structure)
12. [Testing](#testing)
13. [Troubleshooting](#troubleshooting)
14. [Known limitations](#known-limitations)

---

## How a class is scored

Three pillars are each scored 0–100 and combined:

```
final score = 0.8 × Teaching  +  0.1 × Audio  +  0.1 × Video
```

| Final score | Class |
|---|---|
| ≥ 80 | **Excellent** |
| ≥ 70 | **Good** |
| ≥ 50 | **Average** |
| < 50 | **Poor** |

**Rules:**
- **Severe incident cap:** if the teacher shouts or is abusive (a *severe* incident whose quote is verified in
  the transcript), the class is capped at **Average**, whatever the score.
- **Missing video:** if a file has no video stream (audio only), the video pillar is skipped and the remaining
  weights are rescaled.
- **Configuration:** all weights, thresholds and criteria are in [`classeval/config.py`](classeval/config.py).

### 1. Teaching quality (80 %)

One LLM call reads the whole timestamped transcript, with every line labelled `TEACHER` or `STUDENT`, and
scores 18 criteria from 0 to 10. Each criterion returns a short rationale and 1–4 pieces of evidence (timestamp +
verbatim quote).

There are three kinds of criterion:
- **Core:** always counts. If the practice is missing, it scores low.
- **Conditional:** only counts when the situation happens (e.g. a new student joined).
- **Bonus:** only counts when it happens. If it doesn't, nothing is lost.

| Criterion | Weight | Kind | What earns a high score |
|---|---|---|---|
| Positive tone & warmth | 9 | core | Happy, kind, respectful tone throughout |
| Student engagement | 9 | core | Many children answering, volunteering moves, asking questions |
| Opening greeting & rapport | 8 | core | Greets students by name, small talk, settles the class |
| Chess concept clarity | 8 | core | Topic introduced, defined and explained correctly at the children's level |
| Revision or warm-up at start | 7 | core | Either a recap of earlier lessons **or** a warm-up activity (e.g. puzzle race) before new content |
| Closing recap | 7 | core | Summarises what was learned and checks understanding before a warm goodbye |
| Patience | 7 | core | Re-explains calmly, lets children think, handles wrong answers kindly |
| Praise & encouragement | 7 | core | Specific praise, encouragement after mistakes |
| Guided problem solving | 7 | core | Uses puzzles, asks "what would you play?", gives hints rather than answers |
| Frequent questioning | 6 | core | Asks questions often instead of lecturing |
| Questions to individual students | 6 | core | Asks students by name and gives everyone a turn |
| Energy & enthusiasm | 6 | core | Lively, energetic delivery |
| Classroom management | 5 | core | Turn-taking, "raise your hand", handles mic/camera issues, keeps focus |
| Homework discussed | 5 | bonus | Previous homework discussed **at any point** in the class |
| Homework given | 5 | bonus | Practice/homework given **at any point** in the class |
| New student introduction | 4 | conditional | New student introduced and welcomed by classmates |
| Humour & laughter | 3 | bonus | Light humour, playful moments |
| Topic video shown & discussed | 3 | bonus | A related video is shown **and** discussed |

**Teaching score** = weighted average of the criteria that count (scaled to 0–100), minus incident penalties.

**Negative incidents.** The model also lists every moment where the **teacher** shows frustration, impatience,
irritation, raises their voice, is sarcastic, harsh or uses inappropriate language. Each incident is shown with
its timestamp, quote, category and penalty:

| Severity | Penalty | Examples |
|---|---|---|
| minor | −2 | Brief sigh or irritation |
| moderate | −5 | Clear scolding, repeated impatience |
| severe | −10 (and the class is capped at Average) | Shouting, humiliating or abusive language |

The total penalty is capped at −25. Only the **teacher's** behaviour is penalised: if a student swears, that is
not held against the teacher.

**Evidence check (guards against made-up quotes).** Every quote the model cites is fuzzy-matched against the
transcript:
- ✓ **verified:** the quote is found near the cited time.
- ✓\* **corrected:** the quote is found elsewhere, and the timestamp is fixed automatically.
- **?** **unmatched:** shown with a warning. An unmatched incident only counts **half** its penalty.

Transcription errors are expected ("back rank meat" = back-rank mate, "Czech" = check), and the prompt tells the
model not to penalise the teacher for them. The model is also told that classes mix English with Hindi,
Kannada, Tamil or Malayalam.

### 2. Audio quality (10 %)

This is pure signal processing in Python (numpy/scipy), with no AI and no API calls. The transcript's
timestamps tell the analysis where people are speaking, which makes the noise and SNR measurements far more
accurate than guessing.

| Metric | Weight | What it measures | Full marks when |
|---|---|---|---|
| Background noise (SNR) | 25 | How far speech sits above the background (fan, AC, hiss) | SNR ≥ 30 dB |
| Volume level | 15 | Loudness of the teacher's speech (ITU-R BS.1770 LUFS) | Between −24 and −14 LUFS |
| Clipping / distortion | 15 | Samples hitting full scale, i.e. distorted loud moments | Practically none |
| Room echo (reverb) | 15 | Estimated reverberation time (RT60) from how fast words die away | RT60 ≤ 0.35 s |
| Sudden noises | 10 | Keyboard clacks, bangs, sudden loud bursts outside speech | ≤ 0.5 per minute |
| Audio dropouts | 10 | Gaps of digital silence in the middle of the teacher's sentences (network loss) | None |
| Electrical hum | 5 | 50/60 Hz mains hum and its harmonics | Less than 6 dB above surrounding noise |
| Voice level consistency | 5 | How much the teacher's volume jumps between turns | Within ±3 dB |

Each metric card shows the measured value (e.g. "SNR ≈ 19 dB") and a plain-English verdict, plus timestamps for
dropouts and sudden noises. Speech bandwidth (muffled vs full voice) is also reported for information.

### 3. Video quality (10 %)

The video is **never fully decoded**. Technical details come from the file header (ffprobe), and about 36 single
frames are pulled with fast seeking: 16 spread across the class, plus 20 taken **while the teacher is speaking**
(chosen from the transcript).

| Metric | Weight | What it measures | Full marks when |
|---|---|---|---|
| Branded background (teacher cam) | 40 | The academy's virtual background is visible in teacher-speaking frames | Seen **3 or more times** (1 = 33, 2 = 67) |
| Resolution | 15 | Width × height, aspect ratio | 1080p or higher |
| Frame rate | 10 | Smoothness of motion | ≥ 24 fps |
| Sharpness | 10 | Image detail / focus | Crisp detail |
| Bitrate | 10 | Compression budget per pixel | ≥ 0.08 bits/pixel, or no visible artefacts |
| Noise / grain | 5 | Sensor noise, usually from poor lighting | Clean image |
| Compression artefacts | 5 | Visible 8×8 blocking in flat areas | No visible blocking |
| Exposure / brightness | 5 | Too dark, washed out, or black (camera-off) frames | Well-exposed |

**How the branded background is detected.** The teacher's camera can be full screen, a small tile in gallery
view, or a thumbnail beside a shared chessboard, and the teacher's body hides part of the background. The
detector therefore looks for the **logo anywhere in the frame**:
1. **Matching:** SIFT keypoints of the logo are matched against the frame. This works at any size, so
   thumbnails are fine.
2. **Geometric check:** RANSAC requires the matches to fit one consistent position, scale and rotation.
3. **Correlation check:** the logo is warped onto the matched spot and must correlate with the actual pixels.
   This rejects look-alikes such as a chessboard screen-share, which shares the logo's checker pattern.

The Video tab shows every teacher frame that was checked, with a **green box** where the logo was found. The
reference image is [`assets/brand_background.webp`](assets/brand_background.webp); replace it to detect a
different background.

---

## How it works (pipeline)

```
 Upload / Drive link
        │  streamed to disk (never into RAM), fingerprinted for the cache
        ▼
 ffprobe metadata ──► ffmpeg: one pass → 16 kHz WAV (analysis) + 48 kbps MP3 (API)
        │
        ▼
 Transcribe & identify teacher  (OpenAI gpt-4o-transcribe-diarize)
   • audio split into ~3-min chunks, cut at silences
   • chunk 1 first → teacher = speaker with the most TALK TIME
   • a 3–9 s clip of the teacher's voice is sent with every other chunk
     (known_speaker_references) so the teacher keeps one label across the class
   • remaining chunks run 8 at a time
        │
        ├──► Teaching evaluation (LLM) ─────────────┐  runs in parallel
        ├──► Audio analysis (local signal processing) │
        └──► Video analysis (≈36 frames)              │
                 │ video + audio files deleted here    │
                 ▼                                     ▼
                      Scoring & explainable report
```

**Why talk time and not turn count?** Children often produce many short turns ("yes", "okay"), while the
teacher produces fewer, longer ones. In a real 53-minute sample class, the teacher held 63 % of the talk time,
against 11 % for the next speaker.

**Fallbacks:**
- If voice matching fails for a chunk, that chunk falls back to "most talk time" on its own.
- The mapping confidence (high / medium / low) is recorded in the downloadable report.

---

## Privacy and data handling

| Data | Where it goes |
|---|---|
| Video | **Never leaves the server.** Deleted as soon as the audio and frames are extracted. |
| Audio | Sent to OpenAI for transcription only, then deleted locally. |
| Transcript text | Sent to OpenAI for the teaching evaluation. |
| Results | Cached on the server as JSON plus small frame thumbnails (no video or audio). |

The OpenAI API key is read from an environment variable or Space secret. It is never written to disk, shown in
the UI, or included in error messages. Check that sending class audio of children to OpenAI is allowed under
your organisation's data policy.

---

## Speed and cost

| Step | Typical time for a 1-hour class |
|---|---|
| Upload / download | Depends on your connection (a 175 MB file uploads in seconds on a fast link) |
| Audio extraction | ~10 s |
| **Transcription** | **~5–6 min** (the slowest step; it grows with video length) |
| Audio + video analysis | ~10–30 s (runs while the LLM works) |
| Teaching evaluation (LLM) | ~30–90 s |

These are estimates: transcription speed was measured on short clips and depends on OpenAI's load.

- **Paid steps:** transcription (per audio minute) and one LLM call per class (~15k input tokens for a
  53-minute class). Everything else runs locally and is free.
- **Cache:** results are stored by a fingerprint of the file (or the Google Drive file id), so re-checking the
  same video skips all paid steps.
- **Failed runs:** if a run fails part-way (e.g. the credit runs out), the completed steps are kept and are not
  billed again on retry.

---

## Running locally

**Requirements:** Python 3.11+ and ffmpeg.

```powershell
# Windows (PowerShell), from the project folder
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env          # then edit .env (see below)
.\.venv\Scripts\python.exe app.py
```

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # then edit .env
python app.py
```

Open **http://127.0.0.1:7860**. Stop with `Ctrl+C`. `.env` is read only at start-up, so restart after changing
it.

**Minimum `.env`:**
```ini
OPENAI_API_KEY=sk-...
APP_USERNAME=team          # optional – enables the login screen
APP_PASSWORD=choose-a-strong-password
```

**ffmpeg on Windows.** Install it with `winget install --id Gyan.FFmpeg -e --source winget`. If `ffmpeg` is
still not recognised in a new terminal, point the app at the folder that contains `ffmpeg.exe` and `ffprobe.exe`:
```ini
FFMPEG_DIR=C:/Users/<you>/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_.../ffmpeg-9.0.2-full_build/bin
```
`FFMPEG_DIR` is for local Windows use only. Do **not** set it on Hugging Face.

---

## Configuration reference

All settings are environment variables (`.env` locally, **Variables and secrets** on Hugging Face). Only
`OPENAI_API_KEY` is required.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | – | **Required.** OpenAI key used for transcription and evaluation |
| `APP_USERNAME` / `APP_PASSWORD` | empty | If both are set, the app asks for this login |
| `LLM_MODEL` | `gpt-5.4-mini` | Model for the teaching evaluation |
| `LLM_REASONING_EFFORT` | `medium` | Reasoning effort for reasoning models (`low` is cheaper/faster) |
| `TRANSCRIBE_MODEL` | `gpt-4o-transcribe-diarize` | Must be a diarizing model (speaker labels are required) |
| `TRANSCRIBE_LANGUAGE` | empty (auto) | ISO code to force a language, e.g. `en` |
| `TRANSCRIBE_CHUNK_SECONDS` | `180` | Chunk length; shorter chunks = more parallelism |
| `TRANSCRIBE_PARALLELISM` | `8` | Chunks transcribed at the same time |
| `TRANSCRIBE_TIMEOUT` | `300` | Seconds before a stuck transcription call is retried (2 retries) |
| `MAX_UPLOAD` | `4gb` | Largest accepted upload |
| `CACHE_DIR` | `/data/classeval_cache` if available, else `.cache/` | Where results are cached |
| `CACHE_MAX_GB` | `5` | Oldest cached results are removed beyond this size |
| `VIDEO_SAMPLE_FRAMES` | `16` | General frames sampled for picture quality |
| `TEACHER_FRAMES` | `20` | Teacher-speaking frames checked for the branded background |
| `BRAND_FULL_MARKS_HITS` | `3` | Sightings of the background needed for full marks |
| `FFMPEG_DIR` | empty | Local Windows only: folder containing `ffmpeg.exe` / `ffprobe.exe` |

Weights, class thresholds, criteria and penalties are constants in
[`classeval/config.py`](classeval/config.py). The evaluation instructions given to the LLM are in
[`classeval/prompts.py`](classeval/prompts.py). After editing the prompt, bump `PROMPT_VERSION` so old cached
evaluations are not reused.

---

## Project structure

```
app.py                     Gradio UI (entry point for Hugging Face)
classeval/
  config.py                weights, thresholds, criteria, settings
  pipeline.py              orchestration, progress tracking, caching, clean-up
  ingest.py                Google Drive / URL streaming download
  media.py                 ffprobe / ffmpeg wrappers, chunk planning, frame grabs
  transcript.py            transcript model, teacher/student mapping, LLM formatting
  transcribe.py            chunked parallel diarized transcription (OpenAI)
  prompts.py               evaluation prompt and rubric
  llm_eval.py              structured-output LLM call + evidence verification
  audio_eval.py            signal-level audio analysis
  video_eval.py            metadata + frame-quality analysis
  background.py            branded background (logo) detector
  scoring.py               pillar weighting, penalties, classification
  report_html.py           live progress panel and report rendering
  cache.py                 per-video JSON cache with size-based eviction
  errors.py                friendly error messages (e.g. OpenAI credit exhausted)
assets/brand_background.webp   reference image for the background detector
packages.txt               system packages for the Space (ffmpeg)
requirements.txt           pinned Python packages
tests/                     offline tests and synthetic test data
```

---

## Testing

The tests run **offline**: no OpenAI calls and no API key are needed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

They cover:
- teacher mapping, including on a real Zoom transcript when one is present in `samples/`
- chunk planning and evidence verification
- scoring, thresholds and the severe-incident cap
- the background detector on 13 synthetic Zoom-style scenes (full frame, gallery tile, screen-share
  thumbnail, chessboard look-alikes, …)
- friendly OpenAI error messages

To generate a synthetic class video (text-to-speech voices plus Zoom-style frames) for manual testing:
```powershell
.\.venv\Scripts\python.exe -m tests.make_sample samples/synthetic
```

---

## Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| *"the OpenAI account has run out of credit (insufficient_quota)"* | Add credit at platform.openai.com → Billing, then retry. Completed steps are cached and not billed again. |
| *"OPENAI_API_KEY is missing or invalid"* | Set or fix the `OPENAI_API_KEY` secret, then restart the Space or app. |
| *"Google Drive did not return the file"* | Set the file's sharing to **Anyone with the link → Viewer** and paste the link to a single file, not a folder. |
| *"'ffmpeg' is not installed"* | Locally: install ffmpeg or set `FFMPEG_DIR`. On a Space: make sure `packages.txt` contains `ffmpeg`. |
| Transcription seems slow | Normal for long classes (~5–6 min per hour). The progress panel shows a live timer, and the terminal / Space logs show each chunk finishing. |
| Login screen doesn't accept the password on huggingface.co | Open the direct `https://<username>-<space>.hf.space` URL instead. |
| Branded background scores low but the teacher used it | Check the frames in the Video tab: the camera may have been off, or the tile very small, during those moments. |
| Upload of a very large file fails | Use a Google Drive link instead (more reliable above ~1 GB). |

---

## Known limitations

- **Calibration:** the class thresholds and the audio/video scoring curves are reasoned starting points. They
  should be tuned against a set of classes that reviewers have already rated.
- **LLM variability:** the evaluation is evidence-based and quotes are verified, but scores can vary by about
  a point between runs. Treat borderline results (within ~2 points of a threshold) as borderline.
- **Teacher detection:** this assumes the teacher talks the most. In unusual classes (e.g. student presentations)
  the mapping confidence will show *medium* or *low*.
- **Background detector:** this was tested on synthetic Zoom-style frames. Spot-check it on real recordings
  from the Video tab.
