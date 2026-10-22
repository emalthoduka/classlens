# ClassLens · Class Quality Evaluator

ClassLens reviews a **recorded live online class** (built for group chess classes for children) and rates it
**Poor / Average / Good / Excellent**. Every score comes with an explanation, and teaching scores cite
**timestamped quotes** from the class, so a reviewer can jump straight to that moment in the recording.

---

## Using the hosted app

**App:** https://classlens-ai-build-challenge.streamlit.app/

1. Open the link. If the app has been idle it may show *"This app has gone to sleep"*. Click
   **Yes, get this app back up!** and wait about a minute.
2. **Log in** with the username : **"admin"** and password : **"admin@123"**
3. Give it a class recording, in one of two ways:
   - **Upload recording:** an mp4, mkv, mov, webm, m4a or mp3 file, up to 1 GB.
   - **Google Drive / link:** a Drive share link set to *Anyone with the link → Viewer*. Use this for files over
     1 GB. If both are given, the link is used.
4. Click **Evaluate class** and keep the tab open. The panel on the right shows each step live. Transcription
   is the slowest step, taking about 5–6 minutes for a 1-hour class.
5. Read the result: the overall rating and score at the top, then the **Teaching**, **Audio** and **Video**
   tabs with the reasons for each score. The full report can be downloaded as JSON.

### A kind request about usage

Each evaluation relies on OpenAI’s paid API for specific steps, while the rest of the processing is handled locally. The account has a small budget of
about **$7**, and one full evaluation of a class video costs about **$0.80**, so there is room for roughly
**8 evaluations** in total.

Please test thoughtfully!. Evaluating the
same video again is free and instant, because results are cached. Thank you!

---

## How a class is scored

```
final score = 0.8 × Teaching  +  0.1 × Audio  +  0.1 × Video      (each 0–100)
```

| Final score | ≥ 80 | ≥ 70 | ≥ 50 | < 50 |
|---|---|---|---|---|
| **Class** | Excellent | Good | Average | Poor |

A **severe** teacher incident (for example shouting or abusive language) caps the class at **Average**.

- **Teaching quality (80 %):** an LLM reads the timestamped transcript, with every line labelled teacher or
  student. It scores 18 criteria from 0 to 10, such as warmth, engagement, concept clarity, questioning,
  patience, revision and recap. Each score is backed by quotes, and every quote is checked against the
  transcript. Teacher frustration or harsh language is listed with its timestamp and deducts points.
- **Audio quality (10 %):** signal analysis with no AI involved. It measures background noise (SNR), volume,
  clipping, echo, sudden noises, dropouts, hum and consistency.
- **Video quality (10 %):** frame and metadata analysis. It checks the academy's **branded background** behind
  the teacher (logo detection), resolution, frame rate, sharpness, bitrate, noise, compression and exposure.

Weights, thresholds and criteria are in [`classeval/config.py`](classeval/config.py). The LLM rubric is in
[`classeval/prompts.py`](classeval/prompts.py).

---

## How it works

1. **Ingest:** the upload or Drive file is saved to disk and fingerprinted, so results can be cached.
2. **Extract audio:** ffmpeg pulls the audio out of the video.
3. **Transcribe:** OpenAI `gpt-4o-transcribe-diarize` transcribes the audio in parallel ~3-minute chunks. The
   teacher is identified as the speaker with the most talk time.
4. **Evaluate:** the teaching evaluation (LLM), audio analysis and video analysis (about 36 frames) all run in
   parallel.
5. **Score:** the pillar scores are combined into the explainable report.

**Privacy:** the video never leaves the server and is deleted once processed. Only the audio and the transcript
are sent to OpenAI. The API key is stored as a secret and is never shown in the app.

---

## Running locally

Requires Python 3.11+ and ffmpeg.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # add OPENAI_API_KEY (and optionally APP_USERNAME / APP_PASSWORD)
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then open http://localhost:8501.

---

## Project structure

```
streamlit_app.py         Streamlit UI (entry point)
.streamlit/config.toml   theme and upload limit
classeval/               pipeline: ingest, transcription, LLM evaluation, audio/video analysis, scoring, report
assets/                  reference image for the branded-background detector
requirements.txt         Python packages
packages.txt             system packages (ffmpeg)
```
