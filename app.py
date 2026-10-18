"""Guitar backtrack: YouTube link or audio file -> backtrack (song minus guitar) + guitar, bass and drum stems."""
import queue
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
MODEL = "htdemucs_6s"  # the only open model with a guitar stem
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")  # what htdemucs_6s writes
CORE = ("guitar", "drums", "bass")  # the instruments a song must have
TRACKS = ("backtrack", "guitar", "bass", "drums")  # mp3s served to the player, plus original.mp3
MAX_GAP = 60  # seconds without guitar, drums or bass before a song is rejected
# shortcut: guessed level for "instrument present" (stem bleed sits below it); calibrate on ~20 real songs
SILENCE_DB = -40
MAX_SECONDS = 600
MAX_UPLOAD = 60 * 1024 * 1024
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac"}
YT_ID = re.compile(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)([\w-]{11})")

# shortcut: in-memory job table, lost on restart (finished and rejected songs are still found on disk); use Redis/DB if you run several processes
jobs: dict[str, dict] = {}
work: queue.Queue[str] = queue.Queue()  # one worker = one separation at a time, so a 4GB GPU never runs out of memory


def youtube_id(url: str) -> str | None:
    m = YT_ID.search(url.strip())
    return m.group(1) if m else None


def longest_gap(mono: np.ndarray, sr: int, thresh_db: float = SILENCE_DB) -> int:
    """Longest run of whole seconds in which the signal stays below thresh_db (RMS)."""
    n = len(mono) // sr
    rms = np.sqrt(np.mean(mono[: n * sr].reshape(n, sr) ** 2, axis=1))
    best = run = 0
    for quiet in 20 * np.log10(rms + 1e-9) < thresh_db:
        run = run + 1 if quiet else 0
        best = max(best, run)
    return best


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def mp3(src: Path, dst: Path) -> None:
    run("ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", "192k", str(dst))


def process(jid: str) -> None:
    job, d = jobs[jid], DATA / jid
    if job.get("video"):
        job["status"] = "downloading"
        run(sys.executable, "-m", "yt_dlp", "-x", "--audio-format", "wav", "--no-playlist",
            "--match-filter", f"duration <= {MAX_SECONDS}", "-o", str(d / "input.%(ext)s"),
            f"https://www.youtube.com/watch?v={job['video']}")
    src = next(d.glob("input.*"), None)
    if not src:
        raise RuntimeError(f"No audio downloaded (is the video longer than {MAX_SECONDS // 60} minutes?)")

    job["status"] = "separating"
    run(sys.executable, "-m", "demucs", "-n", MODEL, "--segment", "7",  # --segment 7 keeps VRAM low
        "-o", str(d), "--filename", "{stem}.{ext}", str(src))
    stems = d / MODEL
    backtrack = core = None
    for stem in STEMS:  # one stem in memory at a time
        audio, sr = sf.read(stems / f"{stem}.wav", dtype="float32")
        if stem != "guitar":
            backtrack = audio if backtrack is None else backtrack + audio
        if stem in CORE:
            core = audio.mean(axis=1) if core is None else core + audio.mean(axis=1)

    gap = longest_gap(core, sr)
    if gap > MAX_GAP:
        msg = (f"This song can't be isolated: no guitar, drums or bass were found for "
               f"{gap // 60}:{gap % 60:02d} (more than a minute), so the separation wouldn't be useful.")
        (d / "rejected.txt").write_text(msg)  # remembered, so the same link isn't separated again
        shutil.rmtree(stems)
        src.unlink()
        raise RuntimeError(msg)

    sf.write(stems / "backtrack.wav", backtrack, sr)
    for name in TRACKS:
        mp3(stems / f"{name}.wav", d / f"{name}.mp3")
    mp3(src, d / "original.mp3")
    shutil.rmtree(stems)
    src.unlink()
    (d / "done").touch()  # written last: a crash mid-way never looks finished
    job["status"] = "done"


def worker() -> None:
    while True:
        jid = work.get()
        try:
            process(jid)
        except subprocess.CalledProcessError as e:
            jobs[jid].update(status="error", error=(e.stderr or str(e))[-400:])
        except Exception as e:
            jobs[jid].update(status="error", error=str(e))


threading.Thread(target=worker, daemon=True).start()
app = FastAPI()


@app.post("/api/jobs")
async def create_job(url: str = Form(""), file: UploadFile | None = None):
    if file and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in AUDIO_EXT:
            raise HTTPException(400, f"Unsupported file type {ext}")
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "File too large (60 MB max)")
        jid = uuid.uuid4().hex
        (DATA / jid).mkdir()
        (DATA / jid / f"input{ext}").write_bytes(data)
        jobs[jid] = {"status": "queued"}
    else:
        vid = youtube_id(url)
        if not vid:
            raise HTTPException(400, "Paste a YouTube link or upload an audio file")
        jid, d = vid, DATA / vid  # same video -> same job, so each song is separated once
        if (d / "done").exists():
            jobs[jid] = {"status": "done", "cached": True}
            return {"id": jid}
        if (d / "rejected.txt").exists():
            jobs[jid] = {"status": "error", "error": (d / "rejected.txt").read_text(), "cached": True}
            return {"id": jid}
        if jobs.get(jid, {}).get("status") not in (None, "error"):
            return {"id": jid}  # already queued or running
        d.mkdir(exist_ok=True)
        jobs[jid] = {"status": "queued", "video": vid}
    work.put(jid)
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "Unknown job")
    job = jobs[jid]
    return {k: job.get(k) for k in ("status", "error", "cached")} | {"queue": work.qsize()}


app.mount("/files", StaticFiles(directory=DATA), name="files")


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")
