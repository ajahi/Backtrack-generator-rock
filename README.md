# Backtrack

Paste a YouTube link (or upload an audio file) and get:

- **Backtrack**: the song with the guitar removed, to play along with
- **Guitar**, **Bass** and **Drums**: each part on its own, to learn it
- **Original**: for comparison

Songs with more than a minute without any guitar, drums or bass are rejected (nothing useful to separate).
The same YouTube link is never separated twice: finished and rejected songs are answered from `data/` instantly.

The player switches between the tracks in sync, slows down without changing the key, and loops an A–B section.

It uses Demucs `htdemucs_6s`, the only open model with a guitar stem. Expect some bleed, especially on dense mixes.

## Run it

You need Python 3.10+ and ffmpeg (`sudo apt install ffmpeg`, or `choco install ffmpeg` on Windows).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# GPU (NVIDIA): install the CUDA build of PyTorch first. Pick your CUDA version at pytorch.org
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
# torchaudio 2.9+ dropped the save backend Demucs 4.0.1 uses, so stay on 2.8

pip install -r requirements.txt
uvicorn app:app --port 8000
```

Open http://localhost:8000. The first separation downloads the model weights (about 50 MB).

Without a GPU it still works, just slower: a few minutes per song instead of tens of seconds.

## Notes

- **4 GB GPU**: one job runs at a time and Demucs uses `--segment 7`, which fits. If you still hit out-of-memory, lower it to `--segment 5` in `app.py`.
- **Songs are cached** by YouTube video ID in `data/`. The same link twice is instant.
- **Limits**: 10-minute songs, 60 MB uploads (`MAX_SECONDS`, `MAX_UPLOAD` in `app.py`).
- **YouTube downloads** work from a home connection. They often break on cloud servers (YouTube blocks datacenter IPs). Keep yt-dlp updated: `pip install -U yt-dlp`.
- **Tests**: `pytest test_app.py` (fakes the Demucs step, so no GPU needed).
