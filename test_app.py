"""Run: python -m pytest test_app.py  (fakes the Demucs step so it runs without a GPU or model download)."""
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app

SR = 8000
VID = "TESTtest123"
c = TestClient(app.app)


def tone(seconds, on=True):
    t = np.arange(int(seconds * SR)) / SR
    x = (0.3 * np.sin(2 * np.pi * 220 * t) if on else np.zeros_like(t)).astype("float32")
    return np.stack([x, x], axis=1)


def fake_demucs(monkeypatch, stems):
    """Replace the demucs call with one that writes the given {stem: audio} wavs where demucs would."""
    real_run = app.run

    def fake_run(*cmd):
        if "demucs" not in cmd:
            return real_run(*cmd)
        out = Path(cmd[cmd.index("-o") + 1]) / app.MODEL
        out.mkdir(parents=True)
        for stem in app.STEMS:
            sf.write(out / f"{stem}.wav", stems.get(stem, tone(len(stems["vocals"]) / SR, on=False)), SR)

    monkeypatch.setattr(app, "run", fake_run)


def wait(jid):
    for _ in range(100):
        status = c.get(f"/api/jobs/{jid}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.1)
    raise AssertionError(status)


def upload(tmp_path):
    song = tmp_path / "song.wav"
    sf.write(song, tone(3), SR)
    return c.post("/api/jobs", files={"file": ("song.wav", song.read_bytes())}).json()["id"]


@pytest.fixture(autouse=True)
def cleanup():
    before = set(app.DATA.iterdir())
    yield
    for d in set(app.DATA.iterdir()) - before:
        shutil.rmtree(d)
    app.jobs.clear()


def test_youtube_id():
    assert app.youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert app.youtube_id("https://youtu.be/dQw4w9WgXcQ?si=abc") == "dQw4w9WgXcQ"
    assert app.youtube_id("https://youtube.com/watch?list=PL1&v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert app.youtube_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert app.youtube_id("https://evil.com/?v=dQw4w9WgXcQ") is None
    assert app.youtube_id("file:///etc/passwd") is None


def test_longest_gap():
    on, off = tone(5)[:, 0], tone(70, on=False)[:, 0]
    assert app.longest_gap(np.concatenate([on, off, on]), SR) == 70
    assert app.longest_gap(np.concatenate([off[: 30 * SR], on, off[: 20 * SR]]), SR) == 30
    assert app.longest_gap(on, SR) == 0
    assert app.longest_gap(on * 0.001, SR) == 5  # -70 dB bleed counts as silence


def test_bad_input():
    assert c.post("/api/jobs", data={"url": "https://example.com"}).status_code == 400
    assert c.post("/api/jobs", files={"file": ("x.exe", b"MZ")}).status_code == 400


def test_separates_all_stems(tmp_path, monkeypatch):
    fake_demucs(monkeypatch, {s: tone(3) for s in app.STEMS})
    jid = upload(tmp_path)
    assert wait(jid)["status"] == "done"
    for name in (*app.TRACKS, "original"):
        r = c.get(f"/files/{jid}/{name}.mp3")
        assert r.status_code == 200 and len(r.content) > 1000, name
    assert not (app.DATA / jid / app.MODEL).exists()  # wav stems cleaned up
    assert c.get("/").status_code == 200


def test_rejects_song_without_guitar_drums_bass(tmp_path, monkeypatch):
    fake_demucs(monkeypatch, {"vocals": tone(65), "guitar": np.concatenate([tone(2), tone(63, on=False)])})
    jid = upload(tmp_path)
    status = wait(jid)
    assert status["status"] == "error" and "can't be isolated" in status["error"], status
    assert (app.DATA / jid / "rejected.txt").exists()
    assert not (app.DATA / jid / "backtrack.mp3").exists()


def test_cached_link_is_not_separated_again(monkeypatch):
    monkeypatch.setattr(app, "run", lambda *cmd: pytest.fail(f"should not run {cmd}"))
    d = app.DATA / VID
    d.mkdir()
    (d / "done").touch()
    url = f"https://youtu.be/{VID}"
    jid = c.post("/api/jobs", data={"url": url}).json()["id"]
    assert c.get(f"/api/jobs/{jid}").json() | {"queue": 0} == {"status": "done", "error": None, "cached": True, "queue": 0}

    (d / "done").unlink()
    (d / "rejected.txt").write_text("This song can't be isolated: test")
    app.jobs.clear()
    status = c.get(f"/api/jobs/{c.post('/api/jobs', data={'url': url}).json()['id']}").json()
    assert status["status"] == "error" and status["cached"] and "can't be isolated" in status["error"]
