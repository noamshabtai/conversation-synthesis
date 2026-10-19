"""The synthesizer, with Kokoro faked: a tone per voice, so every step after
synthesis -- timing, mixing, levels, the file -- runs for real and can be
checked by ear-free arithmetic."""

import io
import pathlib
import sys
import types
import wave

import numpy as np
import pytest

from conversation_synthesis import synthesize

KOKORO_RATE = 24000


class FakeKokoro:
    """Says every text as a 0.5 s tone, whose pitch tells which voice spoke."""

    voices = ["af_heart", "bm_george", "am_michael"]

    def __init__(self, model, voices):
        self.paths = (model, voices)
        self.calls = []

    def get_voices(self):
        return self.voices

    def create(self, text, voice, speed, lang):
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        time = np.arange(int(0.5 * KOKORO_RATE / speed)) / KOKORO_RATE
        return 0.5 * np.sin(2 * np.pi * (200 + 100 * self.voices.index(voice)) * time), KOKORO_RATE


SCENE = {
    "sampling_frequency": 16000,
    "fade_out": 0.05,
    "speakers": {
        "dana": {"channel": 0, "voice": "af_heart"},
        "ravi": {"channel": 1, "voice": "bm_george"},
    },
    "utterances": [
        {"id": "hello", "speaker": "dana", "text": "Can you hear me?"},
        {"id": "reply", "speaker": "ravi", "text": "Loud and clear.", "gap": 0.25},
        {"id": "mhm", "speaker": "dana", "text": "Mhm.", "after": "reply", "offset": 0.1},
    ],
}


@pytest.fixture
def kokoro():
    return FakeKokoro("model", "voices")


def test_language_follows_the_accent_letter():
    assert synthesize.language_of("af_heart") == "en-us"
    assert synthesize.language_of("bm_george") == "en-gb"
    with pytest.raises(ValueError, match="not an English Kokoro voice"):
        synthesize.language_of("ef_dora")


def test_cache_dir_honours_the_override_then_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("CONVERSATION_SYNTHESIS_CACHE", str(tmp_path / "ci"))
    assert synthesize.cache_dir() == tmp_path / "ci"
    monkeypatch.delenv("CONVERSATION_SYNTHESIS_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert synthesize.cache_dir() == tmp_path / "xdg" / "conversation-synthesis"


def test_a_scene_by_name_is_a_shipped_one():
    assert synthesize.scene_path("conference") == synthesize.SCENES_DIR / "conference.yaml"
    assert synthesize.scene_path("gaming").is_file()


def test_a_scene_by_path_is_that_file(tmp_path):
    own = tmp_path / "mine.yaml"
    own.write_text("{}")
    assert synthesize.scene_path(str(own)) == own


def test_an_unknown_scene_names_the_shipped_ones(tmp_path):
    with pytest.raises(FileNotFoundError, match="conference, gaming"):
        synthesize.scene_path("party")
    with pytest.raises(FileNotFoundError, match="No scene file"):
        synthesize.scene_path(str(tmp_path / "missing.yaml"))


def test_the_model_is_fetched_over_https_only(monkeypatch):
    assert synthesize.model_url("x.bin").startswith("https://")
    monkeypatch.setattr(synthesize, "MODEL_URL", "file:///etc/")
    with pytest.raises(ValueError, match="non-HTTPS"):
        synthesize.model_url("passwd")


class Response(io.BytesIO):
    def __init__(self, body, length=None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_retries_a_cut_transfer(monkeypatch, tmp_path):
    monkeypatch.setenv("CONVERSATION_SYNTHESIS_CACHE", str(tmp_path))
    replies = iter([Response(b"half", length=8), Response(b"complete")])
    monkeypatch.setattr(synthesize.urllib.request, "urlopen", lambda url, timeout: next(replies))

    path = synthesize.download("model.bin")

    assert path.read_bytes() == b"complete"


def test_download_keeps_a_complete_copy_without_fetching_it_again(monkeypatch, tmp_path):
    monkeypatch.setenv("CONVERSATION_SYNTHESIS_CACHE", str(tmp_path))
    (tmp_path / "model.bin").write_bytes(b"complete")
    monkeypatch.setattr(synthesize.urllib.request, "urlopen", lambda url, timeout: Response(b"XXXXXXXX"))

    assert synthesize.download("model.bin").read_bytes() == b"complete"


def test_download_uses_the_local_copy_when_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CONVERSATION_SYNTHESIS_CACHE", str(tmp_path))
    (tmp_path / "model.bin").write_bytes(b"old")

    def offline(url, timeout):
        raise OSError("no network")

    monkeypatch.setattr(synthesize.urllib.request, "urlopen", offline)
    assert synthesize.download("model.bin").read_bytes() == b"old"


def test_load_kokoro_refuses_a_voice_it_lacks(monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
    monkeypatch.setattr(synthesize, "download", lambda name: pathlib.Path(name))

    assert isinstance(synthesize.load_kokoro(SCENE["speakers"]), FakeKokoro)
    with pytest.raises(ValueError, match="not one of Kokoro's"):
        synthesize.load_kokoro({"zed": {"channel": 0, "voice": "am_nobody"}})


# rate and pitch both stretch the delivery; Kokoro's speed is the inverse.
def test_synthesis_asks_kokoro_for_the_style_speed_and_the_voice_language(kokoro):
    synthesize.synthesize(kokoro, "bm_george", "Hi.", 16000, {"rate": 1.25, "pitch": 1.0})

    assert kokoro.calls == [{"text": "Hi.", "voice": "bm_george", "speed": 0.8, "lang": "en-gb"}]


def test_synthesis_normalizes_then_applies_volume(kokoro):
    loud = synthesize.synthesize(kokoro, "af_heart", "Hi.", 16000, {"volume": 0.5})
    raw = synthesize.synthesize(kokoro, "af_heart", "Hi.", 16000, {"normalize": False})

    assert np.abs(loud).max() == pytest.approx(0.5, abs=0.02)
    assert np.abs(raw).max() == pytest.approx(0.5, abs=0.02)


def test_synthesis_resamples_to_the_scene_rate(kokoro):
    audio = synthesize.synthesize(kokoro, "af_heart", "Hi.", 16000, {})

    assert len(audio) == 8000


def test_truncate_cuts_and_fades_to_silence():
    cut = synthesize.truncate(np.ones(1000), keep=0.5, fade_out=0.01, sampling_frequency=10000)

    assert len(cut) == 500
    assert cut[0] == 1.0
    assert cut[-1] == 0.0


def test_onsets_follow_the_chain_and_anchors_do_not_advance_it():
    rate = SCENE["sampling_frequency"]
    audio_of = {"hello": np.zeros(rate), "reply": np.zeros(rate // 2), "mhm": np.zeros(rate)}

    onset_of = synthesize.resolve_onsets(SCENE, audio_of)

    assert onset_of == {"hello": 0.0, "reply": 1.25, "mhm": pytest.approx(1.35)}


def test_mix_puts_each_line_on_its_speakers_channel_at_its_onset():
    rate = SCENE["sampling_frequency"]
    audio_of = {"hello": np.ones(4), "reply": 2 * np.ones(4), "mhm": 3 * np.ones(4)}

    mixed = synthesize.mix(SCENE, audio_of, {"hello": 0.0, "reply": 1.0, "mhm": 2.0})

    assert mixed.shape == (2 * rate + 4, 2)
    assert mixed[0, 0] == 1.0 and mixed[rate, 1] == 2.0 and mixed[2 * rate, 0] == 3.0
    assert mixed[rate, 0] == 0.0


def test_normalize_levels_every_channel_to_the_same_peak():
    leveled = synthesize.normalize(np.array([[0.1, 2.0], [-0.2, 1.0]]))

    assert np.abs(leveled).max(axis=0) == pytest.approx([synthesize.PEAK, synthesize.PEAK])


# Past full scale, integer conversion would wrap a sample to the opposite sign.
def test_write_clips_rather_than_wrapping(tmp_path):
    path = tmp_path / "out.wav"

    synthesize.write(np.array([[1.5, -0.5], [-1.5, 0.25]]), 16000, path)

    with wave.open(str(path)) as file:
        samples = np.frombuffer(file.readframes(2), np.int16).reshape(-1, 2)
        assert (file.getnchannels(), file.getframerate()) == (2, 16000)
    assert samples[0, 0] == 32767 and samples[1, 0] == -32767
    assert samples[0, 1] < 0 < samples[1, 1]


def test_main_writes_the_scene_one_talker_per_channel(monkeypatch, tmp_path):
    scene = tmp_path / "duo.yaml"
    scene.write_text(synthesize.yaml.safe_dump(SCENE))
    monkeypatch.setattr(synthesize, "load_kokoro", lambda speakers: FakeKokoro("model", "voices"))
    output = tmp_path / "duo.wav"

    synthesize.main([str(scene), "--output", str(output)])

    with wave.open(str(output)) as file:
        audio = np.frombuffer(file.readframes(file.getnframes()), np.int16).reshape(-1, file.getnchannels())
    rate = SCENE["sampling_frequency"]
    assert audio.shape[1] == 2
    assert np.abs(audio[: rate // 4, 0]).max() > 0
    assert np.abs(audio[: rate // 4, 1]).max() == 0


def test_main_names_the_file_after_the_scene(monkeypatch, tmp_path):
    scene = tmp_path / "duo.yaml"
    scene.write_text(synthesize.yaml.safe_dump(SCENE))
    monkeypatch.setattr(synthesize, "load_kokoro", lambda speakers: FakeKokoro("model", "voices"))
    monkeypatch.chdir(tmp_path)

    synthesize.main([str(scene)])

    assert (tmp_path / "duo.wav").is_file()
