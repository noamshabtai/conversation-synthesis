# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["kokoro-onnx>=0.4", "numpy", "pyyaml", "scipy"]
# ///
"""Render a multi-talker script into a multi-channel demo input.

Voices are Kokoro's (Apache-2.0, trained only on permissively licensed audio),
so the result may be published, commercially too. kokoro-onnx supports no
Python past 3.13 yet, so this script carries its own requirements above and
`uv run scripts/create_input.py` runs it apart from the project's 3.14.

Each participant is synthesized separately and written to its own channel, as
if every leg of the call had been recorded locally. Channels therefore contain
no crosstalk: a channel is silent unless its own speaker is talking.

The script to render is given on the command line, and names its own output file:
`conference.yaml` for the calm three-party call, `gaming.yaml` for the
five-player squad that shouts over each other.

A script may also carry a `background` section, in which case the game the
players are talking over is synthesized onto a channel of its own — see
`game_background`.
"""

import argparse
import pathlib
import shutil
import urllib.parse
import urllib.request
import wave

import game_background
import numpy as np
import scipy.signal
import yaml

SCRIPT_DIR = pathlib.Path(__file__).parent
DEFAULT_SCRIPT_PATH = SCRIPT_DIR / "conference.yaml"
VOICES_DIR = SCRIPT_DIR / "voices"
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
# A Kokoro voice's first letter is its accent.
LANGUAGE_OF_ACCENT = {"a": "en-us", "b": "en-gb"}
DTYPE = np.int16
PEAK = 0.7
PITCH_RESOLUTION = 100
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = 30


def model_url(name):
    """Build the URL of one of Kokoro's model files.

    The scheme is checked rather than trusted: urlopen honours `file:` as
    readily as `https:`.
    """
    url = MODEL_URL + name
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError(f"Refusing to fetch a model over a non-HTTPS URL: {url}")
    return url


def download(name):
    """Fetch a model file, verifying the size so a cut transfer is retried.

    A truncated model only fails later, at load time. The timeout matters as
    much as the retry: without it a stalled transfer never raises, so the whole
    render just hangs on a half-written model.
    """
    path = VOICES_DIR / name
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(model_url(name), timeout=DOWNLOAD_TIMEOUT) as response:  # noqa: S310
                expected_size = int(response.headers["Content-Length"])
                if path.exists() and path.stat().st_size == expected_size:
                    return path
                print(f"Downloading {path.name} ({expected_size / 1e6:.0f} MB, attempt {attempt + 1})", flush=True)
                with open(path, "wb") as voice_file:
                    shutil.copyfileobj(response, voice_file)
        except OSError as error:
            if path.exists():
                print(f"{path.name}: {error}; using the local copy", flush=True)
                return path
            print(f"{path.name}: {error}, retrying", flush=True)
            continue
        if path.stat().st_size == expected_size:
            return path
    raise RuntimeError(f"Could not download {path.name} in one piece after {DOWNLOAD_ATTEMPTS} attempts")


def language_of(voice):
    """Kokoro's language for a voice, from its accent letter: af_heart is American."""
    accent = voice[0]
    if accent not in LANGUAGE_OF_ACCENT:
        raise ValueError(f"{voice} is not an English Kokoro voice (expected one starting with a or b)")
    return LANGUAGE_OF_ACCENT[accent]


def load_kokoro(speakers):
    """Download Kokoro if needed and load it, checking every speaker's voice exists.

    One model holds every voice, so a speaker only names theirs.
    """
    from kokoro_onnx import Kokoro  # noqa: PLC0415 -- only rendering needs it

    VOICES_DIR.mkdir(exist_ok=True)
    kokoro = Kokoro(str(download(MODEL_FILE)), str(download(VOICES_FILE)))
    known = set(kokoro.get_voices())
    for name, speaker in speakers.items():
        if speaker["voice"] not in known:
            raise ValueError(f"{name}'s voice {speaker['voice']} is not one of Kokoro's")
        language_of(speaker["voice"])
    return kokoro


def pitch_shift(audio, pitch):
    """Shorten the utterance by resampling, which lifts pitch and formants together.

    Formants moving with the pitch is exactly what makes the result sound like a
    younger speaker rather than a sped-up adult, and the caller has already asked
    Kokoro for a proportionally slower delivery, so the line still takes as long
    as the script's timing expects.
    """
    return scipy.signal.resample_poly(audio, PITCH_RESOLUTION, round(PITCH_RESOLUTION * pitch))


def overdrive(audio, drive):
    """Soft clip the utterance, the way a cheap headset mic breaks up when shouted into.

    The shaping is applied to the levelled signal, so a shout distorts hard while
    a muttered line barely leaves the linear part of the curve.
    """
    return np.tanh(drive * audio) / np.tanh(drive)


def synthesize(kokoro, voice, text, sampling_frequency, style):
    """Synthesize text in the given vocal style and resample it to the target sampling frequency.

    `rate` and `pitch` both stretch the delivery, as a length factor: Kokoro's
    speed is its inverse. Each utterance is brought to full scale unless the
    style switches that off, which is what lets a shout stay louder than the
    line it interrupts: normalized utterances all come back at full scale, so
    only their timing would differ. `volume` scales what is left.
    """
    pitch = style.get("pitch", 1.0)
    audio, rate = kokoro.create(
        text, voice=voice, speed=1.0 / (style.get("rate", 1.0) * pitch), lang=language_of(voice)
    )
    audio = np.asarray(audio, dtype=np.float64)
    if style.get("normalize", True):
        audio = audio / np.abs(audio).max()
    audio = style.get("volume", 1.0) * audio
    audio = pitch_shift(audio, pitch)
    audio = overdrive(audio, style["drive"]) if "drive" in style else audio
    return scipy.signal.resample_poly(audio, sampling_frequency, rate)


def truncate(audio, keep, fade_out, sampling_frequency):
    """Cut an utterance short and fade it, modelling a speaker being cut off."""
    kept_samples = round(len(audio) * keep)
    audio = audio[:kept_samples].copy()
    fade_samples = min(round(fade_out * sampling_frequency), kept_samples)
    audio[kept_samples - fade_samples :] *= np.linspace(1.0, 0.0, fade_samples)
    return audio


def render_utterances(script, kokoro):
    """Render every utterance, keyed by id."""
    sampling_frequency = script["sampling_frequency"]
    styles = script.get("styles", {})
    audio_of = {}
    for utterance in script["utterances"]:
        speaker = script["speakers"][utterance["speaker"]]
        style = styles.get(utterance.get("style", "normal"), {})
        style = style | {"pitch": style.get("pitch", 1.0) * speaker.get("pitch", 1.0)}
        audio = synthesize(kokoro, speaker["voice"], utterance["text"], sampling_frequency, style)
        if "keep" in utterance:
            audio = truncate(audio, utterance["keep"], script["fade_out"], sampling_frequency)
        audio_of[utterance["id"]] = audio
        print(f"{utterance['id']}: {len(audio) / sampling_frequency:.1f} s")
    return audio_of


def resolve_onsets(script, audio_of):
    """Turn the script's relative timing into absolute onsets, in seconds.

    Utterances anchored with `after` start relative to the *start* of the
    utterance they name and do not advance the chain, so a back-channel can
    land mid-sentence without shifting the turn that follows it.
    """
    sampling_frequency = script["sampling_frequency"]
    onset_of = {}
    chain_end = 0.0
    for utterance in script["utterances"]:
        duration = len(audio_of[utterance["id"]]) / sampling_frequency
        if "after" in utterance:
            onset_of[utterance["id"]] = onset_of[utterance["after"]] + utterance["offset"]
        else:
            onset_of[utterance["id"]] = max(0.0, chain_end + utterance.get("gap", 0.0))
            chain_end = onset_of[utterance["id"]] + duration
    return onset_of


def mix(script, audio_of, onset_of):
    """Lay every utterance onto its speaker's channel at its onset."""
    sampling_frequency = script["sampling_frequency"]
    channel_of = {name: speaker["channel"] for name, speaker in script["speakers"].items()}
    ends = [
        round(onset_of[utterance["id"]] * sampling_frequency) + len(audio_of[utterance["id"]])
        for utterance in script["utterances"]
    ]
    mixed = np.zeros((max(ends), len(channel_of)))
    for utterance in script["utterances"]:
        audio = audio_of[utterance["id"]]
        start = round(onset_of[utterance["id"]] * sampling_frequency)
        mixed[start : start + len(audio), channel_of[utterance["speaker"]]] += audio
    return mixed


def normalize(mixed):
    """Level the channels, as a conference bridge's gain control would."""
    return PEAK * mixed / np.abs(mixed).max(axis=0)


def write(mixed, sampling_frequency, output_path):
    """Write the mix as an interleaved multi-channel WAV file.

    Clipped first: past full scale, the conversion to integers would wrap a
    sample around to the opposite sign, a crack rather than a flattened peak.
    """
    interleaved = (np.clip(mixed, -1.0, 1.0) * np.iinfo(DTYPE).max).astype(DTYPE).reshape(-1)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(mixed.shape[1])
        wav_file.setsampwidth(DTYPE().itemsize)
        wav_file.setframerate(sampling_frequency)
        wav_file.writeframes(interleaved.tobytes())


def main():
    """Synthesize a multi-talker script into a multi-channel WAV file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", nargs="?", default=DEFAULT_SCRIPT_PATH, type=pathlib.Path)
    arguments = parser.parse_args()

    with open(arguments.script, "r") as script_file:
        script = yaml.safe_load(script_file)
    output_path = arguments.script.parent / script["output"]

    kokoro = load_kokoro(script["speakers"])
    audio_of = render_utterances(script, kokoro)
    onset_of = resolve_onsets(script, audio_of)
    mixed = normalize(mix(script, audio_of, onset_of))
    if "background" in script:
        mixed = game_background.add(mixed, script, onset_of)
    write(mixed, script["sampling_frequency"], output_path)

    duration = len(mixed) / script["sampling_frequency"]
    print(f"Successfully created {output_path} ({duration:.1f} s, {mixed.shape[1]} channels)")


if __name__ == "__main__":
    main()
