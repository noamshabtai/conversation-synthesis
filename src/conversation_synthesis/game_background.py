"""Synthesize the game's own audio bed for the squad chat demo.

The five voice channels carry only what the players say into their headsets.
What they are reacting to — the firefight itself — has no channel, so the demo
asks the listener to believe in a shootout they cannot hear. This module
renders it: gunfire, grenades, ricochets and footsteps, laid on the same
timeline as the script so the shots land where the callouts say they do.

It is synthesized rather than sampled, so the demo stays self contained and the
whole bed comes out of numpy at the script's own sampling frequency. Eight
kilohertz is a hard constraint on how a gunshot can sound: the crack of a rifle
lives well above the Nyquist frequency here, so what is left is the body and the
tail. That is the right compromise anyway — this is game audio arriving over the
same narrowband path as the voices, not a cinema mix.

Unlike the players, the game is not a talker at a direction. It is the mix the
engine hands the headset, identical in both ears, so it goes to a channel the
spatial renderer leaves alone.
"""

import numpy as np
import scipy.signal

TAIL_LEVEL = 0.3
BURST_JITTER = 0.12


def envelope(nsamples, attack, decay, sampling_frequency):
    """Build a percussive envelope: an attack too fast to hear, then an exponential decay."""
    time_N = np.arange(nsamples) / sampling_frequency
    return (1 - np.exp(-time_N / attack)) * np.exp(-time_N / decay)


def bandpass(audio, low, high, sampling_frequency):
    """Band limit a noise burst, clamped below Nyquist so the filter design stays valid."""
    high = min(high, 0.98 * sampling_frequency / 2)
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=sampling_frequency, output="sos")
    return scipy.signal.sosfilt(sos, audio)


def lowpass(audio, cutoff, sampling_frequency):
    """Roll off the top of a burst, which is what distance and walls do to it."""
    cutoff = min(cutoff, 0.98 * sampling_frequency / 2)
    sos = scipy.signal.butter(4, cutoff, btype="lowpass", fs=sampling_frequency, output="sos")
    return scipy.signal.sosfilt(sos, audio)


def unit(audio):
    """Scale a rendered sound to unit peak, so an event's `level` means the same thing for all of them."""
    peak = np.abs(audio).max()
    return audio / peak if peak else audio


def gunshot(sampling_frequency, rng, brightness=3600, tail=0.25, thump=0.6):
    """One shot: a band limited crack, a low thump from the muzzle, and the room behind both.

    Brightness and tail together carry the distance. A shot next to the listener
    is bright and dry; one across the map has lost its high frequencies to the
    air and gained a long tail off the geometry, which is why the two are the
    only parameters the callers vary.
    """
    nsamples = round((0.06 + tail) * sampling_frequency)
    crack = bandpass(rng.standard_normal(nsamples), 350, brightness, sampling_frequency)
    crack *= envelope(nsamples, 5e-4, 0.018, sampling_frequency)
    body = np.sin(2 * np.pi * rng.uniform(90, 130) * np.arange(nsamples) / sampling_frequency)
    body *= envelope(nsamples, 1e-3, 0.035, sampling_frequency)
    room = lowpass(rng.standard_normal(nsamples), 0.5 * brightness, sampling_frequency)
    room *= envelope(nsamples, 0.012, tail / 3, sampling_frequency)
    return unit(crack + thump * body + TAIL_LEVEL * room)


def burst(sampling_frequency, rng, shots=5, rate=9.0, **shot):
    """A burst of automatic fire, with the trigger discipline of someone under pressure.

    The spacing is jittered and every shot is synthesized separately, because a
    burst built by repeating one rendered shot reads immediately as a loop. The
    shot's own parameters are passed straight through, so a burst is as far away
    as the shots that make it up.
    """
    spacings = np.ones(shots) / rate * (1 + BURST_JITTER * rng.standard_normal(shots))
    onsets = np.round(np.cumsum(np.hstack((0.0, spacings[:-1]))) * sampling_frequency).astype(int)
    shot_audio = [gunshot(sampling_frequency, rng, **shot) for _ in range(shots)]
    audio = np.zeros(onsets[-1] + len(shot_audio[-1]))
    for onset, rendered in zip(onsets, shot_audio):
        audio[onset : onset + len(rendered)] += rng.uniform(0.8, 1.0) * rendered
    return unit(audio)


def explosion(sampling_frequency, rng, decay=0.5, debris=0.25):
    """A grenade: a low blast that lasts, with debris raining down out of it."""
    nsamples = round((3 * decay + 0.6) * sampling_frequency)
    blast = lowpass(rng.standard_normal(nsamples), 500, sampling_frequency)
    blast *= envelope(nsamples, 2e-3, decay, sampling_frequency)
    sweep = scipy.signal.chirp(np.arange(nsamples) / sampling_frequency, 75, 3 * decay, 35)
    sweep *= envelope(nsamples, 3e-3, 0.6 * decay, sampling_frequency)
    rubble = bandpass(rng.standard_normal(nsamples), 700, 3800, sampling_frequency)
    rubble *= rng.random(nsamples) ** 6 * envelope(nsamples, 0.05, 1.5 * decay, sampling_frequency)
    return unit(blast + 0.7 * sweep + debris * rubble)


def ricochet(sampling_frequency, rng, duration=0.22):
    """A round glancing off cover: a falling whine, the one game sound that is pure pitch."""
    nsamples = round(duration * sampling_frequency)
    time_N = np.arange(nsamples) / sampling_frequency
    whine = scipy.signal.chirp(time_N, rng.uniform(2600, 3400), duration, rng.uniform(600, 900))
    whine *= envelope(nsamples, 2e-3, duration / 3, sampling_frequency)
    strike = bandpass(rng.standard_normal(nsamples), 900, 3800, sampling_frequency)
    strike *= envelope(nsamples, 3e-4, 0.01, sampling_frequency)
    return unit(whine + 0.6 * strike)


def footsteps(sampling_frequency, rng, steps=6, rate=2.2):
    """Someone moving on hard ground, close enough for the engine to bother rendering."""
    nsamples = round(0.06 * sampling_frequency)
    spacings = np.ones(steps) / rate * (1 + 0.15 * rng.standard_normal(steps))
    onsets = np.round(np.cumsum(np.hstack((0.0, spacings[:-1]))) * sampling_frequency).astype(int)
    audio = np.zeros(onsets[-1] + nsamples)
    for onset in onsets:
        step = bandpass(rng.standard_normal(nsamples), 150, rng.uniform(700, 1100), sampling_frequency)
        audio[onset : onset + nsamples] += (
            rng.uniform(0.6, 1.0) * step * envelope(nsamples, 5e-4, 0.02, sampling_frequency)
        )
    return unit(audio)


def reload(sampling_frequency, rng, clicks=(0.0, 0.09, 0.18)):
    """A magazine out, a magazine in, a bolt released — three clicks and their spacing."""
    nsamples = round((clicks[-1] + 0.05) * sampling_frequency)
    click_samples = round(0.012 * sampling_frequency)
    audio = np.zeros(nsamples)
    for click in clicks:
        onset = round(click * sampling_frequency)
        metal = bandpass(rng.standard_normal(click_samples), 800, 3800, sampling_frequency)
        audio[onset : onset + click_samples] += metal * envelope(click_samples, 2e-4, 0.004, sampling_frequency)
    return unit(audio)


SOUNDS = {
    "gunshot": gunshot,
    "burst": burst,
    "explosion": explosion,
    "ricochet": ricochet,
    "footsteps": footsteps,
    "reload": reload,
}


def ambience(nsamples, sampling_frequency, rng, breath=0.15, rumble_period=9.0):
    """The map itself: wind with no events in it, and the fight going on somewhere else.

    Pink noise rather than white, because a flat spectrum sits on top of speech
    exactly where speech needs the room. The slow amplitude drift is what stops
    the bed from reading as tape hiss.
    """
    pink = lowpass(rng.standard_normal(nsamples), 300, sampling_frequency)
    time_N = np.arange(nsamples) / sampling_frequency
    wind = 1 + breath * np.sin(2 * np.pi * time_N / 7.3) * np.sin(2 * np.pi * time_N / 11.7)
    audio = unit(pink) * wind
    for onset in np.arange(rumble_period, nsamples / sampling_frequency, rumble_period):
        distant = burst(sampling_frequency, rng, shots=rng.integers(3, 7), rate=7.0, brightness=900, tail=1.1)
        start = round((onset + rng.uniform(-2.0, 2.0)) * sampling_frequency)
        end = min(start + len(distant), nsamples)
        audio[start:end] += 0.5 * distant[: end - start]
    return unit(audio)


def render(script, onset_of, nsamples):
    """Render the whole bed onto the speech timeline.

    Events are anchored to utterance ids the same way back channels are, so a
    burst is placed against the callout that reports it rather than against a
    stopwatch: retiming a line drags its gunfire along with it.
    """
    background = script["background"]
    sampling_frequency = script["sampling_frequency"]
    rng = np.random.default_rng(background.get("seed", 0))
    audio = background["ambience"]["level"] * ambience(
        nsamples, sampling_frequency, rng, **{k: v for k, v in background["ambience"].items() if k != "level"}
    )
    for event in background["events"]:
        arguments = {k: v for k, v in event.items() if k not in ("sound", "after", "offset", "at", "level")}
        sound = event["level"] * SOUNDS[event["sound"]](sampling_frequency, rng, **arguments)
        onset = event["at"] if "at" in event else onset_of[event["after"]] + event.get("offset", 0.0)
        start = round(onset * sampling_frequency)
        end = min(start + len(sound), nsamples)
        audio[start:end] += sound[: end - start]
        print(f"{event['sound']} at {onset:.1f} s")
    return audio


def add(mixed, script, onset_of):
    """Append the bed to the mix as its own channel, at the level the script asks for."""
    background = script["background"]
    audio = background["level"] * unit(render(script, onset_of, len(mixed)))
    channels = max(mixed.shape[1], background["channel"] + 1)
    padded = np.zeros((len(mixed), channels))
    padded[:, : mixed.shape[1]] = mixed
    padded[:, background["channel"]] = audio * np.abs(mixed).max()
    return padded
