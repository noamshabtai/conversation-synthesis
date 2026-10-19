"""The game soundtrack: shapes and levels, from the real synthesis."""

import numpy as np
import pytest

from conversation_synthesis import game_background

RATE = 16000


def test_unit_scales_to_unit_peak_and_leaves_silence_alone():
    assert np.abs(game_background.unit(np.array([0.2, -0.5]))).max() == 1.0
    assert not game_background.unit(np.zeros(3)).any()


def test_envelope_rises_then_decays():
    shape = game_background.envelope(RATE, attack=0.001, decay=0.1, sampling_frequency=RATE)

    assert shape[0] == 0.0
    assert shape.argmax() < RATE // 100
    assert shape[-1] < 0.001


@pytest.mark.parametrize("sound", sorted(game_background.SOUNDS))
def test_every_sound_is_finite_and_audible(sound):
    audio = game_background.SOUNDS[sound](RATE, np.random.default_rng(0))

    assert np.isfinite(audio).all()
    assert np.abs(audio).max() > 0


def scene(level=1.4):
    return {
        "sampling_frequency": RATE,
        "background": {
            "channel": 2,
            "level": level,
            "seed": 0,
            "ambience": {"level": 0.2},
            "events": [{"sound": "gunshot", "after": "hello", "offset": 0.5, "level": 0.9}],
        },
    }


def test_events_land_against_the_line_they_follow():
    audio = game_background.render(scene(), {"hello": 1.0}, 3 * RATE)
    quiet = game_background.render({**scene(), "background": {**scene()["background"], "events": []}}, {}, 3 * RATE)

    shot = np.abs(audio - quiet)
    assert shot[: int(1.5 * RATE) - 10].max() == pytest.approx(0, abs=1e-9)
    assert shot[int(1.5 * RATE) : int(1.6 * RATE)].max() > 0.1


def test_the_bed_is_the_same_every_time():
    first = game_background.render(scene(), {"hello": 1.0}, RATE)
    again = game_background.render(scene(), {"hello": 1.0}, RATE)

    assert np.array_equal(first, again)


# Relative to the voices' peak, on a channel of its own, the voices untouched.
def test_add_puts_the_bed_on_its_channel_at_its_level_against_the_voices():
    voices = np.zeros((RATE, 2))
    voices[:, 0] = 0.7

    mixed = game_background.add(voices, scene(level=1.4), {"hello": 0.2})

    assert mixed.shape == (RATE, 3)
    assert np.array_equal(mixed[:, :2], voices)
    assert np.abs(mixed[:, 2]).max() == pytest.approx(1.4 * 0.7)
