# conversation-synthesis

Synthesizes a scripted conversation between several people into one WAV file
with **one talker per channel**: each voice alone on its own channel, as if
every leg of a call had been recorded separately. That is the input a spatial
audio renderer needs, which then places each channel at its own direction.

Voices are [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)'s, Apache-2.0
and trained only on permissively licensed audio, so the result may be
published, commercially too.

## Use

No install needed, with [uv](https://docs.astral.sh/uv/):

```bash
uvx --from git+https://github.com/noamshabtai/conversation-synthesis synthesize-conversation conference
```

- `conference` and `gaming` are the scenes shipped here
  (`src/conversation_synthesis/scenes/`); a path to a `.yaml` of your own works too.
- The WAV goes to `<scene>.wav` in the current folder, or to `--output <path>`.
- Kokoro's model (about 350 MB) is downloaded once, into
  `~/.cache/conversation-synthesis`, or into `$CONVERSATION_SYNTHESIS_CACHE`.

## A scene

```yaml
sampling_frequency: 16000
fade_out: 0.05
speakers:
  dana: {channel: 0, voice: af_heart}     # a American, b British; f female, m male
  ravi: {channel: 1, voice: bm_george}
utterances:
  - {id: hello, speaker: dana, text: "Can you hear me?"}
  - {id: reply, speaker: ravi, text: "Loud and clear.", gap: -0.4}
```

Utterances follow each other, `gap` seconds apart (negative overlaps them), or
are anchored to another's start with `after` and `offset`. `styles` vary the
delivery (`rate`, `pitch`, `volume`, `drive`, `normalize`), and a `background`
section adds a synthesized game soundtrack on a channel of its own; see
`scenes/gaming.yaml`.

## Develop

```bash
uv sync
uv run pre-commit install
uv run pytest
```
