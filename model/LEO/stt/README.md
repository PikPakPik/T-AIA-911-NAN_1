# STT (LEO)

Module de transcription audio FR basé sur `SpeechRecognition`.

## Fichiers

- `audio_transcribe.py`: transcription de fichiers WAV.
- `microphone_recognition.py`: capture micro puis transcription.
- `stt_bridge.py`: API utilisée par l'app Streamlit (`transcribe_wav_bytes`, `transcribe_microphone`).
- `requirements.txt`: dépendances locales STT.

## Installation

```bash
pip install -r model/LEO/stt/requirements.txt
```

Selon la plateforme, `PyAudio` peut nécessiter une installation système (PortAudio).

## Test WAV

Le script lit les WAV présents dans `model/LEO/stt/wav/`:

```bash
python3 model/LEO/stt/audio_transcribe.py
```

## Test microphone

```bash
python3 model/LEO/stt/microphone_recognition.py
```

## Intégration

`model/ALEXANDRE/pathfinding/chat_pathfinding_app.py` charge ce module via `stt_bridge.py` pour l'entrée audio dans le chat.
