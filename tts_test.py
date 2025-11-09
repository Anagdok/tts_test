#!/usr/bin/env python3
import numpy as np
import wave
from rknnlite.api import RKNNLite
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to your RKNN models (use absolute paths)
ENCODER_MODEL = os.path.join(BASE_DIR, "mms_tts_eng_encoder_200.rknn")
DECODER_MODEL = os.path.join(BASE_DIR, "mms_tts_eng_decoder_200.rknn")

def save_wav(filename, audio, sample_rate=22050):
    """Save numpy float32 audio array to a wav file."""
    if audio is None:
        print("No audio data to save.")
        return
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    print(f"Saved {filename}")

def main():
    # 1. Load encoder
    encoder = RKNNLite()
    if encoder.load_rknn(ENCODER_MODEL) != 0:
        print("Failed to load encoder model")
        return
    if encoder.init_runtime() != 0:
        print("Failed to initialize encoder runtime")
        return
    print("Encoder model loaded and initialized")

    # 2. Load decoder
    decoder = RKNNLite()
    if decoder.load_rknn(DECODER_MODEL) != 0:
        print("Failed to load decoder model")
        return
    if decoder.init_runtime() != 0:
        print("Failed to initialize decoder runtime")
        return
    print("Decoder model loaded and initialized")

    # 3. Get input text
    text = input("Enter text to synthesize: ").strip()
    if not text:
        print("Empty text input")
        return

    # NOTE: Replace this with your real text preprocessing.
    # Currently, this is a dummy placeholder to let the model run without crashing.
    text_ids = np.array([[ord(c) % 256 for c in text]], dtype=np.int32)

    # 4. Run encoder
    print("Running encoder inference...")
    try:
        encoder_outputs = encoder.inference(inputs=[text_ids])
    except Exception as e:
        print(f"Encoder inference error: {e}")
        return

    if not encoder_outputs or encoder_outputs[0] is None:
        print("Encoder returned no output. Likely wrong input shape or data format.")
        return

    # 5. Run decoder
    print("Running decoder inference...")
    try:
        decoder_outputs = decoder.inference(inputs=encoder_outputs)
    except Exception as e:
        print(f"Decoder inference error: {e}")
        return

    if decoder_outputs is None or len(decoder_outputs) == 0:
        print("Decoder returned no data.")
        return

    wav_output = decoder_outputs[0]

    # 6. Save audio
    save_wav("output.wav", wav_output, sample_rate=22050)
    print("Saved output.wav")

if __name__ == "__main__":
    main()
