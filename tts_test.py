#!/usr/bin/env python3
import numpy as np
import wave
from rknnlite.api import RKNNLite

# Paths to your RKNN models
ENCODER_MODEL = "mms_tts_encoder.rknn"
DECODER_MODEL = "mms_tts_decoder.rknn"

def save_wav(filename, audio, sample_rate=22050):
    """Save numpy float32 audio array to a wav file"""
    audio = np.clip(audio, -1.0, 1.0)   # clip to [-1, 1]
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

def main():
    # 1. Load encoder
    encoder = RKNNLite()
    ret = encoder.load_rknn(ENCODER_MODEL)
    if ret != 0:
        print("Failed to load encoder model")
        return
    encoder.init_runtime()

    # 2. Load decoder
    decoder = RKNNLite()
    ret = decoder.load_rknn(DECODER_MODEL)
    if ret != 0:
        print("Failed to load decoder model")
        return
    decoder.init_runtime()

    # ---- Input text ----
    text = input("Enter text to synthesize: ")

    # NOTE: Normally you'd tokenize/convert text to phonemes/IDs first.
    # Here we mock it with dummy numpy input for illustration.
    # Replace this with your tokenizer that matches training.
    text_ids = np.array([[ord(c) % 256 for c in text]], dtype=np.int32)

    # 3. Run encoder
    encoder_outputs = encoder.inference(inputs=[text_ids])

    # 4. Run decoder (vocoder)
    wav_output = decoder.inference(inputs=encoder_outputs)[0]

    # 5. Save audio
    save_wav("output.wav", wav_output, sample_rate=22050)
    print("Saved output.wav")

if __name__ == "__main__":
    main()
