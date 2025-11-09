#!/usr/bin/env python3
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import onnxruntime

# Prefer rknnlite, but fall back to rknn if not available
RKNN_CLASS = None
try:
    from rknnlite.api import RKNNLite as _RKNNLite
    RKNN_CLASS = _RKNNLite
except Exception:
    try:
        from rknn.api import RKNN as _RKNN
        RKNN_CLASS = _RKNN
    except Exception:
        RKNN_CLASS = None

onnxruntime.set_default_logger_severity(3)

MAX_LENGTH = 200

vocab = {' ': 19, "'": 1, '-': 14, '0': 23, '1': 15, '2': 28, '3': 11, '4': 27, '5': 35, '6': 36, '_': 30,
         'a': 26, 'b': 24, 'c': 12, 'd': 5, 'e': 7, 'f': 20, 'g': 37, 'h': 6, 'i': 18, 'j': 16, 'k': 0,
         'l': 21, 'm': 17, 'n': 29, 'o': 22, 'p': 13, 'q': 34, 'r': 25, 's': 8, 't': 33, 'u': 4, 'v': 32,
         'w': 9, 'x': 31, 'y': 3, 'z': 2, '–': 10}


def init_model(model_path, target=None, device_id=None):
    """
    Initialize model. Supports .rknn via RKNN/RKNNLite and .onnx via onnxruntime.
    Returns a model object whose type string contains 'rknn' for RKNN and 'onnx' for onnxruntime.
    """
    if model_path.endswith(".rknn"):
        if RKNN_CLASS is None:
            raise RuntimeError("RKNN runtime is not available on this system.")
        model = RKNN_CLASS()
        print('--> Loading model:', model_path)
        ret = model.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f'Load RKNN model "{model_path}" failed with code {ret}')
        print('done')
        print('--> Init runtime environment (target=%s, device_id=%s)' % (target, device_id))
        # RKNNLite and RKNN use init_runtime(...) signature
        ret = model.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            raise RuntimeError('Init runtime environment failed (ret=%s)' % str(ret))
        print('done')
        return model

    elif model_path.endswith(".onnx"):
        print('--> Loading ONNX model:', model_path)
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print('done')
        return session

    else:
        raise ValueError("Unsupported model extension: %s" % model_path)


def release_model(model):
    if model is None:
        return
    t = str(type(model)).lower()
    if 'rknn' in t and hasattr(model, 'release'):
        try:
            model.release()
        except Exception:
            pass
    elif 'onnx' in t:
        try:
            del model
        except Exception:
            pass


def run_encoder(encoder_model, input_ids_array, attention_mask_array):
    """
    Run encoder. Returns tuple (log_duration, input_padding_mask, prior_means, prior_log_variances)
    Works with RKNN/RKNNLite or ONNX.
    """
    t = str(type(encoder_model)).lower()
    if 'rknn' in t:
        out = encoder_model.inference(inputs=[input_ids_array, attention_mask_array])
        if out is None:
            raise RuntimeError("Encoder inference returned None (RKNN). Check input shapes/dtypes.")
        # Expecting 4 outputs
        if len(out) < 4:
            raise RuntimeError("Encoder RKNN inference returned unexpected number of outputs: %d" % len(out))
        return out[0], out[1], out[2], out[3]
    elif 'inference' in dir(encoder_model) and 'onnx' in t:
        # onnxruntime session
        out = encoder_model.run(None, {"input_ids": input_ids_array, "attention_mask": attention_mask_array})
        if out is None or len(out) < 4:
            raise RuntimeError("Encoder ONNX inference returned unexpected outputs.")
        return out[0], out[1], out[2], out[3]
    else:
        raise RuntimeError("Unsupported encoder model type: %s" % t)


def run_decoder(decoder_model, attn, output_padding_mask, prior_means, prior_log_variances):
    """
    Run decoder/vocoder. Returns waveform (numpy array or list)
    """
    t = str(type(decoder_model)).lower()
    if 'rknn' in t:
        out = decoder_model.inference(inputs=[attn, output_padding_mask, prior_means, prior_log_variances])
        if out is None:
            raise RuntimeError("Decoder inference returned None (RKNN). Check shapes/dtypes.")
        if not isinstance(out, (list, tuple, np.ndarray)):
            raise RuntimeError("Decoder RKNN returned unexpected type: %s" % type(out))
        # Many RKNN inference returns list of outputs; first output is waveform
        waveform = out[0]
        return waveform
    elif 'inference' in dir(decoder_model) and 'onnx' in t:
        out = decoder_model.run(None, {"attn": attn, "output_padding_mask": output_padding_mask,
                                       "prior_means": prior_means, "prior_log_variances": prior_log_variances})
        if out is None or len(out) == 0:
            raise RuntimeError("Decoder ONNX inference returned no outputs.")
        return out[0]
    else:
        raise RuntimeError("Unsupported decoder model type: %s" % t)


def pad_or_trim(token_id, attention_mask, max_length):
    pad_len = max_length - len(token_id)
    if pad_len <= 0:
        token_id = token_id[:max_length]
        attention_mask = attention_mask[:max_length]
    if pad_len > 0:
        token_id = token_id + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len
    return token_id, attention_mask


def preprocess_input(text, vocab, max_length):
    text = list(text.lower())
    input_id = []
    for token in text:
        if token not in vocab:
            continue
        input_id.append(0)
        input_id.append(int(vocab[token]))
    input_id.append(0)
    attention_mask = [1] * len(input_id)

    input_id, attention_mask = pad_or_trim(input_id, attention_mask, max_length)

    input_ids_array = np.array(input_id, dtype=np.int32)[None, ...]
    attention_mask_array = np.array(attention_mask, dtype=np.int32)[None, ...]
    return input_ids_array, attention_mask_array


def middle_process(log_duration, input_padding_mask, max_length):
    # Work using torch as in example, then convert outputs to numpy for RKNN/ONNX input
    log_duration = torch.tensor(log_duration)
    input_padding_mask = torch.tensor(input_padding_mask)

    speaking_rate = 1
    length_scale = 1.0 / speaking_rate
    duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
    predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

    predicted_lengths_max_real = predicted_lengths.max()
    predicted_lengths_max = max_length * 2

    indices = torch.arange(predicted_lengths_max, dtype=predicted_lengths.dtype)
    output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
    output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

    attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
    batch_size, _, output_length, input_length = attn_mask.shape
    cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
    indices = torch.arange(output_length, dtype=duration.dtype)
    valid_indices = indices.unsqueeze(0) < cum_duration
    valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
    padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
    attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

    # Convert to numpy for feeding into RKNN/ONNX. Ensure correct dtype (int32 or float32).
    attn_np = attn.numpy().astype(np.float32)
    output_padding_mask_np = output_padding_mask.numpy().astype(np.int32)

    return attn_np, output_padding_mask_np, int(predicted_lengths_max_real)


def main():
    parser = argparse.ArgumentParser(description='MMS_TTS RKNN/RKNNLite/ONNX Demo', add_help=True)
    parser.add_argument('--encoder_model_path', type=str, required=True, help='model path, .rknn or .onnx')
    parser.add_argument('--decoder_model_path', type=str, required=True, help='model path, .rknn or .onnx')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform (rk3566/rk3588 etc)')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    parser.add_argument('--text', type=str, default=None, help='text to synthesize (optional)')
    args = parser.parse_args()

    if args.text is None:
        # example text from Radxa sample
        text = "Mister quilter is the apostle of the middle classes and we are glad to welcome his gospel."
    else:
        text = args.text

    print("Preprocessing input...")
    input_ids_array, attention_mask_array = preprocess_input(text, vocab, max_length=MAX_LENGTH)
    print("input_ids shape:", input_ids_array.shape, "attention_mask shape:", attention_mask_array.shape)

    # Init models
    print("Initializing encoder model...")
    encoder_model = init_model(args.encoder_model_path, args.target, args.device_id)
    print("Initializing decoder model...")
    decoder_model = init_model(args.decoder_model_path, args.target, args.device_id)

    try:
        print("Running encoder...")
        log_duration, input_padding_mask, prior_means, prior_log_variances = run_encoder(
            encoder_model, input_ids_array, attention_mask_array
        )
        # Basic validation of encoder outputs
        if log_duration is None or input_padding_mask is None or prior_means is None or prior_log_variances is None:
            raise RuntimeError("One or more encoder outputs are None. Check model and input shapes.")
        print("Encoder outputs shapes:", np.shape(log_duration), np.shape(input_padding_mask),
              np.shape(prior_means), np.shape(prior_log_variances))

        # Middle process -> produce attn and output_padding_mask for decoder
        attn, output_padding_mask, predicted_lengths_max_real = middle_process(log_duration, input_padding_mask, MAX_LENGTH)
        print("Middle process done. attn shape:", attn.shape, "output_padding_mask shape:", output_padding_mask.shape,
              "predicted_lengths_max_real:", predicted_lengths_max_real)

        # Run decoder
        print("Running decoder...")
        waveform = run_decoder(decoder_model, attn, output_padding_mask, prior_means, prior_log_variances)
        if waveform is None:
            raise RuntimeError("Decoder returned None. Check decoder inputs and model.")
        # waveform may be numpy array or list; ensure numpy
        waveform_np = np.asarray(waveform)
        print("Decoder waveform shape:", waveform_np.shape)

        # Determine length: sample chunk size (example uses 256 frame-per-step)
        frame_per_step = 256
        length_samples = predicted_lengths_max_real * frame_per_step
        # waveform might be [batch, samples, ...] or [samples]
        if waveform_np.ndim == 1:
            audio = waveform_np[:length_samples]
        else:
            # take first batch entry and flatten if necessary
            audio = waveform_np[0].reshape(-1)[:length_samples]

        # Normalize if values appear outside [-1,1] or are int16
        if audio.dtype.kind in ('i', 'u'):
            # convert int to float in [-1,1]
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / float(max_val)
        else:
            audio = audio.astype(np.float32)

        # Save audio to output.wav at 16000 Hz (as in the sample)
        audio_save_path = "output.wav"
        sf.write(file=audio_save_path, data=audio, samplerate=16000)
        print("\nThe output wav file is saved:", audio_save_path)

    finally:
        release_model(encoder_model)
        release_model(decoder_model)


if __name__ == '__main__':
    main()
