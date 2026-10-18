#!/usr/bin/env python3
"""Small deterministic GGUF forward checked against independent NumPy SAN-M/CTC."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--llama', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama / 'gguf-py'))
    from gguf import GGUFWriter
    rng = np.random.default_rng(7)
    f, d, h, k, v = 560, 32, 4, 3, 16
    prefix = 'encoder.encoders0.0.'
    tensors = {'embed.weight': (rng.integers(-8, 8, (16, f)) / 16).astype(np.float32)}
    def norm(name, width):
        tensors[name + '.weight'] = np.ones(width, np.float32)
        tensors[name + '.bias'] = np.zeros(width, np.float32)
    def linear(name, width, outputs):
        tensors[name + '.weight'] = rng.normal(0, 0.1, (outputs, width)).astype(np.float32)
        tensors[name + '.bias'] = rng.normal(0, 0.01, outputs).astype(np.float32)
    norm(prefix + 'norm1', f)
    norm(prefix + 'norm2', d)
    linear(prefix + 'self_attn.linear_q_k_v', f, 3*d)
    tensors[prefix + 'self_attn.fsmn_block.weight'] = rng.normal(0, 0.05, (k, d)).astype(np.float32)
    linear(prefix + 'self_attn.linear_out', d, d)
    linear(prefix + 'feed_forward.w_1', d, 2*d)
    linear(prefix + 'feed_forward.w_2', 2*d, d)
    norm('encoder.after_norm', d)
    norm('encoder.tp_norm', d)
    linear('ctc.ctc_lo', d, v)
    def ln(x):
        return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)
    def lin(x, name):
        return x @ tensors[name + '.weight'].T + tensors[name + '.bias']
    def reference(frames):
        x = np.concatenate((tensors['embed.weight'][:4], frames)) * np.sqrt(np.float32(d))
        freq = np.exp(-np.arange(f//2) * np.log(10000.0) / (f/2.0 - 1.0))
        position = np.arange(1, len(x)+1)[:, None] * freq[None, :]
        x += np.concatenate((np.sin(position), np.cos(position)), axis=1).astype(np.float32)
        q, key, value = np.split(lin(ln(x), prefix + 'self_attn.linear_q_k_v'), 3, axis=1)
        fsmn = value.copy()
        padded = np.pad(value, ((1, 1), (0, 0)))
        for j in range(k):
            fsmn += padded[j:j+len(x)] * tensors[prefix + 'self_attn.fsmn_block.weight'][j]
        q, key, value = (a.reshape(len(x), h, d//h).transpose(1, 0, 2) for a in (q, key, value))
        scores = (q @ key.transpose(0, 2, 1)) / np.sqrt(np.float32(d//h))
        scores -= scores.max(axis=-1, keepdims=True)
        probability = np.exp(scores)
        probability /= probability.sum(axis=-1, keepdims=True)
        attn = (probability @ value).transpose(1, 0, 2).reshape(len(x), d)
        x = lin(attn, prefix + 'self_attn.linear_out') + fsmn
        x += lin(np.maximum(lin(ln(x), prefix + 'feed_forward.w_1'), 0), prefix + 'feed_forward.w_2')
        best = lin(ln(ln(x)), 'ctc.ctc_lo').argmax(axis=-1)
        return [int(token) for i, token in enumerate(best) if token != 0 and (i == 0 or token != best[i-1])]
    # 9 and 37 rows; padded, masked attention (HTP only) is checked by test-sensevoice-backend.
    cases = [rng.normal(0, 0.5, (n, f)).astype(np.float32) for n in (5, 33)]
    expected = [reference(frames) for frames in cases]
    if not all(expected):
        raise AssertionError('fixture must have nonempty CTC output')
    with tempfile.TemporaryDirectory(prefix='sensevoice-test-') as directory:
        tmp = Path(directory)
        features = []
        for frames in cases:
            features.append(tmp / f'features-{len(frames)}.bin')
            features[-1].write_bytes(struct.pack('<ii', len(frames), f) + frames.tobytes())
        for dtype in (np.float32, np.float16):
            model = tmp / ('f16-embed.gguf' if dtype == np.float16 else 'f32.gguf')
            writer = GGUFWriter(str(model), 'sensevoice')
            for name, val in {'output_size':d, 'attention_heads':h, 'num_blocks':1, 'tp_blocks':0, 'kernel_size':k, 'vocab_size':v, 'blank_id':0}.items():
                writer.add_uint32('sv.'+name, val)
            writer.add_array('sv.query_tokens', [0, 1, 2, 3])
            for name, tensor in tensors.items():
                writer.add_tensor(name, tensor.astype(dtype) if name == 'embed.weight' else tensor)
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
            for fb, want in zip(features, expected):
                command = [str(args.binary.resolve()), '-m', str(model), '-f', str(fb), '--ids', '--threads', '2']
                reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
                if reply.returncode or [int(s) for s in reply.stdout.split()] != want:
                    raise AssertionError(f'{dtype} {fb.name}: expected {want}, got {reply.stdout}: {reply.stderr}')
            for extra in (['--backend','hexagon'], ['--backend','hexagon','--backend-lib',str(tmp/'missing.so')], ['--backend','unknown'], ['--threads','0']):
                failed = subprocess.run(command + extra, capture_output=True, text=True, timeout=15)
                if failed.returncode == 0:
                    raise AssertionError('invalid/unavailable backend silently succeeded')
        invalid = tmp / 'invalid.bin'
        invalid.write_bytes(struct.pack('<ii', -1, f))
        failed = subprocess.run([str(args.binary.resolve()), '-m', str(model), '-f', str(invalid)], capture_output=True, text=True, timeout=15)
        if failed.returncode == 0:
            raise AssertionError('negative frame count accepted')
    print('SenseVoice F32/F16 embeddings match independent SAN-M/CTC reference; invalid inputs/backend checks pass')


if __name__ == '__main__':
    main()
