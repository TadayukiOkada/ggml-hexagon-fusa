#!/usr/bin/env python3
"""Resident memory of a running speech server over back-to-back requests.

Sends N requests one at a time, cycling over the WAV files, and reads VmRSS of
the server process after each. The growth is the difference of the maximum RSS
between the second and the first half of the requests after the warm-up,
divided by the requests between them: the allocator returns and takes back
memory, so single samples swing by tens of MiB while a leak shows as a rising
maximum. Fails when the growth exceeds --max-kib or a request fails.

A SenseVoice server grew by about 71 KiB per request with the tensor metadata
of its weight views (43.6 KiB by this measure over 600); with the fix it stays within
a few KiB of 0.

  --pcm        raw 16 kHz mono PCM16 in the body (sensevoice-server /api/transcribe)
  --multipart  the WAV file as form field "file", plus --field k=v
               (nemo-speech serve /v1/audio/transcriptions)

usage: scripts/rss-soak.py --pid PID --url URL (--pcm | --multipart) [-n 600] [--warmup 100]
                           [--max-kib 16] [--field k=v]... [--csv OUT] WAV...
"""

import argparse
import os
import sys
import time
import urllib.request
import uuid
import wave


def rss_kib(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError(f"no VmRSS for pid {pid}")


def pcm_body(path):
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: --pcm needs 16 kHz mono PCM16")
        return w.readframes(w.getnframes()), "application/octet-stream"


def multipart_body(path, fields):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    with open(path, "rb") as f:
        data = f.read()
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, required=True, help="server process")
    ap.add_argument("--url", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pcm", action="store_true")
    mode.add_argument("--multipart", action="store_true")
    ap.add_argument("-n", type=int, default=600, help="requests (default 600)")
    ap.add_argument("--warmup", type=int, default=100, help="requests left out of the growth (default 100)")
    ap.add_argument("--max-kib", type=float, default=16.0, help="allowed growth in KiB per request (default 16)")
    ap.add_argument("--field", action="append", default=[], help="extra form field k=v for --multipart")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    ap.add_argument("--csv", help="write request, seconds, rss_kib, status")
    ap.add_argument("wav", nargs="+")
    a = ap.parse_args()
    if a.n < a.warmup + 20:
        raise SystemExit("-n must exceed --warmup by at least 20")

    fields = [f.split("=", 1) for f in a.field]
    bodies = [pcm_body(p) if a.pcm else multipart_body(p, fields) for p in a.wav]
    out = open(a.csv, "w") if a.csv else None
    if out:
        out.write("i,seconds,rss_kib,status\n")

    rss, failed, t0 = [], 0, time.time()
    for i in range(a.n):
        body, ctype = bodies[i % len(bodies)]
        req = urllib.request.Request(a.url, data=body, headers={"Content-Type": ctype}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                r.read()
                status = r.status
        except Exception as e:  # HTTP errors, a dead server, timeouts
            status = getattr(e, "code", -1)
        if status != 200:
            failed += 1
        rss.append(rss_kib(a.pid))
        if out:
            out.write(f"{i},{time.time() - t0:.1f},{rss[-1]},{status}\n")
            out.flush()

    tail = rss[a.warmup:]
    half = len(tail) // 2
    first, second = max(tail[:half]), max(tail[half:])
    growth = (second - first) / half
    print(f"rss-soak: {a.n} requests, {failed} failed, rss {rss[0]} -> {rss[-1]} KiB, "
          f"max after warm-up {first} -> {second} KiB, growth {growth:.1f} KiB/request (limit {a.max_kib})")
    if failed or growth > a.max_kib:
        print("rss-soak: FAIL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
