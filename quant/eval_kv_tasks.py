#!/usr/bin/env python3
"""Retrieval and draft acceptance checks with independently selected target/draft K/V.

Runs one local llama-server at a time. No third-party Python dependencies. Keeps
requests, responses, server commands and logs; SIGTERM shuts down only its own server.
Use a wrapper as --tool when the packaged server needs a launcher/environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from eval_kv import InProcessTool, sha256


class SharedServer:
    """Use the tool's exported stop function; no signals or process restart per pair."""
    def __init__(self, path):
        self.tool = InProcessTool(path, "_Z12llama_serveriPPc")
        self.shutdown = getattr(self.tool.lib, "_Z22llama_server_terminatev")
        self.shutdown.argtypes = []
        self.shutdown.restype = None
        self.result = None
        self.thread = None

    def launch(self, cmd, log):
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("previous server has not stopped")
        self.result = None

        def work():
            try:
                self.result = self.tool(cmd, dict(os.environ), log)
            except BaseException as exc:
                self.result = -1
                print(f"shared server error: {exc}", flush=True)

        self.thread = threading.Thread(target=work)
        self.thread.start()
        return self

    def poll(self):
        return None if self.thread.is_alive() else self.result

    def terminate(self):
        self.shutdown()

    def wait(self, timeout):
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise RuntimeError("shared server did not stop within the deadline")
        return self.result


def request(port, route, data=None):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def start(cmd, port, log, shared=None):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))  # fail before launching if someone else owns this port
    p = shared.launch(cmd, log.name) if shared else subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        rc = p.poll()
        if rc is not None:
            raise RuntimeError(f"server exited {rc}; command={cmd}")
        try:
            if request(port, "/health").get("status") == "ok":
                return p
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    p.terminate()
    p.wait(timeout=30)
    raise RuntimeError("server did not become ready")


def stop(p):
    if p.poll() is None:
        p.terminate()
        p.wait(timeout=30)


def retrieval_case(port, length, fraction, language):
    # Unique, meaningful registry records rather than a repeated word list. Keep
    # the same documents for every K/V pair and use a known answer independent of F16.
    key = "Aster-" + str(length) + "-" + str(int(fraction * 100))
    answer = "NX-" + hashlib.sha256(key.encode()).hexdigest()[:6].upper()
    if language == "ja":
        intro = "以下は点検台帳です。指定された装置のアクセスコードを台帳から検索してください。\n"
        needle = f"装置 {key} のアクセスコードは {answer} です。この値をそのまま記録します。\n"
        question = f"\n装置 {key} のアクセスコードは何ですか。コードだけを出力してください。/no_think"
        record = lambda i: f"記録{i:04d}: 区域{i % 23:02d}の温度センサーは正常です。前回点検時の測定値は{18 + i % 15}度でした。\n"
    else:
        intro = "The following maintenance registry contains the requested device access code.\n"
        needle = f"Device {key} has access code {answer}. Preserve this exact code in the registry.\n"
        question = f"\nWhat is the access code for device {key}? Return only the code. /no_think"
        record = lambda i: f"Record {i:04d}: sensor in sector {i % 23:02d} passed inspection, with a recorded temperature of {18 + i % 15} degrees.\n"

    def document(n):
        rows = [record(i) for i in range(n)]
        rows.insert(int(n * fraction), needle)
        return intro + "".join(rows) + question

    # Tokenize through this model's tokenizer; leave room for its chat template.
    lo, hi = 1, 2048
    while lo + 1 < hi:
        n = (lo + hi) // 2
        count = len(request(port, "/tokenize", {"content": document(n)})["tokens"])
        if count <= length - 128:
            lo = n
        else:
            hi = n
    prompt = document(lo)
    return {"name": f"retrieve-{length}-{int(fraction*100)}-{language}", "prompt": prompt,
            "expected": answer, "requested_context": length,
            "input_tokens": len(request(port, "/tokenize", {"content": prompt})["tokens"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", nargs="+", default=["f16:f16", "q8_0:q8_0", "q8_0:q4_0"])
    ap.add_argument("--device", default="HTP0")
    ap.add_argument("--gpu-layers", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=13056)
    ap.add_argument("--port", type=int, default=18773)
    ap.add_argument("--draft-model")
    ap.add_argument("--draft-pairs", nargs="+", default=["f16:f16", "q8_0:q8_0", "q8_0:q4_0", "q4_0:q4_0"])
    ap.add_argument("--draft-device", default="HTP0")
    ap.add_argument("--lengths", nargs="+", type=int, default=[4096, 12000])
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--in-process-library", help="Linux libllama-server-impl.so; retain the backend session across pairs")
    ap.add_argument("--log-verbosity", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    shared = SharedServer(args.in_process_library) if args.in_process_library else None
    for pair in args.pairs:
        for dpair in (["none"] + args.draft_pairs if args.draft_model else ["none"]):
            k, v = pair.split(":")
            tag = pair.replace(":", "-") + "_draft-" + dpair.replace(":", "-")
            cmd = [args.tool, "-m", args.model, "-ngl", str(args.gpu_layers), "-dev", args.device, "-fa", "on",
                   "-ctk", k, "-ctv", v, "-c", str(args.ctx), "-np", "1", "-t", "8",
                   "--host", "127.0.0.1", "--port", str(args.port), "--no-warmup", "-lv", str(args.log_verbosity)]
            if dpair != "none":
                dk, dv = dpair.split(":")
                cmd += ["-md", args.draft_model, "-ngld", "99", "-devd", args.draft_device,
                        "-ctkd", dk, "-ctvd", dv, "--spec-type", "draft-simple", "--spec-draft-n-max", "4"]
            with (out / (tag + ".server.log")).open("w") as log:
                p = start(cmd, args.port, log, shared)
                try:
                    if args.draft_model:
                        tasks = [
                            {"name": "translation", "prompt": "Translate into Japanese: The red car stopped in front of the school. /no_think"},
                            {"name": "code", "prompt": "Write a Python function that returns the sum of even integers in a list. Output only code. /no_think"},
                            {"name": "copy", "prompt": "Convert to YAML, preserve all values: {\"sensor\":\"front\",\"count\":12,\"active\":true}. /no_think"},
                            {"name": "sequence", "prompt": "Continue with only the next 20 numbers: 1, 2, 3, 4, 5, /no_think"},
                        ]
                    else:
                        tasks = [retrieval_case(args.port, length, fraction, lang)
                                 for length in args.lengths for fraction in (0.2, 0.8) for lang in ("en", "ja")]
                    for task in tasks:
                        payload = {"messages": [{"role": "user", "content": task["prompt"]}],
                                   "temperature": 0, "seed": 17, "max_tokens": args.max_tokens,
                                   "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
                        t0 = time.monotonic()
                        response = request(args.port, "/v1/chat/completions", payload)
                        msg = response["choices"][0]["message"]
                        content = msg.get("content") or ""
                        result = {"pair": pair, "draft_pair": dpair, "task": task, "cmd": cmd,
                                  "seconds": time.monotonic() - t0, "request": payload, "response": response,
                                  "content": content, "correct": task.get("expected", "") in content
                                  if "expected" in task else None}
                        if args.in_process_library:
                            result["server_library_sha256"] = sha256(args.in_process_library)
                        records.append(result)
                        (out / f"{tag}-{task['name']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                        print(f"{tag} {task['name']}: correct={result['correct']} seconds={result['seconds']:.2f} {content!r}", flush=True)
                finally:
                    stop(p)
            logtext = (out / (tag + ".server.log")).read_text(errors="replace")
            if re.search(r"buffer mapping failed|fastrpc_mmap failed", logtext):
                raise RuntimeError("DSP mapping failed; abort subsequent process launches")
    (out / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
