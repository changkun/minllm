"""
Serve a model over HTTP, speaking the protocol of OpenAI's chat completions
API, so that any client written for that API can talk to it.

    uv run scripts/serve.py --model out/sft-words
    curl localhost:8000/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Write a short story using these words: garden, kitten."}]}'

This server answers one request at a time. See docs/10-stack.md for what a
production server does instead.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from minllm import chat, checkpoint
from minllm.common import pick_device
from minllm.data import EOT
from minllm.inference import stream_text

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen", help="qwen, or the directory of a trained model")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

device = pick_device()
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
stop = [tok.special_tokens[chat.IM_END], tok.special_tokens[EOT]]
gpu = threading.Lock()  # one generation at a time


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            return self.reply(200, {"object": "list", "data": [{"id": args.model, "object": "model"}]})
        self.reply(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self.reply(404, {"error": {"message": "not found"}})
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = chat.encode(tok, request["messages"])
        except (KeyError, ValueError, TypeError) as error:
            return self.reply(400, {"error": {"message": f"bad request: {error}"}})
        sampling = dict(temperature=request.get("temperature", 0.7), top_p=request.get("top_p"), stop=stop)
        max_tokens = min(request.get("max_tokens", 256), model.cfg.max_seq_len - len(prompt))
        if max_tokens <= 0:
            return self.reply(400, {"error": {"message": "the conversation does not fit the model's context"}})
        base = {"id": f"chatcmpl-{time.time_ns()}", "created": int(time.time()), "model": args.model}

        with gpu:
            pieces = stream_text(model, tok, prompt, max_tokens, **sampling)
            if not request.get("stream"):
                message = {"role": "assistant", "content": "".join(pieces)}
                choice = {"index": 0, "message": message, "finish_reason": "stop"}
                return self.reply(200, base | {"object": "chat.completion", "choices": [choice]})
            # Streaming: server-sent events, one per piece of text, as the OpenAI API sends them.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in pieces:
                chunk = base | {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format, *arguments):
        print(f"{self.address_string()} {format % arguments}")


print(f"serving {args.model} on http://localhost:{args.port}/v1/chat/completions ({device})")
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
