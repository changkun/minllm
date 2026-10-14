"""
Serve a model over HTTP, speaking the protocol of OpenAI's chat completions
API, so that any client written for that API can talk to it.

    uv run scripts/serve.py --model out/sft-words
    curl localhost:8000/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Write a short story using these words: garden, kitten."}]}'

Requests that arrive together are answered together: they join a running
batch of --slots rows between two steps, and leave it when they end
(docs/04c-continuous-batching.md). See docs/10-stack.md for what a
production server does besides.
"""

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from minllm import chat, checkpoint
from minllm.batching import Batcher
from minllm.common import pick_device
from minllm.data import EOT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen", help="qwen, or the directory of a trained model")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--slots", type=int, default=16, help="requests answered at once")
parser.add_argument("--context", type=int, default=4096, help="tokens a request may hold, prompt and reply: every slot is allocated for this many")
args = parser.parse_args()

device = pick_device()
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
stop = [tok.special_tokens[chat.IM_END], tok.special_tokens[EOT]]
batcher = Batcher(model, args.slots, args.context, stop=stop)
room = batcher.cache.max_len
work = threading.Condition()  # guards the batcher, which the request threads add to
listeners = {}  # request id -> the queue its tokens go to


def run():
    """The one thread that uses the GPU: a step whenever there is work, and every new token to the request it belongs to."""
    while True:
        with work:
            while not batcher.busy():
                work.wait()
            moved = batcher.step()
        for request, token, done in moved:
            listeners[request].put((token, done))
            if done:
                del listeners[request]


def pieces(tokens):
    """Text as it is written. A token can end in the middle of a multi-byte character, so a piece is held back until the bytes so far decode cleanly."""
    ids, shown = [], ""
    for token in tokens:
        ids.append(token)
        text = tok.decode(ids)
        if not text.endswith("�"):
            yield text[len(shown) :]
            shown = text


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
        max_tokens = min(request.get("max_tokens", 256), room - len(prompt))
        if max_tokens <= 0:
            return self.reply(400, {"error": {"message": "the conversation does not fit the model's context"}})
        base = {"id": f"chatcmpl-{time.time_ns()}", "created": int(time.time()), "model": args.model}

        tokens = queue.Queue()
        with work:
            listeners[batcher.add(prompt, max_tokens, temperature=request.get("temperature", 0.7), top_p=request.get("top_p"))] = tokens
            work.notify()
        ending = {}

        def written():
            while True:
                token, done = tokens.get()
                if token is not None:
                    yield token
                if done:
                    ending["reason"] = "stop" if token is None else "length"
                    return

        if not request.get("stream"):
            message = {"role": "assistant", "content": "".join(pieces(written()))}
            choice = {"index": 0, "message": message, "finish_reason": ending["reason"]}
            return self.reply(200, base | {"object": "chat.completion", "choices": [choice]})
        # Streaming: server-sent events, one per piece of text, as the OpenAI API sends them.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in pieces(written()):
            chunk = base | {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format, *arguments):
        print(f"{self.address_string()} {format % arguments}")


threading.Thread(target=run, daemon=True).start()
print(f"serving {args.model} on http://localhost:{args.port}/v1/chat/completions ({device}, {args.slots} requests at once)")
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
