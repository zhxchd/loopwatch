"""Kimi request relay outside the experimental VM; credentials never enter it."""
import http.server
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dotenv import dotenv_values
from runner.common import write_json


class Relay:
    def __init__(self, config, output, dotenv_path):
        self.config, self.output = config, output
        name = config["model"]["key_env"]
        # Consume only at runtime; never print or serialize the key or .env.
        self.key = os.environ.get(name) or dotenv_values(dotenv_path).get(name)
        if not self.key:
            raise RuntimeError(f"Missing {name}; set it in .env")
        self.responses = 0
        self.attempts = 0
        self.spend = 0.0
        self.last_error = None
        self.transport_version = 2
        self.lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.reply(404, {"error": {"message": "Unsupported endpoint"}})
                    return
                size = int(self.headers.get("Content-Length", 0))
                if size > 4_000_000:
                    self.reply(413, {"error": {"message": "Request too large"}})
                    return
                try:
                    data = json.loads(self.rfile.read(size))
                except (ValueError, TypeError):
                    self.reply(400, {"error": {"message": "Invalid JSON"}})
                    return
                model = outer.config["model"]
                messages = data.get("messages", [])
                omitted = [{"index": i, "message": message} for i, message in enumerate(messages)
                           if message.get("role") == "assistant" and message.get("content") in (None, "")
                           and not any(message.get(k) for k in ("tool_calls", "function_call", "reasoning_content"))]
                # A reasoning-only truncated completion has no command text.
                # AlgoTuner keeps an empty assistant turn, which Kimi rejects.
                # Omit only those empty turns on the wire; retain their evidence.
                omitted_indices = {item["index"] for item in omitted}
                request_body = {
                    "model": model["name"], "messages": [m for i, m in enumerate(messages) if i not in omitted_indices],
                    "temperature": model["temperature"], "max_tokens": model["max_tokens"],
                    "thinking": {"type": model["thinking"]}, "stream": False,
                }
                with outer.lock:
                    # Byte count is a conservative token bound, with room for
                    # chat framing. Reserve the entire requested completion.
                    bound = ((len(json.dumps(request_body["messages"]).encode()) + 4096) * model["input_price"]
                             + model["max_tokens"] * model["output_price"]) / 1e6
                    if outer.spend + bound > outer.config["budget"]["dollars"] or outer.attempts >= outer.config["budget"]["messages"] + 5:
                        self.reply(402, {"error": {"message": "External episode budget exhausted"}})
                        return
                    outer.attempts += 1
                    outer.spend += bound  # Retain reservation if billing is unknown after an error.
                    request = urllib.request.Request(
                        model["api_base"].rstrip("/") + "/chat/completions",
                        data=json.dumps(request_body).encode(),
                        headers={"Authorization": "Bearer " + outer.key, "Content-Type": "application/json"},
                    )
                    started = time.time()
                    try:
                        with urllib.request.urlopen(request, timeout=300) as response:
                            result = json.load(response)
                        usage = result.get("usage") or {}
                        if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                            raise ValueError("Missing usage")
                        cost = (usage["prompt_tokens"] * model["input_price"] + usage["completion_tokens"] * model["output_price"]) / 1e6
                        usage["cost"] = cost
                        result["usage"] = usage
                        outer.spend += cost - bound
                        outer.responses += 1
                        with (outer.output / "trajectory.jsonl").open("a") as stream:
                            stream.write(json.dumps({"request": request_body, "response": result,
                                                     "omitted_empty_assistant_messages": omitted,
                                                     "started": started, "finished": time.time()}) + "\n")
                        outer.save()
                        self.reply(200, result)
                    except urllib.error.HTTPError as exc:
                        detail = exc.read(8192).decode(errors="replace").replace(outer.key, "[REDACTED]")
                        with (outer.output / "api_errors.jsonl").open("a") as stream:
                            stream.write(json.dumps({"request": request_body, "status": exc.code,
                                                     "detail": detail,
                                                     "omitted_empty_assistant_messages": omitted,
                                                     "started": started, "finished": time.time()}) + "\n")
                        outer.save(error=f"Kimi HTTP {exc.code}")
                        self.reply(exc.code, {"error": {"message": f"Kimi HTTP {exc.code}"}})
                    except Exception as exc:
                        outer.save(error=type(exc).__name__)
                        self.reply(502, {"error": {"message": "Kimi request failed"}})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def save(self, **extra):
        if "error" in extra:
            self.last_error = extra["error"]
        write_json(self.output / "relay.json", {"responses": self.responses, "attempts": self.attempts,
                   "conservative_cost_usd": self.spend, "last_error": self.last_error,
                   "transport_version": self.transport_version, **extra})

    def start(self):
        self.thread.start()
        self.save()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.save()
