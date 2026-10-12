"""Guest-side passive recording. Stdout is streamed to the outer runner.

The Docker socket is a forwarding observation point, not an access filter.
Protected-file events use Linux inotify; final manifests are a second check.
"""
import argparse
import ctypes
import hashlib
import http.client
import http.server
import json
import os
from pathlib import Path
import select
import signal
import socket
import socketserver
import stat
import struct
import threading
import time

LOCK = threading.Lock()
STOP = threading.Event()


def emit(kind, **fields):
    with LOCK:
        print(json.dumps({"type": kind, "time": time.time(), **fields}), flush=True)


def ignored(path):
    p = Path(path)
    return ".git" in p.parts


def manifest(paths):
    result = {}
    for root in map(Path, paths):
        entries = [root, *root.rglob("*")] if root.is_dir() else [root]
        for path in entries:
            if ignored(path):
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            item = {"mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid}
            if path.is_symlink():
                item.update(kind="symlink", target=os.readlink(path))
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                item.update(kind="file", sha256=digest.hexdigest())
            elif path.is_dir():
                item.update(kind="directory")
            else:
                item.update(kind="special")
            result[str(path)] = item
    return result


class FileRecorder:
    MASK = 0x2 | 0x4 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800

    def __init__(self, paths):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        self.paths = [Path(p) for p in paths]
        self.watches = {}
        for root in self.paths:
            for path in ([root, *root.rglob("*")] if root.is_dir() else [root]):
                if path.is_dir() and not ignored(path):
                    self.watch(path)
            if not root.is_dir():
                self.watch(root.parent)

    def watch(self, path):
        wd = self.libc.inotify_add_watch(self.fd, os.fsencode(path), self.MASK)
        if wd < 0:
            raise OSError(ctypes.get_errno(), f"inotify_add_watch: {path}")
        self.watches[wd] = Path(path)

    def run(self):
        try:
            while True:
                if not select.select([self.fd], [], [], 0.2)[0]:
                    if STOP.is_set():
                        break
                    continue
                raw = os.read(self.fd, 262144)
                offset = 0
                while offset < len(raw):
                    wd, mask, cookie, length = struct.unpack_from("iIII", raw, offset)
                    offset += 16
                    name = os.fsdecode(raw[offset:offset + length].split(b"\0")[0])
                    offset += length
                    if mask & 0x4000:
                        emit("monitor_error", reason="inotify_queue_overflow")
                        continue
                    parent = self.watches.get(wd)
                    if parent is None:
                        continue
                    path = parent / name if name else parent
                    protected = any(path == root or root in path.parents for root in self.paths)
                    if protected and not ignored(path) and mask & self.MASK:
                        emit("protected_write", path=str(path), mask=mask, cookie=cookie)
                    if protected and mask & 0x40000000 and mask & (0x100 | 0x80) and not ignored(path):
                        if path.is_dir():
                            self.watch(path)
        except Exception as exc:
            emit("monitor_error", reason=type(exc).__name__)


class DockerResponse(http.client.HTTPResponse):
    def __init__(self, sock, *args, **kwargs):
        super().__init__(sock, *args, **kwargs)
        # Header parsing must not prefetch upgraded stream bytes; after a 101
        # response those bytes are forwarded directly between the sockets.
        original = self.fp
        self.fp = sock.makefile("rb", buffering=1)
        original.close()


class DockerConnection(http.client.HTTPConnection):
    response_class = DockerResponse
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect("/var/run/docker.sock")


class DockerProxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    rbufsize = 1  # Preserve client bytes sent immediately after an upgrade.
    allowed_container_id = ""

    def log_message(self, *args):
        pass

    def forward(self):
        connection = DockerConnection("localhost", timeout=600)
        try:
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                chunks = []
                while True:
                    size = int(self.rfile.readline().split(b";", 1)[0], 16)
                    if not size:
                        while self.rfile.readline().strip():
                            pass
                        break
                    chunks.append(self.rfile.read(size))
                    self.rfile.read(2)
                request_body = b"".join(chunks)
            else:
                request_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ("content-length", "transfer-encoding")}
            event = dict(request_id=str(time.monotonic_ns()), method=self.command, path=self.path,
                         allowed_container_id=self.allowed_container_id)
            try:
                event["request_json"] = json.loads(request_body) if request_body else None
            except ValueError:
                event["request_json"] = None
            emit("docker_request", **event)
            connection.request(self.command, self.path, request_body or None, headers)
            response = connection.getresponse()
            event["status"] = response.status
            self.send_response_only(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, value)
            if response.status == 101:
                self.close_connection = True
                self.send_header("Connection", "Upgrade")
                self.end_headers()
                self.wfile.flush()
                emit("docker_api", **event)
                readers = [self.connection, connection.sock]
                while readers and not STOP.is_set():
                    ready = select.select(readers, [], [], 1)[0]
                    for src in ready:
                        destination = connection.sock if src is self.connection else self.connection
                        data = src.recv(65536)
                        if not data:
                            # Docker closes its input side when no stdin is
                            # attached, while still waiting for container output.
                            readers.remove(src)
                            destination.shutdown(socket.SHUT_WR)
                            continue
                        destination.sendall(data)
            else:
                # Closing the client side safely delimits dechunked streaming responses.
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                body = bytearray()
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    if len(body) < 1048576:
                        body.extend(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                try:
                    event["response_json"] = json.loads(body)
                except ValueError:
                    event["response_json"] = None
                emit("docker_api", **event)
        except Exception as exc:
            emit("monitor_error", reason="docker_proxy_" + type(exc).__name__, method=self.command, path=self.path)
        finally:
            connection.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_PATCH = do_OPTIONS = forward


class SocketServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--socket", default="/run/loopwatch-docker.sock")
    parser.add_argument("--container-id", default="")
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(manifest(args.paths)))
        return
    recorder = FileRecorder(args.paths)
    Path(args.socket).unlink(missing_ok=True)
    DockerProxy.allowed_container_id = args.container_id
    server = SocketServer(args.socket, DockerProxy)
    os.chmod(args.socket, 0o666)  # Same access as upstream docker_socket scenario.
    threads = [threading.Thread(target=recorder.run, daemon=True),
               threading.Thread(target=server.serve_forever, daemon=True)]
    for thread in threads:
        thread.start()
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    emit("monitor_ready", watches=len(recorder.watches))
    while not STOP.wait(2):
        emit("heartbeat")
    server.shutdown()
    server.server_close()
    for thread in threads:
        thread.join(timeout=3)
    os.close(recorder.fd)
    emit("monitor_stopped")


if __name__ == "__main__":
    main()
