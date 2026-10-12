"""Disposable QEMU VM around SandboxEscapeBench's Docker socket scenario.

QEMU can run unprivileged inside a CPU allocation. The experimental Docker
daemon is inside the VM, never on a cluster node. No host directories are shared.
"""
import io
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time
import pycdlib
import yaml


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class VM:
    def __init__(self, config, directory, base, key, network=False):
        self.config, self.directory = config, Path(directory).resolve()
        self.base, self.key = Path(base).resolve(), Path(key).resolve()
        self.network = network
        self.directory.mkdir(parents=True, exist_ok=True)
        self.port = free_port()
        self.process = None
        self.tunnels = []
        self.log = None

    def ssh_args(self):
        return ["ssh", "-i", str(self.key), "-p", str(self.port),
                "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.directory / 'known_hosts'}",
                "-o", "LogLevel=ERROR", "ubuntu@127.0.0.1"]

    def start(self):
        for tool in ("qemu-system-x86_64", "qemu-img", "ssh"):
            if not shutil.which(tool):
                raise RuntimeError(f"Missing {tool}; see docs/running.md")
        image = self.directory / "disk.qcow2"
        subprocess.run(["qemu-img", "create", "-q", "-f", "qcow2", "-F", "qcow2",
                        "-b", str(self.base), str(image)], check=True)
        if self.network:
            subprocess.run(["qemu-img", "resize", "-q", str(image), "40G"], check=True)
        pub = self.key.with_suffix(".pub").read_text().strip()
        cloud = {"users": [{"name": "ubuntu", "sudo": "ALL=(ALL) NOPASSWD:ALL",
                            "shell": "/bin/bash", "ssh_authorized_keys": [pub]}],
                 "ssh_pwauth": False, "disable_root": True,
                 "write_files": [{"path": "/etc/ssh/sshd_config.d/80-loopwatch.conf",
                                   "content": "GatewayPorts clientspecified\nPasswordAuthentication no\n"}],
                 "runcmd": [["systemctl", "restart", "ssh"]]}
        iso = pycdlib.PyCdlib()
        iso.new(interchange_level=3, joliet=3, vol_ident="cidata")
        for filename, value in [("user-data", "#cloud-config\n" + yaml.safe_dump(cloud)),
                                 ("meta-data", yaml.safe_dump({"instance-id": str(self.directory), "local-hostname": "experiment-vm"}))]:
            data = value.encode()
            iso.add_fp(io.BytesIO(data), len(data), iso_path="/" + filename.upper().replace("-", "_") + ";1",
                       joliet_path="/" + filename)
        seed = self.directory / "seed.iso"
        iso.write(str(seed))
        iso.close()
        accel = self.config.get("acceleration", "auto")
        if accel == "auto":
            accel = "kvm" if os.access("/dev/kvm", os.R_OK | os.W_OK) else "tcg"
        self.accel = accel
        args = ["qemu-system-x86_64", "-machine", "q35", "-accel", accel,
                "-cpu", "host" if accel == "kvm" else "max",
                "-m", str(self.config["memory_mb"]), "-smp", str(self.config["cpus"]),
                "-drive", f"file={image},format=qcow2,if=virtio",
                "-drive", f"file={seed},format=raw,media=cdrom,readonly=on",
                "-netdev", f"user,id=net0,restrict={'off' if self.network else 'on'},hostfwd=tcp:127.0.0.1:{self.port}-:22",
                "-device", "virtio-net-pci,netdev=net0", "-display", "none", "-monitor", "none",
                "-serial", "stdio", "-no-reboot"]
        self.log = (self.directory / "console.log").open("wb")
        self.process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("VM exited during startup; inspect console.log")
            result = subprocess.run(self.ssh_args() + ["true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                initialized = self.exec("cloud-init status --wait", timeout=600, check=False)
                if initialized.returncode not in (0, 2):
                    raise RuntimeError("cloud-init failed; inspect console.log")
                return
            time.sleep(3)
        raise TimeoutError("VM did not become SSH-ready")

    def exec(self, command, *, timeout=600, check=True, input=None):
        return subprocess.run(self.ssh_args() + [command], input=input, capture_output=True,
                              timeout=timeout, check=check)

    def stream(self, command, output):
        return subprocess.Popen(self.ssh_args() + [command], stdout=output, stderr=subprocess.STDOUT)

    def tunnel(self, local_port):
        args = self.ssh_args()
        args[-1:-1] = ["-N", "-o", "ExitOnForwardFailure=yes", "-R", f"0.0.0.0:8000:127.0.0.1:{local_port}"]
        process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.tunnels.append(process)
        time.sleep(1)
        if process.poll() is not None:
            raise RuntimeError("API relay tunnel could not start")

    def put(self, local, remote):
        self.exec("sudo mkdir -p " + shlex.quote(str(Path(remote).parent)))
        self.exec("sudo tee " + shlex.quote(remote) + " >/dev/null", input=Path(local).read_bytes())

    def get(self, remote, local):
        result = self.exec("sudo cat " + shlex.quote(remote), check=False)
        if result.returncode == 0:
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            Path(local).write_bytes(result.stdout)
            return True
        return False

    def close(self):
        for tunnel in self.tunnels:
            tunnel.terminate()
            tunnel.wait(timeout=10)
        if self.process and self.process.poll() is None:
            try:
                self.exec("sudo poweroff", timeout=10, check=False)
                self.process.wait(timeout=60)
            except (subprocess.TimeoutExpired, OSError):
                self.process.terminate()
                self.process.wait(timeout=20)
        if self.log:
            self.log.close()
