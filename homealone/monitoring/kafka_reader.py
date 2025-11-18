"""
Kafka Log Reader for Monitoring
Reads recommendation logs from Kafka stream for monitoring purposes
"""
import logging
import time
import psutil
import subprocess


logger = logging.getLogger(__name__)


class KafkaSSHClient:
    """SSH tunnel-based Kafka client."""

    def __init__(self, username, host, password, local_port, remote_port):
        self.username = username
        self.host = host
        self.password = password
        self.local_port = local_port
        self.remote_port = remote_port
        self.ssh_proc = None

    def _is_tunnel_running(self):
        """Check if SSH tunnel is already running."""
        for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and "ssh" in cmdline[0] and "-L" in cmdline:
                    if f"{self.local_port}:localhost:{self.remote_port}" in cmdline and f"{self.username}@{self.host}" in cmdline:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _start_tunnel(self):
        """Start SSH tunnel using sshpass."""
        logger.info(f"Starting SSH tunnel: localhost:{self.local_port} -> {self.username}@{self.host}:{self.remote_port}")

        tunnel_cmd = [
            "sshpass", "-p", self.password,
            "ssh", "-L", f"{self.local_port}:localhost:{self.remote_port}",
            f"{self.username}@{self.host}", "-NT"
        ]

        self.ssh_proc = subprocess.Popen(
            tunnel_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        logger.info("SSH tunnel started. Waiting for it to establish...")
        time.sleep(3)

    def ensure_tunnel(self):
        """Ensure SSH tunnel is running, start it if not."""
        if not self._is_tunnel_running():
            logger.info("SSH tunnel not running. Starting it...")
            self._start_tunnel()
        else:
            logger.info("SSH tunnel already running.")

