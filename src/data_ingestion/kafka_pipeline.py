import psutil
import subprocess
from kafka import KafkaConsumer
from dotenv import load_dotenv
import os

class KafkaSSHClient:
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
        """Start SSH tunnel using sshpass if password is provided, else ssh."""
        if self.password:
            tunnel_cmd = [
                "sshpass", "-p", self.password,
                "ssh", "-L", f"{self.local_port}:localhost:{self.remote_port}",
                f"{self.username}@{self.host}", "-NT"
            ]
        else:
            tunnel_cmd = [
                "ssh", "-L", f"{self.local_port}:localhost:{self.remote_port}",
                f"{self.username}@{self.host}", "-NT"
            ]

        self.ssh_proc = subprocess.Popen(
            tunnel_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def ensure_tunnel(self):
        """Ensure SSH tunnel is running, start it if not."""
        if not self._is_tunnel_running():
            print("SSH tunnel not running. Starting it...")
            self._start_tunnel()
        else:
            print("SSH tunnel already running.")

    def consume(self, topic, auto_offset_reset="earliest"):
        """Consume messages from a Kafka topic."""
        self.ensure_tunnel()

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=f"localhost:{self.local_port}",
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=True
        )

        print(f"Listening to topic: {topic}")
        for message in consumer:
            yield message.value.decode('utf-8')

    def close(self):
        """Close SSH tunnel process if started by this client."""
        if self.ssh_proc:
            self.ssh_proc.terminate()
            self.ssh_proc = None
            print("SSH tunnel closed.")


if __name__ == "__main__":
    load_dotenv()
    
    client = KafkaSSHClient(
        username=os.getenv("TUNNEL_USER"),
        host=os.getenv("TUNNEL_IP"),
        password=os.getenv("TUNNEL_PASSWORD"),
        local_port=os.getenv("TUNNEL_PORT"),
        remote_port=os.getenv("TUNNEL_PORT")
    )

    # Consume from topic "movielog10"
    count = 0
    for message in client.consume("movielog10"):
        print(message)
        if count >= 10:
            break
        count += 1