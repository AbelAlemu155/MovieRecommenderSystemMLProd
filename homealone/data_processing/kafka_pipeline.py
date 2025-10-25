import psutil
import subprocess
from kafka import KafkaConsumer
from dotenv import load_dotenv
import os

GROUP_ID = "team10-consumer-group"

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

    def consume(self, topic="movielog10", group_id=GROUP_ID, auto_offset_reset="earliest"):
        """
        Consume messages from a Kafka topic with manual offset control.
        
        Args:
            topic: Kafka topic name
            group_id: Consumer group ID for tracking offsets
            auto_offset_reset: Where to start if no committed offset exists
        """
        self.ensure_tunnel()


        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=f"localhost:{self.local_port}",
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=False,
        )


        print(f"Listening to topic: {topic} with group_id: {group_id}")
        
        try:
            for message in consumer:
                yield message.value.decode('utf-8')
        finally:
            # ADDED: Commit offsets when done consuming
            print("Committing offsets and closing consumer.")
            consumer.commit()
            consumer.close()

    def consume_batch(self, topic="movielog10", group_id=GROUP_ID, batch_size=10, auto_offset_reset="earliest"):
        """
        Consume a specific number of messages and commit offset.
        
        Args:
            topic: Kafka topic name
            group_id: Consumer group ID for tracking offsets
            batch_size: Number of messages to consume before stopping
            auto_offset_reset: Where to start if no committed offset exists
        
        Returns:
            List of consumed messages
        """
        self.ensure_tunnel()
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=f"localhost:{self.local_port}",
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=False,
            consumer_timeout_ms=10000
        )
        
        messages = []
        print(f"Consuming {batch_size} messages from topic: {topic}")
        
        try:
            count = 0
            for message in consumer:
                messages.append(message.value.decode('utf-8'))
                count += 1
                
                if count >= batch_size:
                    break
            consumer.commit()
            print(f"Successfully consumed and committed {count} messages")
            
        except Exception as e:
            print(f"Error during consumption: {e}")
        finally:
            consumer.close()
        
        return messages


    def close(self):
        """Close SSH tunnel process if started by this client."""
        if self.ssh_proc:
            self.ssh_proc.terminate()
            self.ssh_proc = None
            print("SSH tunnel closed.")

# run this to test the kafka ssh client
if __name__ == "__main__":
    load_dotenv()

    while True:
        try:
            print("Starting Kafka SSH Client...")
            client = KafkaSSHClient(
                username=os.getenv("TUNNEL_USER"),
                host=os.getenv("TUNNEL_IP"),
                password=os.getenv("TUNNEL_PASSWORD"),
                local_port=os.getenv("TUNNEL_PORT"),
                remote_port=os.getenv("TUNNEL_PORT")
            )
            
            # First batch
            messages_batch1 = client.consume_batch("movielog10", group_id=GROUP_ID, batch_size=10)
            print(f"First batch: {len(messages_batch1)} messages")
            
            # Second batch - will automatically resume from where first batch stopped
            messages_batch2 = client.consume_batch("movielog10", group_id=GROUP_ID, batch_size=10)
            print(f"Second batch: {len(messages_batch2)} messages")
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")