import numpy as np
from datetime import datetime

def extract_features(packet_data):
    """
    Extracts 39 meaningful features from network packets for anomaly detection.
    
    Args:
        packet_data: List of pyshark packet objects
        
    Returns:
        numpy.ndarray: Array of shape (n_packets, 39) containing extracted features
    """
    features = []
    prev_timestamp = None
    
    for packet in packet_data:
        try:
            # Basic packet information
            length = float(packet.length) if hasattr(packet, 'length') else 0
            protocol = float(packet.highest_layer.encode("utf-8")[0]) if hasattr(packet, 'highest_layer') else 0
            timestamp = float(packet.sniff_time.timestamp()) if hasattr(packet, 'sniff_time') else 0
            
            # Time delta calculations
            time_delta = 0.0
            if prev_timestamp is not None:
                time_delta = timestamp - prev_timestamp
            prev_timestamp = timestamp
            
            # IP layer features
            ip_features = [0] * 5
            if hasattr(packet, 'ip'):
                ip_features = [
                    float(packet.ip.ttl) if hasattr(packet.ip, 'ttl') else 0,
                    1 if hasattr(packet.ip, 'flags') and 'df' in packet.ip.flags else 0,
                    len(packet.ip.src.split('.')) if hasattr(packet.ip, 'src') else 0,
                    len(packet.ip.dst.split('.')) if hasattr(packet.ip, 'dst') else 0,
                    float(packet.ip.len) if hasattr(packet.ip, 'len') else 0
                ]
            
            # TCP layer features
            tcp_features = [0] * 8
            if hasattr(packet, 'tcp'):
                tcp_features = [
                    float(packet.tcp.srcport) if hasattr(packet.tcp, 'srcport') else 0,
                    float(packet.tcp.dstport) if hasattr(packet.tcp, 'dstport') else 0,
                    float(packet.tcp.window_size) if hasattr(packet.tcp, 'window_size') else 0,
                    float(packet.tcp.hdr_len) if hasattr(packet.tcp, 'hdr_len') else 0,
                    sum(int(flag) for flag in [
                        hasattr(packet.tcp, 'flags_ack') and packet.tcp.flags_ack == '1',
                        hasattr(packet.tcp, 'flags_syn') and packet.tcp.flags_syn == '1',
                        hasattr(packet.tcp, 'flags_fin') and packet.tcp.flags_fin == '1',
                        hasattr(packet.tcp, 'flags_push') and packet.tcp.flags_push == '1',
                        hasattr(packet.tcp, 'flags_reset') and packet.tcp.flags_reset == '1'
                    ]),
                    float(packet.tcp.seq) if hasattr(packet.tcp, 'seq') else 0,
                    float(packet.tcp.ack) if hasattr(packet.tcp, 'ack') else 0,
                    float(packet.tcp.time_delta) if hasattr(packet.tcp, 'time_delta') else 0
                ]
            
            # UDP layer features
            udp_features = [0] * 3
            if hasattr(packet, 'udp'):
                udp_features = [
                    float(packet.udp.srcport) if hasattr(packet.udp, 'srcport') else 0,
                    float(packet.udp.dstport) if hasattr(packet.udp, 'dstport') else 0,
                    float(packet.udp.length) if hasattr(packet.udp, 'length') else 0
                ]
            
            # ICMP layer features
            icmp_features = [0] * 2
            if hasattr(packet, 'icmp'):
                icmp_features = [
                    float(packet.icmp.type) if hasattr(packet.icmp, 'type') else 0,
                    float(packet.icmp.code) if hasattr(packet.icmp, 'code') else 0
                ]
            
            # Combine all features
            extracted_features = [
                length, protocol, timestamp, time_delta,  # Basic features
                *ip_features,                              # IP layer features (5)
                *tcp_features,                             # TCP layer features (8)
                *udp_features,                            # UDP layer features (3)
                *icmp_features,                           # ICMP layer features (2)
                # Additional statistical features
                len(packet.layers) if hasattr(packet, 'layers') else 0,  # Number of layers
                int(hasattr(packet, 'eth')),              # Has Ethernet layer
                int(hasattr(packet, 'ipv6')),             # Has IPv6
                int(hasattr(packet, 'arp')),              # Has ARP
                int(hasattr(packet, 'dns'))               # Has DNS
            ]
            
            # Ensure exactly 39 features
            if len(extracted_features) != 39:
                raise ValueError(f"Expected 39 features, got {len(extracted_features)}")
            
            features.append(extracted_features)
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            # On error, use zero vector but maintain 39 features
            features.append([0] * 39)
            continue

    return np.array(features, dtype=np.float32) if features else None