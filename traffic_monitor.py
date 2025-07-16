import sys
import numpy as np
import pyshark
import tensorflow as tf
import joblib
import time
import matplotlib.pyplot as plt
import asyncio
import logging
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QFrame, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load scaler and ML models
scaler = joblib.load('scaler.pkl')
lstm_model = tf.keras.models.load_model('lstm_model.h5')
gan_model = tf.keras.models.load_model('gan_generator.h5')
lstm_model = tf.keras.models.load_model('lstm_model.h5')
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Added compilation

gan_model = tf.keras.models.load_model('gan_generator.h5')
gan_model.compile(optimizer='adam', loss='binary_crossentropy')  # Added compilation



class NetworkTrafficDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Live Network Traffic Metrics")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #000000; color: #E0E0E0;")  # Dark Mode

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # **Network Traffic Metrics**
        top_layout = QHBoxLayout()
        self.network_traffic_label = self.create_stat_card("Network Traffic (MB)", "0", "#2196F3")
        self.unique_ips_label = self.create_stat_card("Unique IPs", "0", "#4CAF50")
        self.anomaly_label = self.create_stat_card("Traffic Status", "Normal ✅", "#FF9800")
        top_layout.addWidget(self.network_traffic_label)
        top_layout.addWidget(self.unique_ips_label)
        top_layout.addWidget(self.anomaly_label)
        main_layout.addLayout(top_layout)

        # **Traffic Trend Graph**
        self.trend_frame = QFrame()
        self.trend_frame.setStyleSheet("background-color: #000000; padding: 10px; border-radius: 8px; border: 2px solid #444;")
        trend_layout = QVBoxLayout(self.trend_frame)
        self.trend_fig, self.trend_ax = plt.subplots(facecolor='#000000')
        self.trend_canvas = FigureCanvas(self.trend_fig)
        trend_layout.addWidget(self.trend_canvas)
        main_layout.addWidget(self.trend_frame)

        # **Capture Packets Button**
        self.capture_button = QPushButton("Capture Packets")
        self.capture_button.setStyleSheet(
            "background-color: #FF9800; color: #000000; padding: 10px; font-size: 16px; border-radius: 5px; font-weight: bold;"
        )
        self.capture_button.clicked.connect(self.start_packet_capture)
        main_layout.addWidget(self.capture_button, alignment=Qt.AlignCenter)

        central_widget.setLayout(main_layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(3000)

        self.network_traffic = 0
        self.unique_ips = set()
        self.traffic_trend = []
        self.capturing = False
        
        # For SYN flood detection
        self.syn_packets = deque(maxlen=100)  # Track last 100 SYN packets
        self.last_update_time = time.time()
        self.alert_active = False
        self.alert_end_time = 0

    def create_stat_card(self, title, value, color):
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {color}; padding: 15px; border-radius: 10px; border: 2px solid #444;")
        layout = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        value_label = QLabel(value)
        value_label.setFont(QFont("Arial", 18, QFont.Bold))
        value_label.setAlignment(Qt.AlignCenter)
        frame.value_label = value_label
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        frame.setLayout(layout)
        return frame

    def update_dashboard(self):
        # Check if we should clear an alert
        if self.alert_active and time.time() > self.alert_end_time:
            self.alert_active = False
            self.anomaly_label.value_label.setText("Normal ✅")
            self.anomaly_label.setStyleSheet(f"background-color: #FF9800; padding: 15px; border-radius: 10px; border: 2px solid #444;")

        self.network_traffic_label.value_label.setText(f"{round(self.network_traffic, 4)} MB")
        self.unique_ips_label.value_label.setText(str(len(self.unique_ips)))

        self.trend_ax.clear()
        self.trend_ax.plot(self.traffic_trend, marker='o', linestyle='-', color='yellow', linewidth=1)
        self.trend_ax.set_title("Total Network Traffic Trend", color="white")
        self.trend_ax.set_facecolor("#000000")
        self.trend_ax.grid(True, linestyle="--", linewidth=0.5, color='gray')
        self.trend_ax.tick_params(colors='white')
        self.trend_canvas.draw()

    def detect_syn_flood(self, packet):
        """Enhanced SYN flood detection with better logging and adaptive threshold"""
        try:
            # Check for TCP SYN packets
            if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'flags'):
                flags = int(packet.tcp.flags, 16)
                if flags & 0x02:  # SYN flag is set
                    src_ip = packet.ip.src if hasattr(packet, 'ip') else 'unknown'
                    self.syn_packets.append((time.time(), src_ip))
                    
                    # Calculate SYN rate from unique sources (more accurate detection)
                    current_time = time.time()
                    recent_syns = [(t, ip) for t, ip in self.syn_packets if current_time - t < 5]
                    unique_syns = len({ip for t, ip in recent_syns})
                    total_syns = len(recent_syns)
                    
                    logging.info(f"SYN packet from {src_ip} - Total SYNs: {total_syns}, Unique IPs: {unique_syns}")
                    
                    # Dynamic threshold - alert if either:
                    # 1. High total SYNs from few sources (classic SYN flood)
                    # 2. Moderate SYNs from many sources (distributed attack)
                    if total_syns > 15 or (unique_syns > 10 and total_syns > 30):
                        self.alert_active = True
                        self.alert_end_time = current_time + 15  # Extend alert duration
                        self.anomaly_label.value_label.setText("SYN Flood Detected! ⚠️")
                        self.anomaly_label.setStyleSheet(
                            "background-color: #D32F2F; padding: 15px; border-radius: 10px; border: 2px solid #444;"
                        )
                        logging.warning(
                            f"🚨 SYN FLOOD DETECTED! {total_syns} SYNs from {unique_syns} IPs in last 5s"
                        )
                        
                        # Visual indicator on graph
                        if len(self.traffic_trend) > 10:
                            self.trend_ax.axvspan(
                                len(self.traffic_trend)-5, len(self.traffic_trend),
                                color='red', alpha=0.3, label='Attack'
                            )
                            self.trend_canvas.draw()
        except Exception as e:
            logging.error(f"SYN detection error: {e}")

    def start_packet_capture(self):
        if not self.capturing:
            self.capturing = True
            self.capture_button.setText("Capturing...")
            self.capture_button.setStyleSheet("background-color: #D32F2F; color: #FFFFFF; font-weight: bold;")
            threading.Thread(target=self.capture_packets, daemon=True).start()
        else:
            self.capturing = False
            self.capture_button.setText("Capture Packets")
            self.capture_button.setStyleSheet("background-color: #FF9800; color: #000000; font-weight: bold;")

    def capture_packets(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Fix 2: Remove buffer_size parameter and add proper cleanup
            capture = pyshark.LiveCapture(
                interface=r"\Device\NPF_{45084FBA-0585-423B-85C9-0E31A347C32C}",
                display_filter="tcp.flags.syn==1 and tcp.flags.ack==0"
            )

            # Fix 3: Add proper cleanup handling
            def cleanup():
                if hasattr(capture, '_running_processes'):
                    capture.close()

            try:
                for packet in capture.sniff_continuously():
                    if not self.capturing:
                        break

                    try:
                        packet_size = int(packet.length) if hasattr(packet, 'length') else 0
                        self.network_traffic += packet_size / (1024 * 1024)

                        if hasattr(packet, "ip") and hasattr(packet.ip, "src"):
                            self.unique_ips.add(packet.ip.src)

                        self.traffic_trend.append(self.network_traffic)
                        self.detect_syn_flood(packet)

                        if len(self.traffic_trend) > 100:
                            self.traffic_trend.pop(0)

                        if time.time() - self.last_update_time > 0.5:
                            QTimer.singleShot(0, self.update_dashboard)
                            self.last_update_time = time.time()

                    except AttributeError as e:
                        logging.debug(f"Skipping malformed packet: {e}")

            finally:
                cleanup()

        except Exception as e:
            logging.error(f"Capture error: {e}")
        finally:
            self.capturing = False
            self.capture_button.setText("Capture Packets")
            self.capture_button.setStyleSheet(
                "background-color: #FF9800; color: #000000; font-weight: bold;"
            )
    def update_dashboard(self):
        """Enhanced dashboard update with attack visualization"""
        if self.alert_active:
            if time.time() > self.alert_end_time:
                self.alert_active = False
                self.anomaly_label.value_label.setText("Normal ✅")
                self.anomaly_label.setStyleSheet(
                    "background-color: #FF9800; padding: 15px; border-radius: 10px; border: 2px solid #444;"
                )

        self.network_traffic_label.value_label.setText(f"{self.network_traffic:.4f} MB")
        self.unique_ips_label.value_label.setText(str(len(self.unique_ips)))

        self.trend_ax.clear()
        self.trend_ax.plot(
            self.traffic_trend, 
            marker='o', 
            linestyle='-', 
            color='yellow', 
            linewidth=1,
            label='Traffic'
        )
        
        # Add attack indicators if any
        if self.alert_active:
            attack_start = max(0, len(self.traffic_trend)-15)
            self.trend_ax.axvspan(
                attack_start, len(self.traffic_trend),
                color='red', alpha=0.2, label='Attack'
            )
        
        self.trend_ax.set_title("Network Traffic with Attack Detection", color="white")
        self.trend_ax.legend()
        self.trend_ax.set_facecolor("#000000")
        self.trend_ax.grid(True, linestyle="--", linewidth=0.5, color='gray')
        self.trend_ax.tick_params(colors='white')
        self.trend_canvas.draw()

def extract_features(packet_data):
    features = []
    for packet in packet_data:
        try:
            length = float(packet.length) if hasattr(packet, 'length') else 0
            protocol = float(packet.highest_layer.encode("utf-8")[0]) if hasattr(packet, 'highest_layer') else 0
            src_port = float(packet.tcp.srcport) if hasattr(packet, 'tcp') else 0

            full_features = [length, protocol, src_port] + [0] * (39 - 3)
            features.append(full_features)

        except Exception as e:
            logging.error(f"❌ Feature extraction error: {e}")

    return np.array(features, dtype=np.float32) if features else None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NetworkTrafficDashboard()
    window.show()
    sys.exit(app.exec_())