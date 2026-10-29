import cv2 as cv
from ultralytics import YOLO
import argparse
import os
import queue
import socket
import json
import threading
import time
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_PI_HOST = "your_pi_IP_adress"
PAN_DIRECTION = -1
TILT_DIRECTION = 1
TRACKING_GAIN = 0.3
MAX_SERVO_STEP = 2
MIN_SERVO_ANGLE = -89.0
MAX_SERVO_ANGLE = 89.0


class PiServoClient:
    def __init__(self, host, port=8765):
        self.host = host.strip()
        self.port = port
        self.connection = None
        self.reader = None
        self._commands = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._send_commands,
            name="servo-sender",
            daemon=True,
        )
        self._worker.start()

    def connect(self):
        try:
            self.connection = socket.create_connection(
                (self.host, self.port), timeout=5
            )
        except socket.gaierror as error:
            raise RuntimeError(
                f"Cannot resolve Raspberry Pi host '{self.host}'. "
                "Use the Pi hostname or pass it with --pi-host."
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"Cannot connect to Raspberry Pi at {self.host}:{self.port}. "
                "Check that pi_servo_server.py is running and the IP/port are correct."
            ) from error
        self.reader = self.connection.makefile("r", encoding="utf-8")
        print(f"[servo] Connected to Raspberry Pi at {self.host}:{self.port}")

    def _send_angles(self, pan, tilt):
        if self.connection is None:
            self.connect()

        request = {
            "command": "set_angles",
            "pan": float(pan),
            "tilt": float(tilt),
        }
        try:
            self.connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
            response = json.loads(self.reader.readline())
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Pi rejected servo command"))
        except (OSError, json.JSONDecodeError, RuntimeError):
            if self.reader is not None:
                self.reader.close()
            if self.connection is not None:
                self.connection.close()
            self.reader = None
            self.connection = None
            raise

    def _send_commands(self):
        while not self._stop_event.is_set():
            try:
                command = self._commands.get(timeout=0.1)
            except queue.Empty:
                continue

            if command is None:
                return

            pan, tilt = command
            try:
                self._send_angles(pan, tilt)
            except (OSError, RuntimeError, json.JSONDecodeError) as error:
                print(f"[servo] Failed to send angles: {error}")

    def set_angles(self, pan, tilt):
        if self._stop_event.is_set():
            raise RuntimeError("Servo client is closed")

        command = (float(pan), float(tilt))
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                pass
            self._commands.put_nowait(command)

    def close(self):
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        try:
            self._commands.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=6)
        if self.reader is not None:
            self.reader.close()
        if self.connection is not None:
            self.connection.close()
        self.reader = None
        self.connection = None


class CameraStream:
    CELL_PHONE_CLASS_ID = 67 #0 - person, 67 - cell phone, 39 - bottle, 56 - chair
    # yolo26: 61 - cell phone

    def __init__(self, camera_index=1, pi_host="raspberrypi.local", device="cpu"):
        print(f"[startup] Opening camera index {camera_index}")
        self.camera = cv.VideoCapture(camera_index)
        print(f"[startup] Camera opened: {self.camera.isOpened()}")

        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, 960) # adjust for performance
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
        print(
            "[startup] Camera resolution:",
            f"{int(self.camera.get(cv.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self.camera.get(cv.CAP_PROP_FRAME_HEIGHT))}",
        )



        print("[startup] Loading YOLO model from yolo26m_original.engine")
        self.model = YOLO("yolo26m_original.engine")
        self.device = device
        print(f"[startup] YOLO device: {self.device}")
        print("[startup] YOLO model loaded")

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True

        self.HFOV = 66.0174
        self.VFOV = 40.1392

        self.servos = PiServoClient(pi_host)

        self.pan_angle_previous = 0
        self.tilt_angle_previous = 0

        self.pan_angle_current = 0
        self.tilt_angle_current = 0
        self._last_frame_error_log = 0
        self._last_no_detection_log = 0
        self._start_time = time.perf_counter()
        self.inference_times = []
        self.inference_durations = []
        self.servo_times = []
        self.pan_commands = []
        self.tilt_commands = []
        self.phone_times = []
        self.phone_pan_angles = []
        self.phone_tilt_angles = []

    def stop(self):
        print("[shutdown] Stopping camera stream")
        self.running = False
        self.servos.close()
        self.camera.release()
        cv.destroyAllWindows()
        #self.plot_run_data()

    def predict(self, frame):
        if frame is None:
            return []

        start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            imgsz=320,
            conf=0.4,
            device=self.device,
            verbose=False,
        )
        self.inference_times.append(start - self._start_time)
        self.inference_durations.append(time.perf_counter() - start)

        return results

    def current_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def update(self):
        print("[camera] Capture thread started")
        while self.running:
            ret, frame = self.camera.read()

            if not ret or frame is None:
                now = time.time()
                if now - self._last_frame_error_log >= 2:
                    print(f"[camera] Failed to read frame (ret={ret})")
                    self._last_frame_error_log = now
                continue

            if self.frame_queue.full():
                self.frame_queue.get_nowait()

            self.frame_queue.put(frame)
            
    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        print("[camera] Capture thread launched")
        return self

    def live_stream_window(self):
        if frame is not None:
            cv.imshow('Live Stream', frame)

    def draw_cell_phone_boxes(self, frame, results):
        if frame is None or not results:
            return frame

        phone_count = 0
        for detection in results[0].boxes:
            if int(detection.cls[0]) != self.CELL_PHONE_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            confidence = float(detection.conf[0])
            label = f"cell phone {confidence:.2f}"
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame,label,(x1, max(y1 - 10, 20)),cv.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)
            phone_count += 1

        if phone_count:
            cv.putText(frame,f"Cell phones: {phone_count}",(10, 30),cv.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)
        
        return frame

    def is_cell_phone(self, results):
        if not results:
            return None

        for detection in results[0].boxes:
            if int(detection.cls[0]) == self.CELL_PHONE_CLASS_ID:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                x = x - 320
                y = y - 180
                return x, y

        now = time.time()
        if now - self._last_no_detection_log >= 2:
            print("[detection] No cell phone detected")
            self._last_no_detection_log = now
        return None

    def servo_reset(self):
        print("[servo] Resetting pan and tilt to 0 degrees")
        self.servos.set_angles(0, 0)
        self.pan_angle_previous = 0
        self.tilt_angle_previous = 0

    def control_loop(self):
        pan_step = TRACKING_GAIN * self.pan_angle_current
        tilt_step = TRACKING_GAIN * self.tilt_angle_current
        pan_step = np.clip(pan_step, -MAX_SERVO_STEP, MAX_SERVO_STEP)
        tilt_step = np.clip(tilt_step, -MAX_SERVO_STEP, MAX_SERVO_STEP)
        pan_command = np.clip(
            self.pan_angle_previous + pan_step,
            MIN_SERVO_ANGLE,
            MAX_SERVO_ANGLE,
        )
        tilt_command = np.clip(
            self.tilt_angle_previous + tilt_step,
            MIN_SERVO_ANGLE,
            MAX_SERVO_ANGLE,
        )
        self.pan_angle_previous = pan_command
        self.tilt_angle_previous = tilt_command
        self.servos.set_angles(pan_command, tilt_command)
        self.servo_times.append(time.perf_counter() - self._start_time)
        self.pan_commands.append(pan_command)
        self.tilt_commands.append(tilt_command)

    def record_phone_angles(self, pan_angle, tilt_angle, detected):
        self.phone_times.append(time.perf_counter() - self._start_time)
        if not detected:
            self.phone_pan_angles.append(np.nan)
            self.phone_tilt_angles.append(np.nan)
        else:
            self.phone_pan_angles.append(pan_angle)
            self.phone_tilt_angles.append(tilt_angle)

    def plot_run_data(self):
        if not self.inference_times and not self.servo_times and not self.phone_times:
            print("[shutdown] No inference, phone, or servo data to plot")
            return

        figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        if self.inference_times:
            axes[0].plot(self.inference_times,np.array(self.inference_durations) * 1000,color="tab:blue")
        axes[0].set_title("Camera inference time")
        axes[0].set_ylabel("Milliseconds")
        axes[0].grid(True)

        if self.phone_times:
            axes[1].plot(self.phone_times,self.phone_pan_angles,label="Pan error",color="tab:red")
            axes[1].plot(self.phone_times,self.phone_tilt_angles,label="Tilt error",color="tab:purple")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Detected cell phone angular error")
        axes[1].set_ylabel("Angle from image center (degrees)")
        axes[1].legend()
        axes[1].grid(True)

        if self.servo_times:
            axes[2].plot(self.servo_times,self.pan_commands,label="Pan",color="tab:orange")
            axes[2].plot(self.servo_times,self.tilt_commands,label="Tilt",color="tab:green")
        axes[2].set_title("Servo commands")
        axes[2].set_xlabel("Seconds since program start")
        axes[2].set_ylabel("Angle (degrees)")
        axes[2].legend()
        axes[2].grid(True)

        figure.tight_layout()
        output_path = "tracking_run_data.png"
        figure.savefig(output_path, dpi=150)
        print(f"[shutdown] Saved run plots to {output_path}")
        plt.show()
        plt.close(figure)


if __name__ == "__main__":
    print("Starting camera tracking application")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pi-host",
        default=os.getenv("PI_HOST", DEFAULT_PI_HOST),
        help="Raspberry Pi IP address or hostname",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("YOLO_DEVICE", "cuda:0"),
        help="Inference device, for example cpu or vulkan:0 or cuda:0",
    )
    args = parser.parse_args()
    pi_host = args.pi_host.strip()
    camera = CameraStream(0, pi_host=pi_host, device=args.device).start()
    time.sleep(1)
    print("Camera stream started")
    try:
        camera.servo_reset()
        while camera.running:
            start_time = time.perf_counter()
            frame = camera.current_frame()
            results = camera.predict(frame)
            detection = camera.is_cell_phone(results)
            frame = camera.draw_cell_phone_boxes(frame, results)

            if detection is not None:
                x, y = detection
                camera.pan_angle_current = PAN_DIRECTION * np.degrees(
                    np.arctan(
                        (2 * x * np.tan(np.radians(camera.HFOV) / 2)) / 960
                    )
                )
                camera.tilt_angle_current = TILT_DIRECTION * np.degrees(
                    np.arctan(
                        (2 * y * np.tan(np.radians(camera.VFOV) / 2)) / 540
                    )
                )
            else:
                camera.pan_angle_current = 0
                camera.tilt_angle_current = 0

            camera.record_phone_angles(camera.pan_angle_current, camera.tilt_angle_current, detection is not None)
            camera.control_loop()
            camera.live_stream_window()
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print("[loop] Loop time:", round(elapsed_time * 1000, 2), "ms")
    finally:
        camera.stop()
