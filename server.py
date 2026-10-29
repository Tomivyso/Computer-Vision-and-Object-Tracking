import json
import socketserver
from threading import Lock

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory


PAN_PIN = 12
TILT_PIN = 13
PORT = 8765
MIN_PULSE_WIDTH = 0.0005
MAX_PULSE_WIDTH = 0.0025


class ServoController:
    def __init__(self):
        factory = PiGPIOFactory()
        servo_options = {
            "min_angle": -89,
            "max_angle": 89,
            "min_pulse_width": MIN_PULSE_WIDTH,
            "max_pulse_width": MAX_PULSE_WIDTH,
            "pin_factory": factory,
        }
        self.pan = AngularServo(PAN_PIN, **servo_options)
        self.tilt = AngularServo(TILT_PIN, **servo_options)
        self.lock = Lock()
        self.set_angles(0, 0)

    def set_angles(self, pan, tilt):
        pan = max(-89, min(89, float(pan)))
        tilt = max(-89, min(89, float(tilt)))
        with self.lock:
            self.pan.angle = pan
            self.tilt.angle = tilt
        return pan, tilt

    def close(self):
        with self.lock:
            self.pan.detach()
            self.tilt.detach()


servos = ServoController()


class ServoRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw_request in self.rfile:
            try:
                request = json.loads(raw_request.decode("utf-8"))
                if request.get("command") != "set_angles":
                    raise ValueError("unknown command")

                pan, tilt = servos.set_angles(request["pan"], request["tilt"])
                print(
                    f"[servo] Applied command: pan={pan:.2f}, tilt={tilt:.2f}",
                    flush=True,
                )
                response = {"ok": True, "pan": pan, "tilt": tilt}
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                response = {"ok": False, "error": str(error)}

            self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
            self.wfile.flush()


class ThreadedServoServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    try:
        with ThreadedServoServer(("0.0.0.0", PORT), ServoRequestHandler) as server:
            print(f"Servo server listening on TCP port {PORT}")
            print(f"Pan GPIO: {PAN_PIN}, tilt GPIO: {TILT_PIN}")
            server.serve_forever()
    finally:
        servos.close()
