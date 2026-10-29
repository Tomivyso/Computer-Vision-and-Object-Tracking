from ultralytics import YOLO
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=("pt", "ncnn", "engine"),
    default="pt",
    help="Backend to benchmark: pt and engine use CUDA, ncnn uses Vulkan",
)
args = parser.parse_args()

if args.model == "pt":
    model = YOLO("yolo26x.pt")
    device = "cuda:0"
elif args.model == "engine":
    model = YOLO("yolov8n.engine")
    device = "cuda:0"
else:
    model = YOLO("yolov8n_ncnn_model")
    device = "vulkan:0"



times = []
frame = "test.jpg"
model.predict(
    source=frame,
    imgsz=640,
    conf=0.4,
    device=device,
    verbose=False,
)  # warmup

start_time = time.time()
end_time = time.time()

for i in range(2000):
    results = model.predict(
                source=frame,
                imgsz=640,
                conf=0.4,
                device=device,
                verbose=False,
            )
    times.append(time.time() - end_time)
    end_time = time.time()
end_time = time.time()

print(sorted(times, reverse=True)[:3])
print(min(times))
elapsed_time = end_time - start_time
average_time_per_inference = elapsed_time / 2000 * 1000
print("Average time per inference:", round(average_time_per_inference, 2), "ms")


results = model.predict(frame, device=0, verbose=False)
print(results[0].speed)