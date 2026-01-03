from ultralytics import YOLO

model = YOLO("yolo12x.pt")

#result = model.predict(source="input_video/input_video.mp4", device=0, save=True)
result = model.track(source="input_video/input_video.mp4", device=0, save=True, persist=True)