import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # Extract actual image frame if stored inside a dict
    if isinstance(output_video_frames[0], dict):
        frames = [frame_dict['frame'] for frame_dict in output_video_frames]
    else:
        frames = output_video_frames

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)
    out.release()

