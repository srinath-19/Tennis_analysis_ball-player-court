from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():

    input_video_path = "input_video/input_video.mp4"
    video_frames = read_video(input_video_path)

    player_tracker= PlayerTracker(model_path="models/yolo12x.pt")
    player_dedections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detection.pkl")


    ball_tracker = BallTracker(model_path="models/tennis_ball_best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detection.pkl")


    court_model_path = "models/keypoints_model_50.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    player_dedections = player_tracker.choose_and_filter_players(court_keypoints, player_dedections)


    output_video_frames = player_tracker.draw_bboxes(video_frames, player_dedections)

    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)


    save_video(output_video_frames, 'output_videos/123.avi')



if __name__ == "__main__":
    main()