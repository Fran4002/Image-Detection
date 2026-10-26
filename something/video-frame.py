import os
from pathlib import Path
import cv2

# Supported video extensions (including .webm)
VIDEO_EXTENSIONS = {".webm", ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}


def extract_frames_from_video(video_path, output_folder, frame_interval=10):
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return 0

    video_name = video_path.stem
    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = output_folder / f"{video_name}_frame_{count // frame_interval:03d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
        count += 1

    cap.release()
    print(f"Saved {saved_count} frames from {video_path.name}")
    return saved_count


def split_video_into_frames(video_path, output_folder, frame_interval=10):
    """
    Extract frames from a video file or all video files in a directory at specified intervals.
    """
    path = Path(video_path)
    if not path.exists():
        script_dir = Path(__file__).resolve().parent
        if (script_dir / path).exists():
            path = script_dir / path

    output_path = Path(output_folder)
    if not output_path.is_absolute() and not output_path.exists():
        script_dir = Path(__file__).resolve().parent
        if (script_dir / output_path.parent).exists() and not (Path.cwd() / output_path).parent.exists():
            output_path = script_dir / output_path

    if path.is_dir():
        video_files = [
            f for f in sorted(path.iterdir())
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not video_files:
            print(f"No supported video files found in {video_path}")
            return 0
        total_frames = 0
        for vf in video_files:
            total_frames += extract_frames_from_video(vf, output_path, frame_interval)
        print(f"Extracted a total of {total_frames} frames to {output_path}")
        return total_frames
    else:
        return extract_frames_from_video(path, output_path, frame_interval)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    videos_dir = script_dir / "videos"
    frames_dir = script_dir / "frames"

    split_video_into_frames(videos_dir, frames_dir, frame_interval=10)
