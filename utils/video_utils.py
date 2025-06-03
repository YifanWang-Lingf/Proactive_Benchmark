import cv2
import math, os
import numpy as np
import torch


def get_video_length_sec(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = frame_count / input_fps
    cap.release()
    return video_duration

def load_video(video_file, start_sec=0, end_sec=100000000000, output_resolution=384, output_fps=2):
    cap = cv2.VideoCapture(video_file)
    # Get original video properties
    pad_color = (0, 0, 0)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / input_fps
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = output_height = output_resolution

    start_sec, end_sec = max(0, start_sec), min(video_duration, end_sec)
    start_frame, end_frame = start_sec * input_fps, end_sec * input_fps
    num_frames_total = math.ceil((end_sec - start_sec) * output_fps)
    frame_sec = [(i / output_fps) + start_sec for i in range(num_frames_total)]
    frame_list, cur_time, frame_index = [], start_sec, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
            if input_width > input_height:
                # Landscape video: scale width to the resolution, adjust height
                new_width = output_resolution
                new_height = int((input_height / input_width) * output_resolution)
            else:
                # Portrait video: scale height to the resolution, adjust width
                new_height = output_resolution
                new_width = int((input_width / input_height) * output_resolution)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            # pad the frame
            canvas = cv2.copyMakeBorder(
                resized_frame,
                top=(output_height - new_height) // 2,
                bottom=(output_height - new_height + 1) // 2,
                left=(output_width - new_width) // 2,
                right=(output_width - new_width + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_color
            )
            frame_list.append(np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)))
            frame_index += 1
        cur_time += 1 / input_fps
        if cur_time > end_sec:
            break
    cap.release()
    return frame_list

