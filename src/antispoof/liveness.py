import cv2
import numpy as np
from antispoof.detector import is_real_face

MOTION_THRESHOLD = 2.5   # tune later
FRAME_WINDOW = 10

def compute_motion(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    motion_score = np.mean(diff)

    return motion_score
def is_live_face(cap):
    frames = []
    motion_scores = []

    for _ in range(FRAME_WINDOW):
        ret, frame = cap.read()
        if not ret:
            return False, 0.0

        frames.append(frame)

    # Check model prediction on last frame
    real, conf = is_real_face(frames[-1])

    # Compute motion
    for i in range(1, len(frames)):
        motion = compute_motion(frames[i-1], frames[i])
        motion_scores.append(motion)

    avg_motion = np.mean(motion_scores)

    # Final decision
    if real and avg_motion > MOTION_THRESHOLD:
        return True, conf
    else:
        return False, conf
    
