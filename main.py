import mediapipe as mp
import cv2
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    model_complexity = 1,
    max_num_hands = 1,
    min_detection_confidence = 0.5,
    min_tracking_conffidence = 0.5) as hands:

    while True:
        ret, frame = video_capture.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hands_landmarks:
            for hand_landmarks in result.multi_hands_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_styles(),
                    mp_drawing_styles.get_default_hand_connections_styles()
                )
        cv2.imshow("Video capture", frame)
        if cv2.waitKey(1) == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
