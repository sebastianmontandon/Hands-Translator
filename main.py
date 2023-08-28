import mediapipe as mp
import cv2
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Hand points
thump_points = [1,2,4]
index_points = [6,7,8]
middle_points = [10,11,12]
ring_points = [14,15,16]
little_points = [18,19,20]
palm_points = [0,1,2,5,9,13,17]
fingertips_points = [8,12,16,20]
finger_base_points = [6,10,14,18]

def hand_centroid(list):
    print(list)
    cord = np.array(list)
    centroid = np.mean(cord, axis=0)
    print(centroid)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

def finger_distance(base, tip, centx, centy):
    center = np.array([centx, centy])
    l1 = np.linalg.norm(base - center, axis=1)
    l2 = np.linalg.norm(tip - center, axis=1)
    if l1 > l2:
        return True
    elif l2 > l1:
        return False

def coordinates_calculator(thump, landmark, points, w, h):
    coordinates = []
    for index in points:
        x = int(landmark.landmark[index].x * w)
        y = int(landmark.landmark[index].y * h)
        coordinates.append([x, y])
    # Finger points
    p1 = np.array(coordinates[0])
    p2 = np.array(coordinates[1])
    p3 = np.array(coordinates[2])
    # Finger points distance
    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)
    # Finger angle calculation
    if thump:
        angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 *l3)))
        return angle
    else:
        x,y = hand_centroid(palm_points)
        return finger_distance(p2, p1, x, y)


with mp_hands.Hands(
    model_complexity = 1,
    max_num_hands = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:

    while True:
        ret, frame = video_capture.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width,_ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                thump_coordinates = coordinates_calculator(True, hand_landmarks, thump_points, width, height)
                index_coordinates = coordinates_calculator(False, hand_landmarks, index_points, width, height)
                middle_coordinates = coordinates_calculator(False, hand_landmarks, middle_points, width, height)
                ring_coordinates = coordinates_calculator(False, hand_landmarks, ring_points, width, height)
                little_coordinates = coordinates_calculator(False, hand_landmarks, little_points, width, height)
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                if thump_coordinates in range(142, 154) and index_coordinates and middle_coordinates and ring_coordinates and little_coordinates:
                    print('A')

                print(f'Thump: {thump_coordinates}')
                print(f'Index: {index_coordinates}')
                print(f'Middle: {middle_coordinates}')
                print(f'Ring: {ring_coordinates}')
                print(f'Little: {little_coordinates}')
        
        cv2.imshow("Video capture", frame)
        if cv2.waitKey(1) == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
