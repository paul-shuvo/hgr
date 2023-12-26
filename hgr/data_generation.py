import numpy as np
import cv2 as cv
from config import model_path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE)
data = np.zeros((1, 21, 3))
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(2)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_landmarker_result = landmarker.detect(mp_image)
        for hlr in hand_landmarker_result.hand_landmarks:
            sample = []
            for i, _landmark in enumerate(hlr):
                # print(_landmark)
                x = _landmark.x
                y = _landmark.y
                z = _landmark.z
                sample.append([x, y, z])
                shape = frame.shape 
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                cv.circle(frame, (relative_x, relative_y), radius=5, color=(225, 0, 100), thickness=1)
            
        # cv2_imshow(image)
            data = np.append(data, np.expand_dims(np.asarray(sample), axis=0), axis=0)
            print(data.shape)
        if data.shape[0] > 300:
            with open('hold.npy', 'wb') as f:
                np.save(f, data)
                # np.save(f, np.array([1, 3]))
            break
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()