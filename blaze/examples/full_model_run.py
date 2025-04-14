import cv2
import numpy as np
import time

import torch

import blaze
from blaze.model import BlazeModel
from blaze.model.detector import BlazeDetector

torch_model_path = blaze.MODEL_PATH

blaze_model = BlazeModel()
blaze_model.load_anchors(blaze.ANCHORS_PATH)
blaze_model.load_state_dict(torch.load(torch_model_path, map_location=torch.device('cpu')))
blaze_model.eval()

detector = BlazeDetector(blaze_model.dbox_list)

cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = 0
frame_count = 0
start_time = time.time()
prev_frame_time = 0
new_frame_time = 0

classes_names = blaze.CLASSES

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    image = np.expand_dims(image, axis=0)
    image_tensor = torch.tensor(np.transpose(image, (0, 3, 1, 2)))

    raw_output = blaze_model(image_tensor)

    predictions = detector(raw_output)

    if (predictions[0, :, 0] > 0.5).any():
        filtered_dets = predictions[0][predictions[0, :, 0] > 0.5]
        x1 = int(filtered_dets[0, 1].item() * width)
        y1 = int(filtered_dets[0, 2].item() * height)
        x2 = int(filtered_dets[0, 3].item() * width)
        y2 = int(filtered_dets[0, 4].item() * height)

        print(classes_names[int(filtered_dets[0, 5].item())])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow('Frame', frame)

    frame_count += 1

    if time.time() - start_time >= 1:
        fps = frame_count
        frame_count = 0
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
