import cv2
import numpy as np
import time
import smbus
from ai_edge_litert.interpreter import Interpreter

bus = smbus.SMBus(1)

pico_address = 0x44

interpreter = Interpreter(model_path='blazeface3_nor_from_pc13.tflite')

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

post_interpreter = Interpreter(model_path='cor_detectTF_opt2.tflite')

post_interpreter.allocate_tensors()

post_input_details = post_interpreter.get_input_details()
post_output_details = post_interpreter.get_output_details()


fps = 0
frame_count = 0
start_time = time.time()
prev_frame_time = 0
new_frame_time = 0

video_path = 'test_video.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

classes_names = ["dislike", "like", "no_gesture"]

while True:
    ret, frame = cap.read()
    if not ret:
        end_time = time.time()
        print(end_time - start_time)
        break

    image_process_time_start = time.time()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    image_tensor = np.expand_dims(image, axis=0)
    image_tensor = np.transpose(image_tensor, (0, 3, 1, 2))

    interpreter.set_tensor(input_details[0]['index'], image_tensor)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    post_interpreter.set_tensor(post_input_details[0]['index'], output_data)

    post_interpreter.invoke()

    post_output_data = post_interpreter.get_tensor(post_output_details[0]['index'])

    print(classes_names[post_output_data])

    byte_to_send = post_output_data + 2
    bus.write_byte(pico_address, byte_to_send)
    image_process_time_end = time.time()
    print(image_process_time_end - image_process_time_start)

cap.release()
