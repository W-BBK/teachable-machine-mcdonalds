from tf_keras.models import load_model
import cv2  # Install openacv-python
import numpy as np
import time
import os

# make sure to turn off continuity camera on your connected iphone
# Settings -> General -> AirPlay & Continuity -> Turn off Continuity Camera

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# Changes URL to where you get sent to for being on your phone
url = "https://jobs.mchire.com"

last_fired = time.time()

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    if not ret:
        continue

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    label = class_name[2:].strip()

    if label == "using phone" and confidence_score > 0.90 and time.time()-last_fired>10:
        last_fired = time.time()
        os.system(f"open -a Safari {url}")
        #time.sleep(10)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
