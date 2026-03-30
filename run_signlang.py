import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model
import traceback

print("Loading model...")
model = load_model('cnn8grps_rad1_model.h5')
print("Model loaded!")

# Create white image
white_base = np.ones((400, 400, 3), np.uint8) * 255
cv2.imwrite("white.jpg", white_base)
print("White image created!")

capture = cv2.VideoCapture(0)
print("Camera opened:", capture.isOpened())

print("Creating hand detectors...")
hd = HandDetector(maxHands=1)
print("hd created")
hd2 = HandDetector(maxHands=1)
print("hd2 created")

offset = 29
print("Starting loop... Press ESC to quit")

def distance(x, y):
    return math.sqrt(((x[0]-y[0])**2) + ((x[1]-y[1])**2))

print("Starting loop... Press ESC to quit")

while True:
    try:
        print("Reading frame...")
        ret, frame = capture.read()
        print("Frame read:", ret, frame.shape if ret else "None")
        if not ret:
            print("Camera read failed")
            break

        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)

        if hands and len(hands[0]) > 0:
            hand = hands[0][0]
            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = min(frame.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(frame.shape[1], x + w + offset)
            image = frame[y1:y2, x1:x2]

            if image.size > 0:
                white = np.ones((400, 400, 3), np.uint8) * 255
                handz = hd2.findHands(image, draw=False, flipType=True)

                if handz and len(handz[0]) > 0:
                    hand2 = handz[0][0]
                    pts = hand2['lmList']

                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15

                    for t in range(0, 4):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
                    for t in range(5, 8):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
                    for t in range(9, 12):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
                    for t in range(13, 16):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
                    for t in range(17, 20):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)

                    cv2.line(white, (pts[5][0]+os, pts[5][1]+os1), (pts[9][0]+os, pts[9][1]+os1), (0,255,0), 3)
                    cv2.line(white, (pts[9][0]+os, pts[9][1]+os1), (pts[13][0]+os, pts[13][1]+os1), (0,255,0), 3)
                    cv2.line(white, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0,255,0), 3)
                    cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[5][0]+os, pts[5][1]+os1), (0,255,0), 3)
                    cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0,255,0), 3)

                    for i in range(21):
                        cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), 2, (0,0,255), 1)

                    white_input = white.reshape(1, 400, 400, 3)
                    prob = np.array(model.predict(white_input)[0], dtype='float32')
                    ch1 = np.argmax(prob, axis=0)

                    label = str(ch1)
                    frame = cv2.putText(frame, "Predicted: " + label, (30, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                    cv2.imshow("Hand Skeleton", white)

        cv2.imshow("Sign Language", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        print("Error:", traceback.format_exc())
        break

capture.release()
cv2.destroyAllWindows()
print("Done!")