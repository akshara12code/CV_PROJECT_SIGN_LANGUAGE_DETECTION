import traceback
try:
    import cv2
    print('cv2 OK')
    import numpy as np
    print('numpy OK')
    from keras.models import load_model
    print('keras OK')
    from cvzone.HandTrackingModule import HandDetector
    print('cvzone OK')
    import enchant
    print('enchant OK')
    import tkinter as tk
    print('tkinter OK')
    from PIL import Image, ImageTk
    print('PIL OK')
    import pyttsx3
    print('pyttsx3 OK')
    model = load_model('cnn8grps_rad1_model.h5')
    print('model loaded OK')
    cap = cv2.VideoCapture(0)
    print('webcam OK:', cap.isOpened())
    cap.release()
except Exception as e:
    traceback.print_exc()