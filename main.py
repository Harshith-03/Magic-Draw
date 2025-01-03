#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import cv2
import mediapipe as mp
import numpy as np
import pytesseract
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QTextEdit, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap




pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

class HandGestureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCamera()

        self.canvas = None
        self.drawing_color = (255, 100, 0) #Blue
        self.drawing_thickness = 7
        self.drawing_enabled = False
        self.previous_x = None
        self.previous_y = None
        self.last_cleared_time = 0
        self.clear_delay = 1.0
        self.erase_radius = 50


    def initUI(self):
        #Initialize UI elements
        self.setWindowTitle("Hand Gesture Controlled Drawing")
        self.setGeometry(100, 100, 1280, 800)

        #Central widgets and layout 
        self.central_widget = QWidget(self)
        self. setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        #Video Feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # QTextEdit for displaying detected text
        self.text_display = QTextEdit(self)
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Detected text will appear here...")
        self.layout.addWidget(self.text_display)

        # Button to clear canvas
        self.clear_button = QPushButton("Clear Canvas", self)
        self.clear_button.clicked.connect(self.clear_canvas)
        self.layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save Detected Text", self)
        self.save_button.clicked.connect(self.save_detected_text)
        self.layout.addWidget(self.save_button)

    def initCamera(self):
        #Initialize camera and setup a timer for frame updates
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            print("Error: Webcam could not be accessed.")
            sys.exit()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)


    def update_frame(self):
        """Process frame for hand tracking and drawing."""
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # Create canvas if not exists
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

        # Process the frame using MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

                h, w, _ = frame.shape
                gesture = self.check_gesture(hand_landmarks)

                if gesture == "STOP":
                    self.drawing_enabled = False
                elif gesture == "WRITE" and time.time() - self.last_cleared_time >= self.clear_delay:
                    self.drawing_enabled = True

                # Draw trajectory
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                if self.drawing_enabled:
                    if self.previous_x is not None and self.previous_y is not None:
                        cv2.line(self.canvas, (self.previous_x, self.previous_y), (index_x, index_y), self.drawing_color, self.drawing_thickness)
                    self.previous_x, self.previous_y = index_x, index_y
                else:
                    self.previous_x, self.previous_y = None, None

                # Check erase gesture
                if self.check_erase_gesture(hand_landmarks):
                    # Enable erase mode
                    cv2.putText(frame, "Erasing...", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                    # Get palm base position as the erase center
                    palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    palm_x, palm_y = int(palm_base.x * w), int(palm_base.y * h)
                
                    # Erase the canvas in a circular region around the palm
                    cv2.circle(self.canvas, (palm_x, palm_y), self.erase_radius, (0, 0, 0), -1)
        


        # Detect text on canvas
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        detected_text = pytesseract.image_to_string(gray_canvas).strip()
        self.text_display.setText(detected_text)
        
        # Combine canvas and frame
        combined = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)

        # Convert to QPixmap for display
        h, w, ch = combined.shape
        bytes_per_line = ch * w
        qt_image = QImage(combined.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def check_erase_gesture(self, hand_landmarks):
        """Check if all fingers are extended for erase gesture."""
        landmarks = hand_landmarks.landmark
    
        # Indices for finger tip and PIP joints
        FINGER_TIPS = [
            mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP,
        ]
        FINGER_PIPS = [
            mp_hands.HandLandmark.THUMB_IP,
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP,
        ]
    
        # Check if all fingers are extended
        fingers_extended = [
            landmarks[tip].y < landmarks[pip].y for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
        ]
        return all(fingers_extended)

    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas = np.zeros_like(self.canvas)
        self.last_cleared_time = time.time()

    def save_detected_text(self):
        """Save detected text to a file."""
        detected_text = self.text_display.toPlainText()
        if not detected_text.strip():
            print("No text to save.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Detected Text", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            with open(file_path, "w") as file:
                file.write(detected_text)
            print(f"Detected text saved to: {file_path}")

    def check_gesture(self, hand_landmarks):
        """Check gestures based on finger positions."""
        landmarks = hand_landmarks.landmark
        index_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP
        index_pip = mp_hands.HandLandmark.INDEX_FINGER_PIP
        pinky_tip = mp_hands.HandLandmark.PINKY_TIP
        pinky_pip = mp_hands.HandLandmark.PINKY_PIP

        index_extended = landmarks[index_tip].y < landmarks[index_pip].y
        pinky_extended = landmarks[pinky_tip].y < landmarks[pinky_pip].y

        if index_extended and pinky_extended:
            return "STOP"
        elif index_extended:
            return "WRITE"
        return "NONE"

    def closeEvent(self, event):
        """Release resources on close."""
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandGestureApp()
    window.show()
    sys.exit(app.exec_())

