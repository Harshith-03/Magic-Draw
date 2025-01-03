# Magic-Draw

This project demonstrates a hand gesture-controlled drawing application using **PyQt5** for the user interface and **MediaPipe** for hand tracking. The application allows users to draw on a canvas using hand gestures and detect text from the drawn content using Tesseract OCR.

---

## Features

- **Hand Gesture Detection:** Detect hand gestures (writing, stop, erase).
- **Drawing Interface:** Draw on the canvas using index finger movements.
- **Erase Gesture:** Clear parts of the canvas by waving with all fingers.
- **Text Detection:** Extract and display detected text from the canvas using OCR.
- **Save Functionality:** Save the detected text to a file.
- **Cross-Platform:** Built using PyQt5, works on Windows, Linux, and macOS.

---

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- PyQt5
- OpenCV
- MediaPipe
- pytesseract

Install these dependencies using:
```bash
pip install opencv-python mediapipe pytesseract PyQt5
```

---

##How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Harshith-03/Magic-Draw.git
   cd Magic-Draw
   ```

2. Run the application:\
   ```bash
   python main.py

3. Use hand gestures to draw on the canvas and interact with the UI

