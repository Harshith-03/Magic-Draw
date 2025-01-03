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

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Harshith-03/Magic-Draw.git
   cd Magic-Draw
   ```

2. Run the application:
   ```bash
   python main.py

3. Use hand gestures to draw on the canvas and interact with the UI

---

## Functionality

- **Drawing Mode:** Activate by raising the index finger.
- **Erase Mode:** Activate by raising all fingers and waving.
- **Clear Canvas:** Click the "Clear Canvas" button on the UI.
- **Save Text:** Use the "Save Detected Text" button to store recognized text.

---

## Contributing

1. Fork this repository.
2. Create a new branch: git checkout -b feature-branch.
3. Make your changes and commit them: git commit -m "Add new feature".
4. Push to the branch: git push origin feature-branch.
5. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

---

## Contact

- Email: harshithstanes@gmail.com
- LinkedIn: www.linkedin.com/in/hvr2503



