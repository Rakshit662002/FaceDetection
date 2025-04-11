# 👁️ Face Detection GUI using Java & OpenCV

This is a desktop application built in **Java (JavaFX/Swing)** using **OpenCV** that allows users to:

- ✅ Detect faces using **Haar Cascades**
- 🎥 Capture live video from the webcam and detect faces in real time
- 🖼️ Upload an image and perform face detection
- 📷 Compare two faces and display whether they **match or not**
- 🖌️ Features a user-friendly **GUI interface** with JavaFX/Swing
- 🎨 Can be styled with **CSS or HTML** (JavaFX version)

---

## 🧠 Technologies Used

- **Java 8+**
- **OpenCV 4.x** (Java Bindings)
- **Swing / JavaFX**
- **Haar Cascade XML classifier**
- **Maven/Gradle or plain JAR-based structure**

---

## 📸 Features

| Feature | Description |
|--------|-------------|
| 👤 Face Detection | Real-time face detection using OpenCV's Haar Cascade |
| 📷 Webcam Support | Start/Stop webcam feed and detect faces live |
| 🖼️ Image Upload | Upload an image and detect all faces in it |
| 🧠 Face Match | Compare uploaded face with webcam face (basic pixel difference method) |
| 🖌️ Styled UI | Use CSS for custom themes and layout (JavaFX) |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/FaceDetectionJava.git
cd FaceDetectionJava



--------------------------------------------------------------------------------
-
---------------------------------------------------------------------------------
javac -cp ".;lib/opencv-490.jar" -d bin src/FaceDetectionGUI.java
java -cp ".;lib/opencv-490.jar;bin" FaceDetectionGUI



----------------------------------------------------------------------------------------
📸 Demo Screenshots
Add your project screenshots here to showcase GUI, webcam, and image upload features.

🤝 Contributions
Feel free to open issues, suggestions, or pull requests.

📄 License
This project is licensed under the MIT License.

