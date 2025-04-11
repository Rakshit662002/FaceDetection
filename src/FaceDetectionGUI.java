import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

public class FaceDetectionGUI extends JFrame {
    static { System.load("D:\\Download\\opencv\\build\\java\\x64\\opencv_java490.dll"); }

    JLabel imageLabel;
    JButton webcamButton, uploadButton, referenceButton, compareButton, exitButton;
    CascadeClassifier faceDetector = new CascadeClassifier("src/haarcascade_frontalface_alt.xml");
    VideoCapture camera;
    Mat referenceFace = null;

    public FaceDetectionGUI() {
        super("Face Detection GUI");

        imageLabel = new JLabel();
        imageLabel.setHorizontalAlignment(JLabel.CENTER);

        webcamButton = new JButton("Start Webcam");
        uploadButton = new JButton("Upload Image");
        referenceButton = new JButton("Upload Reference Face");
        compareButton = new JButton("Compare Face");
        exitButton = new JButton("Exit");

        JPanel buttonPanel = new JPanel();
        buttonPanel.add(webcamButton);
        buttonPanel.add(uploadButton);
        buttonPanel.add(referenceButton);
        buttonPanel.add(compareButton);
        buttonPanel.add(exitButton);

        setLayout(new BorderLayout());
        add(buttonPanel, BorderLayout.NORTH);
        add(imageLabel, BorderLayout.CENTER);

        setSize(900, 600);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setVisible(true);

        webcamButton.addActionListener(e -> startWebcam());
        uploadButton.addActionListener(e -> uploadAndDetect(false));
        referenceButton.addActionListener(e -> uploadAndDetect(true));
        compareButton.addActionListener(e -> compareWithReference());
        exitButton.addActionListener(e -> System.exit(0));
    }

    // Start webcam and display face detection
    public void startWebcam() {
        if (camera != null && camera.isOpened()) camera.release();

        camera = new VideoCapture(0);
        new Thread(() -> {
            Mat frame = new Mat();
            while (camera.read(frame)) {
                Mat gray = new Mat();
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
                MatOfRect faces = new MatOfRect();
                faceDetector.detectMultiScale(gray, faces);

                for (Rect rect : faces.toArray()) {
                    Imgproc.rectangle(frame, rect, new Scalar(0, 255, 0), 2);
                }

                ImageIcon icon = new ImageIcon(matToBufferedImage(frame));
                imageLabel.setIcon(icon);
                imageLabel.repaint();
            }
        }).start();
    }

    // Upload image: either register reference face OR just detect
    public void uploadAndDetect(boolean isReference) {
        JFileChooser fileChooser = new JFileChooser();
        if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
            File imgFile = fileChooser.getSelectedFile();
            Mat img = Imgcodecs.imread(imgFile.getAbsolutePath());
            if (img.empty()) {
                JOptionPane.showMessageDialog(this, "Cannot read image!");
                return;
            }

            Mat face = extractFace(img);
            if (face != null) {
                if (isReference) {
                    referenceFace = face;
                    JOptionPane.showMessageDialog(this, "Reference face registered!");
                }

                Imgproc.rectangle(img, new Rect(face.cols(), face.rows(), face.cols(), face.rows()), new Scalar(0, 255, 0), 2);
            } else {
                JOptionPane.showMessageDialog(this, "No face detected!");
            }

            ImageIcon icon = new ImageIcon(matToBufferedImage(img));
            imageLabel.setIcon(icon);
            imageLabel.repaint();
        }
    }

    // Compare uploaded face with reference face
    public void compareWithReference() {
        if (referenceFace == null) {
            JOptionPane.showMessageDialog(this, "Upload a reference face first!");
            return;
        }

        JFileChooser fileChooser = new JFileChooser();
        if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
            Mat newImg = Imgcodecs.imread(fileChooser.getSelectedFile().getAbsolutePath());
            Mat testFace = extractFace(newImg);

            if (testFace == null) {
                JOptionPane.showMessageDialog(this, "No face found in test image!");
                return;
            }

            boolean match = compareFaces(referenceFace, testFace);
            JOptionPane.showMessageDialog(this, match ? "Faces Match!" : "Faces Do Not Match");

        }
    }

    // Extract face from image
    public Mat extractFace(Mat img) {
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(gray, faces);
        if (faces.toArray().length > 0) {
            Rect rect = faces.toArray()[0];
            return new Mat(img, rect);
        }
        return null;
    }

    // Compare two faces based on pixel diff
    public boolean compareFaces(Mat face1, Mat face2) {
        Mat resized1 = new Mat();
        Mat resized2 = new Mat();
        Imgproc.resize(face1, resized1, new Size(100, 100));
        Imgproc.resize(face2, resized2, new Size(100, 100));

        Mat diff = new Mat();
        Core.absdiff(resized1, resized2, diff);
        Scalar sum = Core.sumElems(diff);
        double totalDiff = sum.val[0] + sum.val[1] + sum.val[2];
        return totalDiff < 10000; // Adjust threshold as needed
    }

    // Convert OpenCV Mat to Java BufferedImage
    public BufferedImage matToBufferedImage(Mat mat) {
        int type = (mat.channels() == 1) ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;
        byte[] b = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, b);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), b);
        return image;
    }

    public static void main(String[] args) {
        new FaceDetectionGUI();
    }
}
