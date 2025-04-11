import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class WebcamGUI extends JFrame {
    static { System.load("D:\\Download\\opencv\\build\\java\\x64\\opencv_java490.dll"); }

    JLabel videoLabel;
    CascadeClassifier faceDetector = new CascadeClassifier("src/haarcascade_frontalface_alt.xml");

    public WebcamGUI() {
        super("Live Face Detection");
        videoLabel = new JLabel();
        add(videoLabel);
        setSize(640, 480);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setVisible(true);
        new Thread(this::startCamera).start();
    }

    public void startCamera() {
        VideoCapture camera = new VideoCapture(0);
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
            videoLabel.setIcon(icon);
            videoLabel.repaint();
        }

        camera.release();
    }

    public BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_3BYTE_BGR;
        byte[] b = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, b);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), b);
        return image;
    }

    public static void main(String[] args) {
        new WebcamGUI();
    }
}
