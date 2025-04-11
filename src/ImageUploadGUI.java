import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class ImageUploadGUI extends JFrame {
    static { System.load("D:\\Download\\opencv\\build\\java\\x64\\opencv_java490.dll");
}

    JLabel label = new JLabel();
    CascadeClassifier faceDetector = new CascadeClassifier("src/haarcascade_frontalface_alt.xml");

    public ImageUploadGUI() {
        super("Upload Image Face Detection");
        JButton uploadBtn = new JButton("Upload Image");
        uploadBtn.addActionListener(e -> selectImage());
        setLayout(new BorderLayout());
        add(uploadBtn, BorderLayout.NORTH);
        add(label, BorderLayout.CENTER);
        setSize(600, 600);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setVisible(true);
    }

    void selectImage() {
        JFileChooser fileChooser = new JFileChooser();
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            detectFace(file.getAbsolutePath());
        }
    }

    void detectFace(String path) {
        Mat src = Imgcodecs.imread(path);
        if (src.empty()) {
            JOptionPane.showMessageDialog(this, "Cannot load image.");
            return;
        }

        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(gray, faces);

        for (Rect rect : faces.toArray()) {
            Imgproc.rectangle(src, rect, new Scalar(0, 255, 0), 2);
        }

        String outPath = "output.jpg";
        Imgcodecs.imwrite(outPath, src);
        label.setIcon(new ImageIcon(outPath));
    }

    public static void main(String[] args) {
        new ImageUploadGUI();
    }
}
