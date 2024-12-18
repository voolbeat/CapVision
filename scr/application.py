import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QLabel, QToolBar, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt
import cv2
from ultralytics import YOLO


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bottle cap detector")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)

        # Using only one YOLO model
        self.yolo_model = YOLO('../runs/detect/train16/weights/best.pt')
        self.yolo_model.fuse()

        self.cap = None

        # Create a toolbar with actions
        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar("")
        self.addToolBar(toolbar)

        open_file_action = QAction(QIcon(), "Открыть файл", self)
        open_file_action.triggered.connect(self.open_file)
        toolbar.addAction(open_file_action)

        open_camera_action = QAction(QIcon(), "Запустить камеру", self)
        open_camera_action.triggered.connect(self.open_camera)
        toolbar.addAction(open_camera_action)

    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Открыть файл", "",
                                                   "Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi *.mkv *.mov)",
                                                   options=options)
        if file_name:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.detect_in_image(file_name)
            elif file_name.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.process_video(file_name)
            else:
                QMessageBox.warning(self, "Ошибка", "Неверный тип файла.")

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.process_video_stream()

    def process_video(self, video_path):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.process_stream()

    def process_video_stream(self):
        self.process_stream()

    def process_stream(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.yolo_model(frame)
            annotated_frame = results[0].plot()

            self.display_image(annotated_frame)
            cv2.waitKey(1)
        self.cap.release()
        self.cap = None

    def detect_in_image(self, image_path):
        image = cv2.imread(image_path)
        results = self.yolo_model(image)
        annotated_image = results[0].plot()

        self.display_image(annotated_image)

    def display_image(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img = img.rgbSwapped()
        pixmap = QPixmap.fromImage(img)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def closeEvent(self, event):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
