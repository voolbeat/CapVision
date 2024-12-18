import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QLabel, QToolBar, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt
import cv2
from ultralytics import YOLO


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bottle cap detector - Model Comparison")
        self.setGeometry(100, 100, 1200, 600)  # Увеличено для отображения двух изображений

        # Метки для отображения изображений
        self.image_label_model_1 = QLabel(self)
        self.image_label_model_1.setAlignment(Qt.AlignCenter)

        self.image_label_model_2 = QLabel(self)
        self.image_label_model_2.setAlignment(Qt.AlignCenter)

        # Размещаем метки горизонтально
        self.image_label_model_1.setGeometry(0, 0, 600, 600)
        self.image_label_model_2.setGeometry(600, 0, 600, 600)

        # Загрузка моделей
        self.yolo_model_1 = YOLO('../runs/detect/train16/weights/best.pt')  # Первая модель
        self.yolo_model_1.fuse()
        self.yolo_model_2 = YOLO('../runs/detect/train/weights/best.pt')  # Вторая модель
        self.yolo_model_2.fuse()

        self.cap = None
        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)

        open_file_action = QAction(QIcon(), "Открыть видео", self)
        open_file_action.triggered.connect(self.open_file)
        toolbar.addAction(open_file_action)

    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Открыть видео", "",
                                                   "Videos (*.mp4 *.avi *.mkv *.mov)",
                                                   options=options)
        if file_name:
            self.process_video(file_name)

    def process_video(self, video_path):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Обработка кадра обеими моделями
            results_model_1 = self.yolo_model_1(frame)
            results_model_2 = self.yolo_model_2(frame)

            # Подсчёт количества детекций
            count_model_1 = len(results_model_1[0].boxes)
            count_model_2 = len(results_model_2[0].boxes)

            # Аннотированные кадры
            annotated_frame_1 = results_model_1[0].plot()
            annotated_frame_2 = results_model_2[0].plot()

            # Отображение информации
            self.display_image(self.image_label_model_1, annotated_frame_1)
            self.display_image(self.image_label_model_2, annotated_frame_2)
            self.setWindowTitle(f"Модель 1: {count_model_1} объектов | Модель 2: {count_model_2} объектов")

            cv2.waitKey(1)

        self.cap.release()
        self.cap = None

    def display_image(self, label, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img = img.rgbSwapped()
        pixmap = QPixmap.fromImage(img)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def closeEvent(self, event):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
