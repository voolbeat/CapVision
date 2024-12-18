import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QLabel, QToolBar, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bottle Cap Analysis")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)

        self.yolo_model = YOLO('../runs/detect/train16/weights/best.pt')
        self.yolo_model.fuse()

        self.all_ratios = []

        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        open_files_action = QAction("Открыть файлы", self)
        open_files_action.triggered.connect(self.open_files)
        toolbar.addAction(open_files_action)

        plot_action = QAction("Построить график", self)
        plot_action.triggered.connect(self.plot_ratios)
        toolbar.addAction(plot_action)

    def open_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите видеофайлы", "",
                                                "Videos (*.mp4 *.avi *.mkv *.mov)", options=options)
        if files:
            for file in files:
                self.process_video(file)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        video_ratios = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.yolo_model(frame)
            frame_ratios = self.analyze_frame(frame, results)

            if frame_ratios:
                video_ratios.extend(frame_ratios)

        cap.release()

        if video_ratios:  # Проверяем, что данные были собраны
            self.all_ratios.extend(video_ratios)
            print(f"Обработано видео: {video_path}")
            print(f"Среднее соотношение: {np.mean(video_ratios):.2f}")
            print(f"Медиана: {np.median(video_ratios):.2f}")
            print(f"Мода: {self.calculate_mode(video_ratios):.2f}")
        else:
            print(f"Видео {video_path} не содержит данных для анализа.")

    def analyze_frame(self, frame, results):
        bottle_boxes = []
        cap_boxes = []
        frame_ratios = []

        # Разбор результатов детекции
        for result in results[0].boxes:
            label = int(result.cls)
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            if label == 0:  # Бутылка
                bottle_boxes.append((x1, y1, x2, y2))
            elif label == 1:  # Крышка
                cap_boxes.append((x1, y1, x2, y2))

        # Сопоставление крышек и бутылок
        for bottle_box in bottle_boxes:
            bx1, by1, bx2, by2 = bottle_box
            bottle_height = by2 - by1

            for cap_box in cap_boxes:
                cx1, cy1, cx2, cy2 = cap_box
                cap_height = cy2 - cy1

                # Проверка пересечения крышки и бутылки по горизонтали
                if cx1 < bx2 and cx2 > bx1:
                    if bottle_height > 0:
                        ratio = cap_height / bottle_height
                        frame_ratios.append(ratio)

                        # Определяем цвет в зависимости от соотношения высот
                        color = (0, 255, 0) if ratio < 0.140 else (0, 0, 255)

                        # Рисуем контуры бутылки и крышки
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 2)

                        # Выводим соотношение над бутылкой
                        text = f"Ratio: {ratio:.3f}"
                        cv2.putText(frame, text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Отображаем кадр с аннотациями
        self.display_image(frame)

        return frame_ratios

    def plot_ratios(self):
        if not self.all_ratios:
            QMessageBox.warning(self, "Ошибка", "Нет данных для построения графика.")
            return

        plt.figure(figsize=(12, 6))
        plt.hist(self.all_ratios, bins=50, color='blue', alpha=0.7)
        plt.axvline(np.mean(self.all_ratios), color='red', linestyle='--', label='Среднее значение')
        plt.axvline(np.median(self.all_ratios), color='green', linestyle='--', label='Медиана')
        plt.xlabel("Соотношение высоты крышки к высоте бутылки")
        plt.ylabel("Частота")
        plt.title("Общий график соотношений высоты крышки к высоте бутылки для всех видео")
        plt.legend()
        plt.show()

    def calculate_mode(self, data):
        values, counts = np.unique(data, return_counts=True)
        index = np.argmax(counts)
        return values[index]

    def display_image(self, img):
        qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img = img.rgbSwapped()
        pixmap = QPixmap.fromImage(img)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def closeEvent(self, event):
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
