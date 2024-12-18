import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QAction, QToolBar
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

THRESHOLD = 0.1395  # Пороговое значение для определения недокрученной крышки

class BottleCapAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bottle Cap Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)

        # Загрузка модели YOLO
        self.yolo_model = YOLO('runs/detect/train/weights/best.pt')
        self.yolo_model.fuse()

        self.cap = None
        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)

        open_file_action = QAction("Открыть видео", self)
        open_file_action.triggered.connect(self.open_file)
        toolbar.addAction(open_file_action)

    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "",
                                                   "Videos (*.mp4 *.avi *.mkv *.mov)",
                                                   options=options)
        if file_name:
            self.process_video(file_name)

    def process_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        # Получаем параметры видео
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Настраиваем VideoWriter для сохранения обработанного видео
        output_path = video_path.rsplit('.', 1)[0] + "_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Используем кодек mp4v
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.yolo_model(frame)
            annotated_frame = frame.copy()

            center_x = frame_width // 2
            cv2.line(annotated_frame, (center_x, 0), (center_x, frame_height), (255, 255, 0), 2)  # Центральная линия

            bottle_boxes = []
            cap_boxes = []

            # Разбор результатов детекции
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                class_id = int(result.cls)

                if class_id == 0:  # Бутылка
                    bottle_boxes.append((x1, y1, x2, y2))
                elif class_id == 1:  # Крышка
                    cap_boxes.append((x1, y1, x2, y2))

            # Обработка каждой бутылки
            for bottle_box in bottle_boxes:
                bottle_x1, bottle_y1, bottle_x2, bottle_y2 = bottle_box
                bottle_center_x = (bottle_x1 + bottle_x2) // 2

                # Проверяем, находится ли бутылка в центре по горизонтали
                if abs(bottle_center_x - center_x) < frame_width * 0.17:
                    bottle_height = bottle_y2 - bottle_y1

                    matched_cap_box = None
                    for cap_box in cap_boxes:
                        cap_x1, cap_y1, cap_x2, cap_y2 = cap_box
                        cap_height = cap_y2 - cap_y1

                        # Проверка пересечения крышки и бутылки
                        if cap_x1 < bottle_x2 and cap_x2 > bottle_x1:
                            matched_cap_box = cap_box
                            break

                    if matched_cap_box is not None:
                        cap_x1, cap_y1, cap_x2, cap_y2 = matched_cap_box
                        cap_height = cap_y2 - cap_y1
                        ratio = cap_height / bottle_height

                        # Определяем цвет в зависимости от соотношения высот
                        if ratio > THRESHOLD:
                            bottle_color = (0, 255, 0)  # Зелёный (нормально)
                            cap_color = (255, 0, 0)  # Синий (нормально)
                        else:
                            bottle_color = (0, 0, 255)  # Красный (дефект)
                            cap_color = (0, 0, 255)  # Красный (дефект)

                        # Рисуем контуры бутылки
                        cv2.rectangle(annotated_frame, (bottle_x1, bottle_y1), (bottle_x2, bottle_y2), bottle_color, 2)

                        # Рисуем контуры крышки
                        cv2.rectangle(annotated_frame, (cap_x1, cap_y1), (cap_x2, cap_y2), cap_color, 2)

                        # Рисуем линию, изменяя цвет в зависимости от положения
                        for y in range(bottle_y1, bottle_y2 + 1):
                            # Определяем, находится ли текущая позиция в области крышки
                            if cap_y1 <= y <= cap_y2:
                                if ratio > THRESHOLD:
                                    line_color = (255, 0, 0)  # Синий (нормально для крышки)
                                else:
                                    line_color = (0, 0, 200)  # Темно-красный для крышки при браке
                            else:  # Вне крышки, но в бутылке
                                if ratio > THRESHOLD:
                                    line_color = (0, 255, 0)  # Зеленый (нормально для бутылки)
                                else:
                                    line_color = (0, 0, 255)  # Ярко-красный для бутылки при браке

                            # Рисуем один пиксель линии
                            cv2.line(annotated_frame, (center_x, y), (center_x, y), line_color, 10)

                        # Остальная часть линии остается белой
                        for y in range(0, frame_height):
                            if bottle_y1 <= y <= bottle_y2:
                                continue  # Пропускаем область бутылки
                            cv2.line(annotated_frame, (center_x, y), (center_x, y), (255, 255, 255), 10)

                        # Выводим соотношение над бутылкой
                        text = f"Ratio: {ratio:.3f}"
                        cv2.putText(annotated_frame, text, (bottle_x1, bottle_y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bottle_color, 2)

            # Отображаем кадр и записываем его в выходное видео
            self.display_image(annotated_frame)
            out.write(annotated_frame)

            # Обрабатываем нажатие клавиш
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Освобождаем ресурсы
        self.cap.release()
        out.release()
        print(f"Видео сохранено в файл: {output_path}")

    def display_image(self, img):
        qformat = QImage.Format_RGB888
        if len(img.shape) == 3:
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
    window = BottleCapAnalyzer()
    window.show()
    sys.exit(app.exec_())
