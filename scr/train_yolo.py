from ultralytics import YOLO


model = YOLO("yolov8.pt")

model.train(
    data="path/to/your/dataset.yaml",  # Укажите путь к dataset.yaml
    epochs=100,                         # Количество эпох обучения
    imgsz=640,                         # Размер изображений, например 640x640
    batch=16,                          # Размер батча (чем больше, тем быстрее, но требует больше памяти)
    name="custom_yolov8_model",        # Имя, под которым будут сохраняться результаты модели
    workers=2                          # Количество потоков для загрузки данных
)
