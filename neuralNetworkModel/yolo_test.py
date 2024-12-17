from ultralytics import YOLO
import cv2
import numpy as np
import os

# Список цветов для различных классов
colors = [
    (255, 0, 0), (0, 255, 0)
]

def process_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    results = model.predict(image)[0]
    
    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

    # Подготовка словаря для группировки результатов по классам
    grouped_objects = {}

    # Рисование рамок и группировка результатов 
    for class_id, box in zip(classes, boxes):
        class_name = classes_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]  # Выбор цвета для класса
        if class_name not in grouped_objects:
            grouped_objects[class_name] = []
        grouped_objects[class_name].append(box)

        # Рисование рамок на изображении
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохранение измененного изображения
    new_image_path = os.path.splitext(image_path)[0] + '_yolo' + os.path.splitext(image_path)[1]
    cv2.imwrite(new_image_path, image)

    # Сохранение данных в текстовый файл
    text_file_path = os.path.splitext(image_path)[0] + '_data.txt'
    with open(text_file_path, 'w') as f:
        for class_name, details in grouped_objects.items():
            f.write(f"{class_name}:\n")
            for detail in details:
                f.write(f"Coordinates: ({detail[0]}, {detail[1]}, {detail[2]}, {detail[3]})\n")

    print(f"Processed {image_path}:")
    print(f"Saved bounding-box image to {new_image_path}")
    print(f"Saved data to {text_file_path}")

# Create a new YOLO model from scratch
#model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
#model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
#results = model.train(data="data.yaml", epochs=10)

# Evaluate the model's performance on the validation set
#results = model.val()

# Export the model to ONNX format
#success = model.export(format="onnx")

model = YOLO("yolov8n.pt")
model = YOLO("model/best.pt")

process_image('buffalo.jpg')

process_image('buffalo 2.jpg')

result = model.predict('buffalo 2.jpg')

result[0].save('result/resultMine.jpg')
