import os
import cv2
import numpy as np

# Константы
SCREENSHOTS_DIR = "raw_data/fairy_fbS"  # Папка с полноразмерными скриншотами
TEMPLATES_DIR = "templates/fairy_fbS"      # Папка с шаблонами
OUTPUT_DIR = "../data/fairy_fbS"              # Папка для выходных данных
EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "examples")  # Папка для примеров
THRESHOLD = 0.3                # Пороговое значение для совпадений
CLASS_ID = 0                     # ID класса для всех объектов (можно изменить)

# Создание выходных папок
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "examples"), exist_ok=True)

# Загрузка шаблонов
templates = []
for template_name in os.listdir(TEMPLATES_DIR):
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        print(f"Ошибка загрузки шаблона: {template_path}")
        continue
    templates.append((template, template.shape[:2]))  # (шаблон, (высота, ширина))

# Функция для нормализации координат
def normalize_bbox(bbox, img_width, img_height):
    x1, y1, w, h = bbox
    x_center = (x1 + w / 2) / img_width
    y_center = (y1 + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

# Обработка скриншотов
for screenshot_name in os.listdir(SCREENSHOTS_DIR):
    screenshot_path = os.path.join(SCREENSHOTS_DIR, screenshot_name)
    screenshot = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    if screenshot is None:
        print(f"Ошибка загрузки скриншота: {screenshot_path}")
        continue

    img_height, img_width = screenshot.shape[:2]
    annotations = []

    # Копия скриншота для рисования bounding box'ов
    annotated_image = screenshot.copy()

    for template, (h, w) in templates:
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= THRESHOLD)

        for pt in zip(*locations[::-1]):
            x1, y1 = pt
            bbox = (x1, y1, w, h)
            normalized_bbox = normalize_bbox(bbox, img_width, img_height)
            annotations.append(normalized_bbox)

            # Рисование bounding box'а
            cv2.rectangle(annotated_image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    # Сохранение аннотаций
    if annotations:
        base_name = os.path.splitext(screenshot_name)[0]
        label_path = os.path.join(OUTPUT_DIR, "labels", f"{base_name}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(annotations))

        # Копирование скриншота в папку images
        output_image_path = os.path.join(OUTPUT_DIR, "images", screenshot_name)
        cv2.imwrite(output_image_path, screenshot)

        # Сохранение примера с bounding box'ами
        example_path = os.path.join(EXAMPLES_DIR, screenshot_name)
        cv2.imwrite(example_path, annotated_image)

print("Разметка завершена!")