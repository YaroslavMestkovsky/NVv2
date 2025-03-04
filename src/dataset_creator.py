import os
import cv2
import numpy as np

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from random import randint


class DataSetCreator:
    def __init__(self, screenshots_dir, templates_dir, output_dir, examples_dir, class_id, num_threads=4):
        # Папка с полноразмерными скриншотами
        self.SCREENSHOTS_DIR = screenshots_dir
        # Папка с шаблонами
        self.TEMPLATES_DIR = templates_dir
        # Папка для выходных данных
        self.OUTPUT_DIR = output_dir
        # Папка для примеров
        self.EXAMPLES_DIR = os.path.join(self.OUTPUT_DIR, examples_dir)
        # Пороговое значение для совпадений
        self.CLASS_ID = class_id
        # ID класса объекта
        self.THRESHOLD = 0.75
        self.NMS_THRESHOLD = 0.5
        # Количество потоков
        self.num_threads = num_threads

        self.amount = len(os.listdir(self.SCREENSHOTS_DIR))
        self._init_templates()
        self._init_dirs()

    def _init_templates(self):
        self.templates = []

        for template_name in os.listdir(self.TEMPLATES_DIR):
            template_path = os.path.join(self.TEMPLATES_DIR, template_name)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)

            if template is None:
                print(f"Ошибка загрузки шаблона: {template_path}")
                continue

            self.templates.append((template, template.shape[:2]))

    def _init_dirs(self):
        os.makedirs(os.path.join(self.OUTPUT_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "examples"), exist_ok=True)

    def normalize_bbox(self, bbox, img_width, img_height):
        x1, y1, w, h = bbox
        x_center = (x1 + w / 2) / img_width
        y_center = (y1 + h / 2) / img_height
        norm_w = w / img_width
        norm_h = h / img_height

        return f"{self.CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

    def non_max_suppression(self, boxes, scores, threshold):
        if len(boxes) == 0:
            return []

        # Сортировка bounding box'ов по убыванию confidence score
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = np.argsort(scores)[::-1]
        boxes = boxes[indices]

        selected_boxes = []

        while len(boxes) > 0:
            # Выбираем bounding box с наибольшим confidence score
            selected_box = boxes[0]
            selected_boxes.append(selected_box)

            # Вычисляем IoU для остальных bounding box'ов
            ious = [self.compute_iou(selected_box, box) for box in boxes[1:]]
            mask = np.array(ious) < threshold
            boxes = boxes[1:][mask]

        return selected_boxes

    @staticmethod
    def compute_iou(box1, box2):
        # box1 и box2 в формате (x1, y1, x2, y2)
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def process_screenshot(self, screenshot_name):
        screenshot_path = os.path.join(self.SCREENSHOTS_DIR, screenshot_name)
        screenshot = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)

        if screenshot is None:
            print(f"Ошибка загрузки скриншота: {screenshot_path}")
            return

        img_height, img_width = screenshot.shape[:2]
        all_boxes = []  # Все bounding box'ы
        all_scores = []  # Confidence scores для всех bounding box'ов

        # Копия скриншота для рисования bounding box'ов
        annotated_image = screenshot.copy()

        for template, (h, w) in self.templates:
            result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.THRESHOLD)

            for pt in zip(*locations[::-1]):
                x1, y1 = pt
                x2, y2 = x1 + w, y1 + h
                all_boxes.append((x1, y1, x2, y2))
                all_scores.append(result[pt[1], pt[0]])  # Confidence score

        # Применение NMS
        if all_boxes:
            filtered_boxes = self.non_max_suppression(all_boxes, all_scores, self.NMS_THRESHOLD)

            annotations = []
            for box in filtered_boxes:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                bbox = (x1, y1, w, h)
                normalized_bbox = self.normalize_bbox(bbox, img_width, img_height)
                annotations.append(normalized_bbox)

                # Рисование bounding box'а
                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (randint(0, 255), randint(0, 255), randint(0, 255)),
                    2,
                )

            # Сохранение аннотаций
            base_name = os.path.splitext(screenshot_name)[0]
            label_path = os.path.join(self.OUTPUT_DIR, "labels", f"{base_name}.txt")
            with open(label_path, "w") as f:
                f.write("\n".join(annotations))

            # Копирование скриншота в папку images
            output_image_path = os.path.join(self.OUTPUT_DIR, "images", screenshot_name)
            cv2.imwrite(output_image_path, screenshot)

            # Сохранение примера с bounding box'ами
            example_path = os.path.join(self.EXAMPLES_DIR, screenshot_name)
            cv2.imwrite(example_path, annotated_image)

    def run(self):
        # Получаем список всех скриншотов
        screenshots = os.listdir(self.SCREENSHOTS_DIR)

        # Создаем пул потоков
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []

            for num, screenshot_name in enumerate(screenshots):
                print(f'Запуск обработки: {num + 1}/{self.amount}')
                futures.append(executor.submit(self.process_screenshot, screenshot_name))

            # Ожидаем завершения всех задач
            for future in as_completed(futures):
                try:
                    future.result()  # Проверяем на ошибки
                except Exception as e:
                    print(f"Ошибка при обработке: {e}")

        print("Разметка завершена!")


# Создаем экземпляр класса и запускаем обработку
creator = DataSetCreator(
    screenshots_dir="raw_data/fairy_fbS",
    templates_dir="templates/fairy_fbS",
    output_dir="../data/fairy_fbS",
    examples_dir="examples",
    class_id=0,
    num_threads=8,  # Укажите количество потоков
)
creator.run()
