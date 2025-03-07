import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DatasetSplitter:
    """
    Класс для разделения датасета на train и val.
    """

    def __init__(self, dataset_dir, output_dir="split_dataset", test_size=0.2, random_state=42):
        """
        Инициализация класса.

        :param dataset_dir: Путь к папке с размеченным датасетом (должна содержать images/ и labels/).
        :param output_dir: Папка для сохранения разделенного датасета.
        :param test_size: Доля данных для валидации (например, 0.2 = 20%).
        :param random_state: Сид для воспроизводимости разбиения.
        """

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state

        # Проверка наличия необходимых папок
        self.images_dir = os.path.join(self.dataset_dir, "images", "raw")
        self.labels_dir = os.path.join(self.dataset_dir, "labels", "raw")

        if not os.path.exists(self.images_dir) or not os.path.exists(self.labels_dir):
            raise ValueError("Папки 'images' и 'labels' должны быть в указанной директории датасета.")

        # Создание выходных папок
        self.train_images_dir = os.path.join(self.output_dir, "images", "train")
        self.val_images_dir = os.path.join(self.output_dir, "images", "val")

        self.train_labels_dir = os.path.join(self.output_dir, "labels", "train")
        self.val_labels_dir = os.path.join(self.output_dir, "labels", "val")

        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.val_images_dir, exist_ok=True)
        os.makedirs(self.train_labels_dir, exist_ok=True)
        os.makedirs(self.val_labels_dir, exist_ok=True)

    def split(self):
        """
        Разделение датасета на train и val.
        """

        # Получение списка файлов изображений
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Разделение на train и val
        train_images, val_images = train_test_split(
            image_files,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Копирование файлов в соответствующие папки
        for img in tqdm(train_images, total=len(train_images), desc="Обработка"):
            label_file = img.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            shutil.copy(os.path.join(self.images_dir, img), os.path.join(self.train_images_dir, img))
            shutil.copy(os.path.join(self.labels_dir, label_file), os.path.join(self.train_labels_dir, label_file))

        for img in tqdm(val_images, total=len(val_images), desc="Обработка"):
            label_file = img.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            shutil.copy(os.path.join(self.images_dir, img), os.path.join(self.val_images_dir, img))
            shutil.copy(os.path.join(self.labels_dir, label_file), os.path.join(self.val_labels_dir, label_file))

        print(f"Датасет успешно разделен: {len(train_images)} train, {len(val_images)} val.")


# Пример использования
if __name__ == "__main__":
    splitter = DatasetSplitter(
        dataset_dir="../data/marked_up_data/fairy_fbS", # Путь к вашему размеченному датасету
        output_dir="../data/marked_up_data/fairy_fbS", # Папка для сохранения разделенного датасета
        test_size=0.2 # 20% данных отправляются в val
    )
    splitter.split()
