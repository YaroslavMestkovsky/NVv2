import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DatasetSplitter:
    """
    Класс для разделения датасета на train, val и test.
    """

    def __init__(self, dataset_dir, output_dir="split_dataset", train_ratio=0.7, val_ratio=0.15):
        """
        Инициализация класса.

        :param dataset_dir: Путь к папке с размеченным датасетом (должна содержать images/ и labels/).
        :param output_dir: Папка для сохранения разделенного датасета.
        :param train_ratio: Доля данных для обучения (например, 0.7 = 70%).
        :param val_ratio: Доля данных для валидации (например, 0.15 = 15%).
        """

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Проверка наличия необходимых папок
        self.images_dir = os.path.join(self.dataset_dir, "images/raw")
        self.labels_dir = os.path.join(self.dataset_dir, "labels/raw")

        if not os.path.exists(self.images_dir) or not os.path.exists(self.labels_dir):
            raise ValueError("Папки 'images' и 'labels' должны быть в указанной директории датасета.")

        # Создание выходных папок
        self.train_images_dir = os.path.join(self.output_dir, "images", "train")
        self.val_images_dir = os.path.join(self.output_dir, "images", "val")
        self.test_images_dir = os.path.join(self.output_dir, "images", "test")
        self.train_labels_dir = os.path.join(self.output_dir, "labels", "train")
        self.val_labels_dir = os.path.join(self.output_dir, "labels", "val")
        self.test_labels_dir = os.path.join(self.output_dir, "labels", "test")

        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.val_images_dir, exist_ok=True)
        os.makedirs(self.test_images_dir, exist_ok=True)
        os.makedirs(self.train_labels_dir, exist_ok=True)
        os.makedirs(self.val_labels_dir, exist_ok=True)
        os.makedirs(self.test_labels_dir, exist_ok=True)

    def split(self):
        """
        Разделение датасета на train, val и test.
        """

        # Получение списка файлов изображений
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Первое разделение: train + val vs test
        train_val_files, test_files = train_test_split(
            image_files,
            test_size=(1 - self.train_ratio - self.val_ratio),
            random_state=42,
        )

        # Второе разделение: train vs val
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=self.val_ratio / (self.train_ratio + self.val_ratio),
            random_state=42,
        )

        # Копирование файлов в соответствующие папки
        self._copy_files(train_files, self.train_images_dir, self.train_labels_dir)
        self._copy_files(val_files, self.val_images_dir, self.val_labels_dir)
        self._copy_files(test_files, self.test_images_dir, self.test_labels_dir)

        print(f"Датасет успешно разделен:")
        print(f"- Train: {len(train_files)} изображений")
        print(f"- Validation: {len(val_files)} изображений")
        print(f"- Test: {len(test_files)} изображений")

    def _copy_files(self, file_list, images_dir, labels_dir):
        """
        Копирование файлов изображений и аннотаций в указанные папки.

        :param file_list: Список файлов изображений.
        :param images_dir: Папка для изображений.
        :param labels_dir: Папка для аннотаций.
        """

        for img in tqdm(file_list, total=len(file_list), desc="Обработка"):
            label_file = img.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            shutil.copy(os.path.join(self.images_dir, img), os.path.join(images_dir, img))
            shutil.copy(os.path.join(self.labels_dir, label_file), os.path.join(labels_dir, label_file))


# Пример использования
if __name__ == "__main__":
    splitter = DatasetSplitter(
        dataset_dir="../data/marked_up_data/fairy_fbS", # Путь к вашему размеченному датасету
        output_dir="../data/marked_up_data/fairy_fbS", # Папка для сохранения разделенного датасета
        train_ratio=0.7, # 70% данных для обучения
        val_ratio=0.15,
    )
    splitter.split()
