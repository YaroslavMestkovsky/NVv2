import os
import time
import win32gui
import win32con
from uuid import uuid4
from tqdm import tqdm
from mss import mss
from PIL import Image


class DatasetCreator:
    """
    Класс для создания скриншотов окна игры.
    """

    def __init__(self, hwnd, output_dir="screenshots", target_size=(1280, 1280), interval=0.5):
        """
        Инициализация класса.

        :param hwnd: Идентификатор окна игры (HWND).
        :param output_dir: Папка для сохранения скриншотов.
        :param target_size: Целевой размер скриншотов (ширина, высота).
        :param interval: Интервал между скриншотами в секундах.
        """

        self.hwnd = hwnd
        self.output_dir = output_dir
        self.target_size = target_size
        self.interval = interval

        # Создание папки для скриншотов
        os.makedirs(self.output_dir, exist_ok=True)

    def _resize_window(self, width, height):
        """
        Изменение размера окна игры.

        :param width: Новая ширина окна.
        :param height: Новая высота окна.
        """

        win32gui.SetWindowPos(
            self.hwnd,
            win32con.HWND_TOP,
            0, 0,  # Положение окна (игнорируется, если окно не перемещается)
            width, height,
            win32con.SWP_NOMOVE | win32con.SWP_NOZORDER,
        )

    def _get_client_area(self):
        """
        Получение координат клиентской области окна без заголовка и рамок.

        :return: Словарь с координатами клиентской области (left, top, width, height).
        """

        # Получаем координаты всего окна (включая заголовок и рамки)
        window_rect = win32gui.GetWindowRect(self.hwnd)

        # Получаем размеры клиентской области (без заголовка и рамок)
        client_rect = win32gui.GetClientRect(self.hwnd)

        # Вычисляем смещение заголовка и рамок
        border_width = (window_rect[2] - window_rect[0]) - client_rect[2]
        title_bar_height = (window_rect[3] - window_rect[1]) - client_rect[3] - 8

        # Координаты клиентской области
        left = window_rect[0] + border_width // 2
        top = window_rect[1] + title_bar_height
        width = client_rect[2]
        height = client_rect[3]

        return {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }

    def capture_screenshots(self, num_shots=None):
        """
        Захват скриншотов окна игры.

        :param num_shots: Максимальное количество скриншотов (None = бесконечно).
        """

        # Устанавливаем целевой размер окна
        self._resize_window(self.target_size[0], self.target_size[1])

        # Получаем координаты клиентской области
        client_area = self._get_client_area()

        with mss() as sct:
            for _ in tqdm(range(1, num_shots + 1), desc="Захват скриншотов", unit="скриншот"):
                # Захват клиентской области
                screenshot = sct.grab(client_area)

                # Преобразование в изображение PIL
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

                # Сохранение скриншота
                output_path = os.path.join(self.output_dir, f"screenshot_{str(uuid4())[:8]}.png")
                img.save(output_path)

                # Ожидание перед следующим скриншотом
                time.sleep(self.interval)

    def run(self, num_shots=None):
        """
        Запуск процесса захвата скриншотов.

        :param num_shots: Максимальное количество скриншотов (None = бесконечно).
        """

        print("Начинаем захват скриншотов...")

        try:
            self.capture_screenshots(num_shots)
        except KeyboardInterrupt:
            print("\nЗахват скриншотов остановлен.")


# Получение HWND окна игры
def find_game_window(title):
    hwnd = win32gui.FindWindow(None, title)

    if hwnd == 0:
        raise ValueError(f"Окно с заголовком '{title}' не найдено!")

    return hwnd

game_title = "Warspear Online"
hwnd = find_game_window(game_title)

# Создание экземпляра класса и запуск захвата скриншотов
screenshotter = DatasetCreator(
    hwnd=hwnd,
    output_dir="raw_data/fairy_fbS",
    target_size=(1280, 1280),
    interval=0.3,
)
screenshotter.run(num_shots=100)
