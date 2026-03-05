#!/usr/bin/env python3
"""
Базовый пример использования библиотеки MSUROVERTEAM-CV.
Детекция стрелок и конусов на изображении.

Использование:
    1. Поместите изображение со стрелкой или конусом в корневую директорию
       репозитория (MSUROVERTEAM-CV/) и назовите его test_image.jpg,
       либо передайте путь к изображению аргументом командной строки.
    2. Запустите скрипт из корня репозитория:
       python3 example_basic.py
       python3 example_basic.py path/to/your_image.jpg
"""

import cv2
import os
import sys
from pathlib import Path

# Добавляем путь к библиотеке
sys.path.append(str(Path(__file__).parent))

os.environ["YOLO_VERBOSE"] = "False"  # Подавляем отладочный вывод YOLO
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"  # Подавляем предупреждения OpenCV
from eureka_nav_lib import NavigationDetector


def main():
    # Шаг 1: Инициализация детектора
    print("Инициализация детектора...")
    weights = Path(__file__).parent / "weights" / "best.pt"
    detector = NavigationDetector(
        weights_path=str(weights),
        device=None  # Автоматический выбор устройства (GPU если доступно)
    )

    # Шаг 2: Загрузка изображения
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = str(Path(__file__).parent / "test_image.jpg")

    print(f"Загрузка изображения: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Ошибка: не удалось загрузить изображение '{image_path}'")
        print("Поместите изображение со стрелкой или конусом в корневую")
        print("директорию репозитория (MSUROVERTEAM-CV/) под именем test_image.jpg,")
        print("либо передайте путь аргументом: python3 example_basic.py <путь>")
        return

    # Шаг 3: Детекция всех объектов
    print("Выполнение детекции...")
    detections = detector.detect_all(image)

    # Шаг 4: Вывод результатов
    print(f"\nОбнаружено объектов: {len(detections)}")
    print("-" * 60)

    for i, det in enumerate(detections, 1):
        print(f"Объект #{i}:")
        print(f"  Тип: {det.object_type}")
        print(f"  Направление: {det.direction}")
        print(f"  Расстояние: {det.distance_m:.2f} м")
        print(f"  Угол: {det.angle_deg:.1f}°")
        print(f"  Уверенность: {det.confidence:.2%}")
        print(f"  Координаты: {det.bbox}")
        print()

    # Шаг 5: Визуализация результатов
    for det in detections:
        x1, y1, x2, y2 = det.bbox

        # Выбор цвета в зависимости от типа
        color = (0, 255, 0) if det.object_type == "arrow" else (0, 165, 255)

        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Подготовка текста
        label = f"{det.object_type}"
        if det.direction != "none":
            label += f" {det.direction}"
        label += f" {det.distance_m:.1f}m"

        # Рисуем текст
        cv2.putText(image, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохранение результата
    output_path = str(Path(__file__).parent / "result.jpg")
    cv2.imwrite(output_path, image)
    print(f"Результат сохранен в {output_path}")


if __name__ == "__main__":
    main()
