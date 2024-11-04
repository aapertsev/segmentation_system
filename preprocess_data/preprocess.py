import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter

def compute_dataset_mean_std(dataset_dir):
    """
    Вычисляет среднее и стандартное отклонение для всего датасета изображений в формате .jpg.

    Args:
        dataset_dir (str): Путь к директории с изображениями.

    Returns:
        tuple: Среднее и стандартное отклонение по всему датасету.
    """
    sum_pixels = 0
    sum_squares = 0
    total_pixels = 0

    for filename in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, filename)

        if filename.endswith('.jpg'):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = image.astype(np.float32)

            sum_pixels += image.sum()
            sum_squares += np.square(image).sum()

            total_pixels += image.size

    mean = sum_pixels / total_pixels

    std = np.sqrt((sum_squares / total_pixels) - mean ** 2)

    return mean, std


def normalize_density(data, mean, std):
    """
    Нормализует плотность данных с использованием Z-нормализации на основе
    глобального среднего и стандартного отклонения всего датасета.

    Args:
        data (numpy.ndarray): Входные данные, представляющие собой массив плотности.
        mean (float): Среднее значение для всего датасета.
        std (float): Стандартное отклонение для всего датасета.

    Returns:
        numpy.ndarray: Нормализованные данные.
    """

    normalized_data = (data - mean) / std
    return normalized_data


def filter_noise(data, sigma=1):
    """
    Фильтрует шумы в данных с использованием гауссового фильтра.

    Args:
        data (numpy.ndarray): Входные данные, на которых нужно применить фильтрацию.
        sigma (float): Стандартное отклонение гауссового фильтра.

    Returns:
        numpy.ndarray: Данные после фильтрации шумов.
    """
    filtered_data = gaussian_filter(data, sigma=sigma)

    return filtered_data


def preprocess(data, mean, std, sigma=1):
    """
    Выполняет предварительную обработку данных, включая нормализацию и фильтрацию шумов.

    Args:
        data (numpy.ndarray): Входные данные для предварительной обработки.
        sigma (float): Стандартное отклонение для фильтрации шумов.

    Returns:
        numpy.ndarray: Данные после обработки.
    """
    normalized = normalize_density(data, mean, std)
    filtered = filter_noise(normalized, sigma)
    return filtered


if __name__ == "__main__":
    pass
