import sys
import os
import re
import numpy as np
import nibabel as nib
import matplotlib
# Если хотите интерактивные окна гистограмм на Windows/Mac/Linux, закомментируйте "Agg":
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import subprocess

from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QPlainTextEdit, QSlider, QListWidget, QListWidgetItem,
    QCheckBox, QLabel, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QTableWidgetItem, QVBoxLayout
)


def convert_jpg_to_nii(input_dir):
    """
    Находит все JPG-файлы в input_dir, создает подпапку 'nii' и конвертирует каждый снимок
    в nii.gz с переименованием по схеме "patient_XXX_0000.nii.gz" (XXX – 3-значный номер).
    Возвращает путь к созданной директории с nii-файлами.
    """
    nii_dir = os.path.join(input_dir, "nii")
    os.makedirs(nii_dir, exist_ok=True)

    jpg_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")])
    if not jpg_files:
        raise RuntimeError("Не найдены JPG-файлы в указанной директории.")

    for idx, jpg_file in enumerate(jpg_files):
        img_path = os.path.join(input_dir, jpg_file)
        try:
            img = Image.open(img_path).convert("L")  # переводим в градации серого
        except Exception as e:
            print(f"Ошибка при открытии {jpg_file}: {e}")
            continue
        arr = np.array(img)
        # Если снимок 2D, делаем (H, W, 1)
        arr3d = np.expand_dims(arr, axis=2)

        new_filename = f"patient_{idx:03d}_0000.nii.gz"
        out_path = os.path.join(nii_dir, new_filename)
        nii_img = nib.Nifti1Image(arr3d, affine=np.eye(4))
        nib.save(nii_img, out_path)

    return nii_dir


class MainWindow(QMainWindow):
    """
    Главное окно:
      - При "Запуск предсказания": пользователь указывает директорию с JPG.
        -> Создаём подпапку nii (конвертируем JPG->NIfTI)
        -> Создаём подпапку results
        -> Запускаем nnUNetv2_predict -i <nii_dir> -o <results_dir>
      - По завершении инференса открываем окно визуализации, которое ищет файлы в 'nii' и 'results'.
      - Кнопка "Визуализация" вручную делает то же самое (просит путь к JPG, а results = подпапка).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сегментация КТ снимков (nnU-Net)")
        self.setMinimumSize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.predict_button = QPushButton("Запуск предсказания")
        self.predict_button.clicked.connect(self.run_inference)

        self.visualize_button = QPushButton("Визуализация")
        self.visualize_button.clicked.connect(self.open_visualization)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)

        main_layout.addWidget(self.predict_button)
        main_layout.addWidget(self.visualize_button)
        main_layout.addWidget(self.log_text)

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.finished.connect(self.on_finished)

        # Сохраняем пути, чтобы открыть окно визуализации
        self.input_dir = None
        self.nii_dir = None
        self.results_dir = None

    def run_inference(self):
        """
        Пользователь указывает директорию с JPG-снимками.
        -> конвертируем JPG->nii в подпапку nii
        -> создаём папку results
        -> запускаем nnUNetv2_predict
        """
        input_dir = QFileDialog.getExistingDirectory(
            self, "Выберите директорию с JPG снимками"
        )
        if not input_dir:
            return
        self.input_dir = input_dir

        try:
            self.nii_dir = convert_jpg_to_nii(input_dir)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка конвертации", str(e))
            return

        self.results_dir = os.path.join(input_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.log_text.clear()

        # Команда инференса
        command = [
            "nnUNetv2_predict",
            "-i", self.nii_dir,
            "-o", self.results_dir,
            "-d", "1",
            "-c", "2d",
            "-device", "cpu",
            "-chk", "checkpoint_best.pth",
            "-f", "4"
        ]
        self.process.start(command[0], command[1:])

    def on_stdout(self):
        text = self.process.readAllStandardOutput().data().decode("utf-8")
        self.log_text.appendPlainText(text)

    def on_stderr(self):
        text = self.process.readAllStandardError().data().decode("utf-8")
        self.log_text.appendPlainText(text)

    def on_finished(self, exit_code, exit_status):
        if exit_code == 0:
            QMessageBox.information(self, "Готово!", "Предсказание завершено.")
            self.open_visualization_automatically()
        else:
            QMessageBox.warning(self, "Ошибка",
                                f"Процесс завершился с ошибкой (код {exit_code}).")

    def open_visualization_automatically(self):
        if not self.input_dir:
            return
        # Папка results = <input_dir>/results
        res_dir = os.path.join(self.input_dir, "results")
        self.vis_window = VisualizationWindow(self.input_dir, res_dir)
        self.vis_window.show()

    def open_visualization(self):
        input_dir = QFileDialog.getExistingDirectory(
            self, "Выберите директорию с JPG снимками (результаты будут в 'results')"
        )
        if not input_dir:
            return
        res_dir = os.path.join(input_dir, "results")
        self.vis_window = VisualizationWindow(input_dir, res_dir)
        self.vis_window.show()


class VisualizationWindow(QWidget):
    """
    Окно визуализации, в котором:
      - Ищем пары: "nii/patient_XXX_0000.nii.gz" и "results/patient_XXX.nii.gz" (пример)
      - Показываем список пациентов
      - Отображаем исходник + маску
      - Кнопка "Таблица плотностей" -> показывает проценты по интервалам плотностей
      - Кнопка "Сохранить отчет (PDF)" -> строим PDF, включаем туда:
          * Исходник + маска
          * Карта плотности
          * Гистограмму
          * Текст статистики
          * (Новое) Таблицу плотностей
    """
    def __init__(self, images_dir, masks_dir, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir   # Папка с JPG (и подпапка 'nii')
        self.masks_dir = masks_dir    # Папка 'results'
        self.setWindowTitle("Визуализация")

        self.pairs = {}
        self.current_patient_id = None
        self.current_image_3d = None
        self.current_mask_3d = None
        self.current_slice = 0
        self.num_slices = 1

        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self.on_patient_selected)

        # Классы: 1=Сердце, 2=Аорта, 3=Легочная
        self.class_indices = [1, 2, 3]
        self.class_names = ["Сердце", "Аорта", "Легочная Артерия"]
        self.class_checkboxes = []
        for name in self.class_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self.on_class_toggled)
            self.class_checkboxes.append(cb)

        self.stats_label = QLabel("Здесь будет легенда со средними/медианами")
        self.stats_label.setStyleSheet("background-color: white;")

        self.image_label = QLabel("Исходник + маска (текущий срез)")
        self.image_label.setFixedSize(512, 512)
        self.image_label.setStyleSheet("background-color: #cccccc;")

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setVisible(False)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        self.density_map_button = QPushButton("Построить карту плотности")
        self.density_map_button.clicked.connect(self.build_density_map)

        self.hist_button = QPushButton("Построить гистограмму плотности")
        self.hist_button.clicked.connect(self.build_histogram)

        self.save_report_button = QPushButton("Сохранить отчет (PDF)")
        self.save_report_button.clicked.connect(self.save_report_to_pdf)

        # Новая кнопка "Таблица плотностей"
        self.density_table_button = QPushButton("Таблица плотностей")
        self.density_table_button.clicked.connect(self.build_density_table)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.list_widget)
        for cb in self.class_checkboxes:
            left_layout.addWidget(cb)
        left_layout.addWidget(self.density_table_button)
        left_layout.addWidget(self.save_report_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.stats_label)
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.slice_slider)
        right_layout.addWidget(self.density_map_button)
        right_layout.addWidget(self.hist_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)
        self.setLayout(main_layout)

        self.find_pairs()
        self.populate_list()

    def find_pairs(self):
        """
        В папке <images_dir>/nii ищем файлы patient_XXX_0000.nii.gz
        В папке masks_dir (results) ищем patient_XXX.nii.gz
        """
        nii_dir = os.path.join(self.images_dir, "nii")
        if not os.path.isdir(nii_dir):
            QMessageBox.warning(self, "Ошибка", f"Не найдена директория nii в {self.images_dir}")
            return

        nii_files = sorted([f for f in os.listdir(nii_dir) if f.lower().endswith(".nii.gz")])
        for f in nii_files:
            # f: "patient_000_0000.nii.gz" -> mask: "patient_000.nii.gz"
            match = re.match(r'^(patient_\d{3})_0000\.nii\.gz$', f)
            if match:
                patient_id = match.group(1)  # "patient_000"
                image_path = os.path.join(nii_dir, f)
                mask_name = f"{patient_id}.nii.gz"  # "patient_000.nii.gz"
                mask_path = os.path.join(self.masks_dir, mask_name)
                if os.path.exists(mask_path):
                    self.pairs[patient_id] = (image_path, mask_path)
                else:
                    print(f"Маска для {patient_id} не найдена в {self.masks_dir}")
            else:
                print(f"Файл {f} не соответствует шаблону patient_XXX_0000.nii.gz")

    def populate_list(self):
        self.list_widget.clear()
        for pid in sorted(self.pairs.keys()):
            item = QListWidgetItem(pid)
            self.list_widget.addItem(item)

    def on_patient_selected(self, current_item, previous_item):
        if not current_item:
            return
        self.current_patient_id = current_item.text()
        img_path, msk_path = self.pairs[self.current_patient_id]

        self.current_image_3d = self.load_nii_as_3d_uint8(img_path)
        self.current_mask_3d = self.load_nii_as_3d_int(msk_path)

        self.num_slices = self.current_image_3d.shape[2]
        if self.num_slices > 1:
            self.slice_slider.setRange(0, self.num_slices-1)
            self.slice_slider.setVisible(True)
        else:
            self.slice_slider.setVisible(False)

        self.current_slice = 0
        self.compute_and_show_stats()
        self.update_display()

    def on_slice_changed(self, value):
        self.current_slice = value
        self.update_display()

    def on_class_toggled(self, state):
        self.update_display()

    def compute_and_show_stats(self):
        if self.current_image_3d is None or self.current_mask_3d is None:
            return
        text_lines = []
        for class_val, class_name in zip(self.class_indices, self.class_names):
            mask_region = (self.current_mask_3d == class_val)
            values = self.current_image_3d[mask_region]
            if values.size > 0:
                mean_val = values.mean()
                median_val = np.median(values)
                text_lines.append(f"{class_name}: mean={mean_val:.2f}, median={median_val:.2f}")
            else:
                text_lines.append(f"{class_name}: не сегментировано")

        self.stats_label.setText("\n".join(text_lines))

    def update_display(self):
        if self.current_image_3d is None or self.current_mask_3d is None:
            return
        z = self.current_slice
        overlay_rgb = self.make_overlay_slice(z)
        h, w, _ = overlay_rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(overlay_rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def build_density_map(self):
        if self.current_image_3d is None or self.current_mask_3d is None:
            return
        z = self.current_slice
        density_rgb = self.make_density_map_slice(z)
        map_dialog = QDialog(self)
        map_dialog.setWindowTitle("Карта плотности")
        label_map = QLabel()
        label_map.setFixedSize(512, 512)
        h, w, _ = density_rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(density_rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label_map.setPixmap(pixmap.scaled(label_map.size(), Qt.KeepAspectRatio))
        layout = QVBoxLayout(map_dialog)
        layout.addWidget(label_map)
        map_dialog.exec_()

    def build_histogram(self):
        if self.current_image_3d is None or self.current_mask_3d is None:
            return
        mask_region = (self.current_mask_3d != 0)
        values = self.current_image_3d[mask_region]
        plt.figure()
        plt.hist(values.flatten(), bins=50, color='gray')
        plt.title("Гистограмма плотности")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.show()

    def make_overlay_slice(self, z):
        img_slice_2d = self.current_image_3d[:, :, z]
        msk_slice_2d = self.current_mask_3d[:, :, z]
        h, w = img_slice_2d.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = img_slice_2d

        color_map = {
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
        }
        alpha = 0.5
        for cb, class_val in zip(self.class_checkboxes, self.class_indices):
            if cb.isChecked():
                c_r, c_g, c_b = color_map[class_val]
                mask_region = (msk_slice_2d == class_val)
                rgb[mask_region, 0] = np.uint8((1 - alpha)*rgb[mask_region, 0] + alpha*c_r)
                rgb[mask_region, 1] = np.uint8((1 - alpha)*rgb[mask_region, 1] + alpha*c_g)
                rgb[mask_region, 2] = np.uint8((1 - alpha)*rgb[mask_region, 2] + alpha*c_b)
        return rgb

    def make_density_map_slice(self, z):
        img_slice_2d = self.current_image_3d[:, :, z]
        msk_slice_2d = self.current_mask_3d[:, :, z]
        h, w = img_slice_2d.shape
        density_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        color_map_array = np.zeros((256, 3), dtype=np.uint8)
        for val in range(256):
            frac = val / 255.0
            r = 255
            g = int(255 * (1.0 - frac))
            b = 0
            color_map_array[val] = (r, g, b)
        for i in range(h):
            for j in range(w):
                intensity = img_slice_2d[i, j]
                if msk_slice_2d[i, j] == 0:
                    density_rgb[i, j] = (intensity, intensity, intensity)
                else:
                    density_rgb[i, j] = color_map_array[intensity]
        return density_rgb

    def load_nii_as_3d_uint8(self, path_nii):
        nifti_img = nib.load(path_nii)
        data = nifti_img.get_fdata()
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        d_min, d_max = data.min(), data.max()
        if abs(d_max - d_min) < 1e-8:
            d_max = d_min + 1
        data_8 = ((data - d_min)/(d_max - d_min)*255).astype(np.uint8)
        return data_8

    def load_nii_as_3d_int(self, path_nii):
        nifti_img = nib.load(path_nii)
        data = nifti_img.get_fdata()
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        data_int = data.astype(np.int32)
        return data_int

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    # ------------------------ Таблица плотностей (по интервалам) ----------------------------
    def build_density_table(self):
        """
        Показывает QDialog с таблицей, где строки = интервалы плотности,
        столбцы = [Сердце, Аорта, Легочная, Все].
        Ячейки = процент пикселей, попавших в данный диапазон.
        """
        if self.current_image_3d is None or self.current_mask_3d is None:
            QMessageBox.warning(self, "Ошибка", "Нет загруженных данных (изображение/маска).")
            return

        intervals = [(0, 50), (51, 100), (101, 150), (151, 200)]
        class_names = ["Сердце", "Аорта", "Легочная артерия", "Все"]
        result_matrix = self.compute_density_matrix(intervals, class_names)

        # Покажем таблицу в QDialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Таблица плотностей (проценты)")

        table = QTableWidget(dialog)
        table.setRowCount(len(intervals))
        table.setColumnCount(len(class_names))
        table.setHorizontalHeaderLabels(class_names)
        row_labels = [f"{low}-{high}" for (low, high) in intervals]
        table.setVerticalHeaderLabels(row_labels)

        for r in range(len(intervals)):
            for c in range(len(class_names)):
                val = result_matrix[r][c]
                item = QTableWidgetItem(f"{val:.2f}%")
                table.setItem(r, c, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout = QVBoxLayout(dialog)
        layout.addWidget(table)
        dialog.exec_()

    def compute_density_matrix(self, intervals, class_names):
        """
        Считает 2D-массив процентов:
        rows = интервалы (low, high)
        cols = [Сердце, Аорта, Легочная, Все]
        """
        # Сопоставим class_names -> class_vals
        # "Все" => mask != 0
        name_to_val = {
            "Сердце": 1,
            "Аорта": 2,
            "Легочная артерия": 3,
            "Все": None
        }
        # Подготовим словарь intensity_arrays[class_name] = массив интенсивностей
        intensity_arrays = {}
        for cname in class_names:
            cval = name_to_val[cname]
            if cval is None:
                # все структуры
                mask_region = (self.current_mask_3d != 0)
            else:
                mask_region = (self.current_mask_3d == cval)
            pix_vals = self.current_image_3d[mask_region]
            intensity_arrays[cname] = pix_vals

        num_rows = len(intervals)
        num_cols = len(class_names)
        result = [[0.0]*num_cols for _ in range(num_rows)]

        for c, cname in enumerate(class_names):
            pix_vals = intensity_arrays[cname]
            total_count = len(pix_vals)
            if total_count == 0:
                # всё 0%
                continue
            for r, (low, high) in enumerate(intervals):
                count_in_range = np.count_nonzero((pix_vals >= low) & (pix_vals <= high))
                percent = (count_in_range / total_count) * 100.0
                result[r][c] = percent
        return result

    # -------------------------- Сохранение отчета (PDF) --------------------------
    def save_report_to_pdf(self):
        """
        Для каждого patient_id формируем несколько страниц:
          1) Титульная страница
          2) Для каждого пациента: (overlay, density, hist, stats)
          3) Страница с таблицей плотностей
        """
        if not self.pairs:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения в отчет.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Выберите директорию, куда сохранить отчет PDF")
        if not save_dir:
            return

        pdf_name = f"report_{os.path.basename(self.images_dir)}.pdf"
        pdf_path = os.path.join(save_dir, pdf_name)

        intervals = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 255)]
        class_names = ["Сердце", "Аорта", "Легочная", "Все"]

        with PdfPages(pdf_path) as pdf:
            # Титульная страница
            fig_title = plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5,
                     f"Отчет по {os.path.basename(self.images_dir)}",
                     ha='center', va='center', fontsize=16)
            plt.axis('off')
            pdf.savefig(fig_title)
            plt.close(fig_title)

            # Для каждого patientID делаем страницу с overlay/density/hist/stats
            for pid in sorted(self.pairs.keys()):
                img_path, msk_path = self.pairs[pid]
                image_3d = self.load_nii_as_3d_uint8(img_path)
                mask_3d = self.load_nii_as_3d_int(msk_path)

                z_central = mask_3d.shape[2] // 2
                overlay_rgb = self.make_overlay_slice_single(image_3d, mask_3d, z_central)
                density_rgb = self.make_density_map_slice_single(image_3d, mask_3d, z_central)
                stats_text = self.generate_stats_text(image_3d, mask_3d, pid)

                # Страница с 4 под-рисунками
                fig = plt.figure(figsize=(10, 8))
                fig.suptitle(f"Файл: {pid}", fontsize=14)

                ax1 = fig.add_subplot(2, 2, 1)
                ax1.set_title("Исходник + маска (центр. срез)")
                ax1.imshow(overlay_rgb)
                ax1.axis('off')

                ax2 = fig.add_subplot(2, 2, 2)
                ax2.set_title("Карта плотности (тот же срез)")
                ax2.imshow(density_rgb)
                ax2.axis('off')

                ax3 = fig.add_subplot(2, 2, 3)
                ax3.set_title("Гистограмма (mask!=0)")
                mask_region = (mask_3d != 0)
                values = image_3d[mask_region]
                ax3.hist(values.flatten(), bins=50, color='gray')
                ax3.set_xlabel("Intensity")
                ax3.set_ylabel("Count")

                ax4 = fig.add_subplot(2, 2, 4)
                ax4.set_title("Статистика (средняя, медиана)")
                ax4.text(0.05, 0.95, stats_text, fontsize=10, va='top', ha='left')
                ax4.axis('off')

                pdf.savefig(fig)
                plt.close(fig)

                # Теперь добавим страницу с таблицей плотностей
                fig_table = plt.figure(figsize=(8, 4))
                ax_table = fig_table.add_subplot(111)
                ax_table.set_title(f"Таблица плотностей: {pid}")

                # Считаем матрицу
                density_mat = self.compute_density_matrix_for_volume(image_3d, mask_3d, intervals, class_names)
                # Строим данные для matplotlib.table
                table_data = []
                # Шапка
                header = ["Интервал"] + class_names
                table_data.append(header)
                row_labels = [f"{low}-{high}" for (low, high) in intervals]
                for r, (low, high) in enumerate(intervals):
                    row = [f"{low}-{high}"]
                    for c in range(len(class_names)):
                        val = density_mat[r][c]
                        row.append(f"{val:.2f}%")
                    table_data.append(row)

                # Строим таблицу
                the_table = ax_table.table(cellText=table_data,
                                           loc='center',
                                           cellLoc='center')
                the_table.scale(1, 2)  # чуть растянуть таблицу
                ax_table.axis('off')

                pdf.savefig(fig_table)
                plt.close(fig_table)

        QMessageBox.information(self, "Сохранено", f"Отчет успешно сохранён:\n{pdf_path}")

    # Дополнительная функция для PDF-генерации таблицы плотностей
    def compute_density_matrix_for_volume(self, image_3d, mask_3d, intervals, class_names):
        """
        Аналог compute_density_matrix, но не зависит от текущих (self.current_image_3d, ...).
        Работает на переданных image_3d, mask_3d.
        """
        name_to_val = {
            "Сердце": 1,
            "Аорта": 2,
            "Легочная": 3,
            "Все": None
        }
        intensity_arrays = {}
        for cname in class_names:
            cval = name_to_val[cname]
            if cval is None:
                mask_region = (mask_3d != 0)
            else:
                mask_region = (mask_3d == cval)
            pix_vals = image_3d[mask_region]
            intensity_arrays[cname] = pix_vals

        num_rows = len(intervals)
        num_cols = len(class_names)
        result = [[0.0]*num_cols for _ in range(num_rows)]

        for c, cname in enumerate(class_names):
            pix_vals = intensity_arrays[cname]
            total_count = len(pix_vals)
            if total_count == 0:
                continue
            for r, (low, high) in enumerate(intervals):
                count_in_range = np.count_nonzero((pix_vals >= low) & (pix_vals <= high))
                percent = (count_in_range / total_count) * 100.0
                result[r][c] = percent
        return result

    # -------------------- Вспомогательные методы для PDF -----------------------
    def make_overlay_slice_single(self, image_3d, mask_3d, z):
        """
        То же, что make_overlay_slice, но без использования self.current_*
        (используем переданные image_3d, mask_3d).
        """
        h, w = image_3d.shape[:2]
        img_slice_2d = image_3d[:, :, z]
        msk_slice_2d = mask_3d[:, :, z]

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = img_slice_2d

        color_map = {
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
        }
        alpha = 0.5
        # Рисуем все классы (1,2,3)
        for cval, (cr, cg, cb) in color_map.items():
            mask_region = (msk_slice_2d == cval)
            rgb[mask_region, 0] = np.uint8((1 - alpha)*rgb[mask_region, 0] + alpha*cr)
            rgb[mask_region, 1] = np.uint8((1 - alpha)*rgb[mask_region, 1] + alpha*cg)
            rgb[mask_region, 2] = np.uint8((1 - alpha)*rgb[mask_region, 2] + alpha*cb)

        return rgb

    def make_density_map_slice_single(self, image_3d, mask_3d, z):
        h, w = image_3d.shape[:2]
        img_slice_2d = image_3d[:, :, z]
        msk_slice_2d = mask_3d[:, :, z]

        density_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        color_map_array = np.zeros((256, 3), dtype=np.uint8)
        for val in range(256):
            frac = val / 255.0
            r = 255
            g = int(255 * (1.0 - frac))
            b = 0
            color_map_array[val] = (r, g, b)

        for i in range(h):
            for j in range(w):
                intensity = img_slice_2d[i, j]
                if msk_slice_2d[i, j] == 0:
                    density_rgb[i, j] = (intensity, intensity, intensity)
                else:
                    density_rgb[i, j] = color_map_array[intensity]
        return density_rgb

    def generate_stats_text(self, image_3d, mask_3d, pid):
        lines = [f"Файл: {pid}"]
        for class_val, class_name in zip(self.class_indices, self.class_names):
            mask_region = (mask_3d == class_val)
            values = image_3d[mask_region]
            if values.size > 0:
                mean_val = values.mean()
                median_val = np.median(values)
                lines.append(f"{class_name}: mean={mean_val:.2f}, median={median_val:.2f}")
            else:
                lines.append(f"{class_name}: не сегментировано")
        return "\n".join(lines)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
