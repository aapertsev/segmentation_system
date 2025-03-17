import sys
import os
import re
import numpy as np
import nibabel as nib

import matplotlib
# Уберите или закомментируйте следующую строку, чтобы использовать интерактивный бэкенд:
# matplotlib.use("Agg")

# Если хотите явно указать Qt5Agg:
# matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QPlainTextEdit, QSlider, QListWidget, QListWidgetItem,
    QCheckBox, QLabel, QDialog, QVBoxLayout
)


class MainWindow(QMainWindow):
    """
    Главное окно (как в предыдущем примере):
      - Кнопка «Запуск предсказания» -> QProcess (nnUNetv2_predict)
      - Кнопка «Визуализация» -> открывает VisualizationWindow
      ...
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сегментация сердца и аорты на снимках КТ")
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

        # Процесс для инференса
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.finished.connect(self.on_finished)

        self.inference_input_dir = None
        self.inference_output_dir = None

    def run_inference(self):
        """
        Запускаем nnUNetv2_predict, вывод лога в QPlainTextEdit.
        """
        input_dir = QFileDialog.getExistingDirectory(
            self,
            "Выберите директорию с входными данными"
        )
        if not input_dir:
            return
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Выберите директорию для сохранения результатов"
        )
        if not output_dir:
            return

        self.inference_input_dir = input_dir
        self.inference_output_dir = output_dir

        self.log_text.clear()

        command = [
            "nnUNetv2_predict",
            "-i", input_dir,
            "-o", output_dir,
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
            QMessageBox.information(
                self, "Готово!", "Предсказание завершено."
            )
            self.open_visualization_automatically()
        else:
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Процесс завершился с ошибкой (код {exit_code})."
            )

    def open_visualization_automatically(self):
        if not self.inference_input_dir or not self.inference_output_dir:
            return
        self.vis_window = VisualizationWindow(
            self.inference_input_dir,
            self.inference_output_dir
        )
        self.vis_window.show()

    def open_visualization(self):
        """
        Вручную выбираем папку с исходниками (images_dir) и папку с масками (masks_dir).
        """
        images_dir = QFileDialog.getExistingDirectory(
            self,
            "Выберите директорию с исходными изображениями"
        )
        if not images_dir:
            return
        masks_dir = QFileDialog.getExistingDirectory(
            self,
            "Выберите директорию с масками"
        )
        if not masks_dir:
            return

        self.vis_window = VisualizationWindow(images_dir, masks_dir)
        self.vis_window.show()


class VisualizationWindow(QWidget):
    """
    Окно визуализации с дополнительной кнопкой "Сохранить отчет (PDF)".
    """
    def __init__(self, images_dir, masks_dir, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.setWindowTitle("Визуализация")

        self.pairs = {}
        self.current_patient_id = None
        self.current_image_3d = None
        self.current_mask_3_d = None
        self.current_slice = 0
        self.num_slices = 1

        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self.on_patient_selected)

        # 3 класса: 1=Сердце, 2=Аорта, 3=Легочная Артерия
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

        # НОВАЯ кнопка "Сохранить отчет (PDF)"
        self.save_report_button = QPushButton("Сохранить отчет (PDF)")
        self.save_report_button.clicked.connect(self.save_report_to_pdf)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.list_widget)
        for cb in self.class_checkboxes:
            left_layout.addWidget(cb)
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

    # --------------------------------------------------------------------------
    # Логика загрузки/отображения (как и раньше)
    # --------------------------------------------------------------------------
    def find_pairs(self):
        pattern_img = re.compile(r'^(.*)_0000\.nii(\.gz)?$', re.IGNORECASE)
        pattern_msk = re.compile(r'^(.*)\.nii(\.gz)?$', re.IGNORECASE)

        images_map = {}
        for f in os.listdir(self.images_dir):
            m = pattern_img.match(f)
            if m:
                pid = m.group(1)
                images_map[pid] = os.path.join(self.images_dir, f)

        masks_map = {}
        for f in os.listdir(self.masks_dir):
            m = pattern_msk.match(f)
            if m:
                pid = m.group(1)
                masks_map[pid] = os.path.join(self.masks_dir, f)

        for pid in images_map:
            if pid in masks_map:
                self.pairs[pid] = (images_map[pid], masks_map[pid])

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
            self.slice_slider.setRange(0, self.num_slices - 1)
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
        # Превращаем numpy -> QImage
        h, w, _ = overlay_rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(overlay_rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    # --------------------------------------------------------------------------
    # Методы для построения (карта плотности, гистограмма)
    # --------------------------------------------------------------------------
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
        plt.hist(values.flatten(), bins=50)
        plt.title("Гистограмма плотности")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.show()  # Показываем интерактивное окно

    # --------------------------------------------------------------------------
    # Вспомогательные методы
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Сохранение отчёта в PDF (как ранее)
    # --------------------------------------------------------------------------
    def save_report_to_pdf(self):
        save_dir = QFileDialog.getExistingDirectory(self, "Выберите директорию, куда сохранить отчет PDF")
        if not save_dir:
            return

        pdf_name = f"report_{os.path.basename(self.images_dir)}.pdf"
        pdf_path = os.path.join(save_dir, pdf_name)

        with PdfPages(pdf_path) as pdf:
            # 1) Титульная страница
            fig_title = plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5,
                     f"Отчет по {os.path.basename(self.images_dir)}",
                     ha='center', va='center', fontsize=16)
            plt.axis('off')
            pdf.savefig(fig_title)
            plt.close(fig_title)

            # 2) Каждому patientID -> страница
            for pid in sorted(self.pairs.keys()):
                img_path, msk_path = self.pairs[pid]
                image_3d = self.load_nii_as_3d_uint8(img_path)
                mask_3d = self.load_nii_as_3d_int(msk_path)
                z_central = mask_3d.shape[2] // 2

                overlay_rgb = self.make_overlay_slice_single(image_3d, mask_3d, z_central)
                density_rgb = self.make_density_map_slice_single(image_3d, mask_3d, z_central)
                stats_text = self.generate_stats_text(image_3d, mask_3d, pid)

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
                ax4.text(0.05, 0.95, stats_text, fontsize=10,
                         va='top', ha='left')
                ax4.axis('off')

                pdf.savefig(fig)
                plt.close(fig)

        QMessageBox.information(self, "Сохранено", f"Отчет успешно сохранён:\n{pdf_path}")


    def make_overlay_slice_single(self, image_3d, mask_3d, z):
        img_slice_2d = image_3d[:, :, z]
        msk_slice_2d = mask_3d[:, :, z]

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
        for class_val in [1, 2, 3]:
            c_r, c_g, c_b = color_map[class_val]
            mask_region = (msk_slice_2d == class_val)
            rgb[mask_region, 0] = np.uint8((1 - alpha)*rgb[mask_region, 0] + alpha*c_r)
            rgb[mask_region, 1] = np.uint8((1 - alpha)*rgb[mask_region, 1] + alpha*c_g)
            rgb[mask_region, 2] = np.uint8((1 - alpha)*rgb[mask_region, 2] + alpha*c_b)
        return rgb

    def make_density_map_slice_single(self, image_3d, mask_3d, z):
        img_slice_2d = image_3d[:, :, z]
        msk_slice_2d = mask_3d[:, :, z]

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
