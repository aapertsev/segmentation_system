import os
import re
import uuid
import glob
import tempfile
import subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
# УТИЛИТЫ
###############################################################################

def convert_jpg_to_nii(jpg_file, outdir):
    """
    Конвертирует один JPG в .nii.gz и возвращает путь к полученному файлу.
    """
    img = Image.open(jpg_file).convert("L")
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]  # (H, W, 1)
    nifti = nib.Nifti1Image(arr, affine=np.eye(4))

    unique_id = uuid.uuid4().hex[:6]
    filename = f"patient_{unique_id}_0000.nii.gz"
    out_path = os.path.join(outdir, filename)
    nib.save(nifti, out_path)
    return out_path


def run_inference(input_dir, output_dir, dataset_id="1", config="2d",
                  device="cpu", checkpoint="checkpoint_best.pth", fold="4"):
    """
    Запускает nnUNetv2_predict.
    ВАЖНО: input_dir должен быть ПАПКОЙ, где лежат patient_XXX_0000.nii.gz.
    """
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", dataset_id,
        "-c", config,
        f"-device={device}",
        "-chk", checkpoint,
        "-f", fold
    ]
    st.write("Выполняем команду:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_nii_as_3d_uint8(path_nii):
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


def load_nii_as_3d_int(path_nii):
    nifti_img = nib.load(path_nii)
    data = nifti_img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    data_int = data.astype(np.int32)
    return data_int


def overlay_slice(img_slice_2d, mask_slice_2d, class_checkboxes):
    """
    Наложение маски на grayscale‑срез. class_checkboxes = {1: bool, 2: bool, 3: bool}
    """
    h, w = img_slice_2d.shape
    arr_norm = img_slice_2d - img_slice_2d.min()
    denom = img_slice_2d.max() - img_slice_2d.min() or 1
    arr_norm = (arr_norm / denom * 255).astype(np.uint8)

    rgb = np.stack([arr_norm]*3, axis=-1)

    color_map = {1: (255, 0, 0),
                 2: (0, 255, 0),
                 3: (0, 0, 255)}
    alpha = 0.5
    for class_val, (cr, cg, cb) in color_map.items():
        if not class_checkboxes.get(class_val, True):
            continue
        region = (mask_slice_2d == class_val)
        rgb[region, 0] = np.uint8((1 - alpha)*rgb[region, 0] + alpha*cr)
        rgb[region, 1] = np.uint8((1 - alpha)*rgb[region, 1] + alpha*cg)
        rgb[region, 2] = np.uint8((1 - alpha)*rgb[region, 2] + alpha*cb)

    return rgb


def make_density_map_slice(img_slice_2d, mask_slice_2d):
    """
    Желто-красная карта плотности внутри mask!=0, а фон = grayscale.
    """
    arr_norm = img_slice_2d - img_slice_2d.min()
    denom = img_slice_2d.max() - img_slice_2d.min() or 1
    arr_norm = (arr_norm/denom * 255).astype(np.uint8)

    h, w = arr_norm.shape
    density_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    color_map_array = np.zeros((256, 3), dtype=np.uint8)
    for val in range(256):
        frac = val/255.0
        r = 255
        g = int(255*(1.0 - frac))
        b = 0
        color_map_array[val] = (r, g, b)

    for i in range(h):
        for j in range(w):
            intensity = arr_norm[i, j]
            if mask_slice_2d[i, j] == 0:
                density_rgb[i, j] = (intensity, intensity, intensity)
            else:
                density_rgb[i, j] = color_map_array[intensity]
    return density_rgb


###############################################################################
# STREAMLIT ПРИЛОЖЕНИЕ
###############################################################################

def main():
    st.title("Сегментация КТ снимков (nnUNet) – Веб-система вместо PyQt5")

    st.markdown("""
    **Использование**:
    1) Загрузите несколько JPG или .nii.gz (поддерживаются и те, и другие).  
    2) Нажмите "Запуск предсказания" — все JPG преобразуются в patient_XXX_0000.nii.gz, создаётся папка, запускается nnUNetv2_predict.  
    3) Снизу появится список "пациентов": выберите, меняйте срез, включайте/выключайте классы, стройте карту плотностей, гистограмму, таблицу, сохраняйте PDF.  
    """)

    # Параметры на боковой панели
    device = st.sidebar.selectbox("Устройство", ["cpu", "cuda:0"])
    dataset_id = st.sidebar.text_input("Dataset ID", "1")
    config = st.sidebar.selectbox("Конфигурация", ["2d", "3d_fullres", "3d_lowres"])
    checkpoint = st.sidebar.text_input("Checkpoint", "checkpoint_best.pth")
    fold = st.sidebar.text_input("Fold", "4")

    uploaded_files = st.file_uploader(
        "Выберите JPG/nii (можно несколько)",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "nii", "nii.gz"]
    )

    if "pairs" not in st.session_state:
        st.session_state["pairs"] = {}
    if "base_tmpdir" not in st.session_state:
        st.session_state["base_tmpdir"] = tempfile.mkdtemp(prefix="nnunet_session_")

    if st.button("Запуск предсказания"):
        if not uploaded_files:
            st.warning("Сначала загрузите файлы!")
        else:
            with st.spinner("Обработка..."):
                base_dir = st.session_state["base_tmpdir"]
                nii_dir = os.path.join(base_dir, "nii")
                os.makedirs(nii_dir, exist_ok=True)
                results_dir = os.path.join(base_dir, "results")
                os.makedirs(results_dir, exist_ok=True)

                # 1) Сохраняем/конвертируем файлы в nii_dir
                for upfile in uploaded_files:
                    ext = ".nii.gz" if upfile.name.endswith(".nii.gz") else os.path.splitext(upfile.name)[1].lower()
                    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    tmpfile.write(upfile.read())
                    tmpfile.close()
                    local_path = tmpfile.name

                    if ext in [".jpg", ".jpeg"]:
                        # конверсия
                        new_nii = convert_jpg_to_nii(local_path, nii_dir)
                        st.write(f"Конвертирован {upfile.name} -> {os.path.basename(new_nii)}")
                    else:
                        # .nii / .nii.gz
                        unique_id = uuid.uuid4().hex[:6]
                        out_name = f"patient_{unique_id}_0000.nii.gz"
                        out_path = os.path.join(nii_dir, out_name)
                        os.rename(local_path, out_path)
                        st.write(f"Скопирован {upfile.name} -> {os.path.basename(out_path)}")

                # 2) Запуск nnUNet
                try:
                    run_inference(
                        input_dir=nii_dir,
                        output_dir=results_dir,
                        dataset_id=dataset_id,
                        config=config,
                        device=device,
                        checkpoint=checkpoint,
                        fold=fold
                    )
                except subprocess.CalledProcessError as e:
                    st.error(f"nnUNetv2_predict завершился с ошибкой: {e}")
                    st.stop()

                st.success("Инференс завершён!")

                # 3) Ищем пары (patient_XXX_0000.nii.gz vs patient_XXX.nii.gz)
                st.session_state["pairs"].clear()

                nii_files = sorted(glob.glob(os.path.join(nii_dir, "patient_*_0000.nii.gz")))
                for nf in nii_files:
                    base = os.path.basename(nf)
                    # pattern: "patient_abcdef_0000.nii.gz"
                    m = re.match(r'^(patient_[0-9a-f]+)_0000\.nii\.gz$', base)
                    if m:
                        pid = m.group(1)
                        mask_name = f"{pid}.nii.gz"
                        mask_path = os.path.join(results_dir, mask_name)
                        if os.path.exists(mask_path):
                            st.session_state["pairs"][pid] = (nf, mask_path)
                        else:
                            st.write(f"Не найдена маска для {pid}: {mask_path}")
                    else:
                        st.write(f"Файл {base} не соответствует шаблону patient_XXX_0000.nii.gz")

                if not st.session_state["pairs"]:
                    st.warning("Похоже, nnUNet не создал соответствующие маски. Проверьте логи.")
                else:
                    st.info("Пары (исходник + маска) собраны! Ниже визуализация.")

    st.subheader("Визуализация результатов")
    pairs_dict = st.session_state["pairs"]
    if pairs_dict:
        pid_list = sorted(pairs_dict.keys())
        selected_pid = st.selectbox("Выберите пациента:", pid_list)

        if selected_pid:
            img_path, msk_path = pairs_dict[selected_pid]
            st.write(f"Исходник: {os.path.basename(img_path)}")
            st.write(f"Маска: {os.path.basename(msk_path)}")

            image_3d = load_nii_as_3d_uint8(img_path)
            mask_3d = load_nii_as_3d_int(msk_path)

            depth = image_3d.shape[2]
            if depth > 1:
                max_slice = depth - 1
                default_value = max_slice // 2
                z_slice = st.slider("Срез Z", 0, max_slice, default_value)
            else:
                st.info("Объём содержит только один срез (Z=0).")
                z_slice = 0

            col1, col2, col3 = st.columns(3)
            class1 = col1.checkbox("Сердце (class=1)", True)
            class2 = col2.checkbox("Аорта (class=2)", True)
            class3 = col3.checkbox("Легочная (class=3)", True)
            class_checkboxes = {1: class1, 2: class2, 3: class3}

            slice_img = image_3d[:, :, z_slice]
            slice_msk = mask_3d[:, :, z_slice]
            overlay_rgb = overlay_slice(slice_img, slice_msk, class_checkboxes)
            st.image(overlay_rgb, caption=f"Наложение (Z={z_slice})", use_column_width=True)

            if st.button("Построить карту плотности"):
                density = make_density_map_slice(slice_img, slice_msk)
                st.image(density, caption="Карта плотности", use_column_width=True)

            if st.button("Построить гистограмму"):
                region = (slice_msk != 0)
                vals = slice_img[region]
                fig, ax = plt.subplots()
                ax.hist(vals.flatten(), bins=50, color='gray')
                ax.set_title("Гистограмма (mask!=0)")
                st.pyplot(fig)

            if st.button("Таблица плотностей"):
                table_html = build_density_table(image_3d, mask_3d,
                                                 intervals=[(0,50),(51,100),(101,150),(151,200)],
                                                 class_names=["Сердце", "Аорта", "Легочная Артерия", "Все"])
                st.markdown(table_html, unsafe_allow_html=True)

            if st.button("Сохранить отчет (PDF)"):
                pdf_data = save_report_to_pdf(selected_pid, pairs_dict,
                                              intervals=[(0,50),(51,100),(101,150),(151,200),(201,255)])
                st.download_button("Скачать PDF", data=pdf_data, file_name=f"report_{selected_pid}.pdf")
    else:
        st.write("Нет данных для отображения.")

def build_density_table(image_3d, mask_3d, intervals, class_names):
    name_to_val = {"Сердце":1, "Аорта":2, "Легочная Артерия":3, "Все":None}
    intensity_arrays = {}
    for cname in class_names:
        cval = name_to_val[cname]
        if cval is None:
            mask_region = (mask_3d != 0)
        else:
            mask_region = (mask_3d == cval)
        pix_vals = image_3d[mask_region]
        intensity_arrays[cname] = pix_vals

    rows = []
    header = ["Интервал"] + class_names
    rows.append(header)

    for (low, high) in intervals:
        row = [f"{low}-{high}"]
        for cname in class_names:
            pix_vals = intensity_arrays[cname]
            total_count = len(pix_vals)
            if total_count==0:
                row.append("0.00%")
            else:
                cnt = np.count_nonzero((pix_vals>=low)&(pix_vals<=high))
                pc = (cnt/total_count)*100.0
                row.append(f"{pc:.2f}%")
        rows.append(row)

    html = "<table border='1' style='border-collapse: collapse;'>"
    for i, rowvals in enumerate(rows):
        html += "<tr>"
        for val in rowvals:
            if i==0:
                html += f"<th style='padding:4px'>{val}</th>"
            else:
                html += f"<td style='padding:4px'>{val}</td>"
        html += "</tr>"
    html+="</table>"
    return html

def save_report_to_pdf(pid, pairs_dict, intervals):
    import io
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig_title = plt.figure(figsize=(8,4))
        plt.text(0.5, 0.5, f"Отчет по {pid}", ha='center', va='center', fontsize=16)
        plt.axis('off')
        pdf.savefig(fig_title)
        plt.close(fig_title)

        img_path, msk_path = pairs_dict[pid]
        image_3d = load_nii_as_3d_uint8(img_path)
        mask_3d = load_nii_as_3d_int(msk_path)

        z_central = mask_3d.shape[2]//2
        overlay_rgb = overlay_slice(image_3d[:,:,z_central], mask_3d[:,:,z_central],
                                    {1:True,2:True,3:True})
        density_rgb = make_density_map_slice(image_3d[:,:,z_central], mask_3d[:,:,z_central])

        fig = plt.figure(figsize=(10,8))
        fig.suptitle(f"Файл: {pid}", fontsize=14)
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title("Исходник+маска (центр)")
        ax1.imshow(overlay_rgb)
        ax1.axis('off')

        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title("Карта плотности (центр)")
        ax2.imshow(density_rgb)
        ax2.axis('off')

        ax3 = fig.add_subplot(2,2,3)
        ax3.set_title("Гистограмма(mask!=0)")
        region = (mask_3d!=0)
        vals = image_3d[region]
        ax3.hist(vals.flatten(), bins=50, color='gray')
        ax3.set_xlabel("Intensity")
        ax3.set_ylabel("Count")

        ax4 = fig.add_subplot(2,2,4)
        ax4.set_title("Статистика")
        txt = generate_stats_text(image_3d, mask_3d, pid)
        ax4.text(0.05,0.95,txt,fontsize=10,va='top',ha='left')
        ax4.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

    data = buf.getvalue()
    buf.close()
    return data

def generate_stats_text(image_3d, mask_3d, pid):
    lines = [f"Файл: {pid}"]
    classes = [(1,"Сердце"), (2,"Аорта"), (3,"Легочная Артерия")]
    for cval, cname in classes:
        region = (mask_3d==cval)
        vals = image_3d[region]
        if vals.size>0:
            mean_v = vals.mean()
            median_v = np.median(vals)
            lines.append(f"{cname}: mean={mean_v:.2f}, median={median_v:.2f}")
        else:
            lines.append(f"{cname}: нет вокселей")
    return "\n".join(lines)

if __name__=="__main__":
    main()
