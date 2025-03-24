import os
import numpy as np
import nibabel as nib
from PIL import Image

def convert_jpg_to_nii(input_dir):
    """
    Находит все JPG-файлы в input_dir, создает подпапку 'nii' и конвертирует каждый снимок
    в nii.gz с переименованием по схеме "patient_XXX_0000.nii.gz" (XXX – 3-значный индекс).
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