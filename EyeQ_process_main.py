#!/usr/bin/env python3
import os
import shutil
import argparse
import pandas as pd
import cv2
from PIL import ImageFile
from sklearn.model_selection import train_test_split
import fundus_prep as prep

ImageFile.LOAD_TRUNCATED_IMAGES = True


def process(image_list, base_dir, save_dir, out_size=512):
    """
    Process images listed in `image_list` found under {base_dir}/Images/
    and save to `save_dir`. Returns list of successfully processed JPG names.
    """
    processed_names = []
    radius_list, centre_list_w, centre_list_h = [], [], []

    images_root = os.path.join(base_dir, "Images")

    for image_path in image_list:
        src_path = os.path.join(images_root, image_path)
        dst_name = os.path.splitext(image_path)[0] + ".jpg"
        dst_path = os.path.join(save_dir, dst_name)

        # Skip if already processed
        if os.path.exists(dst_path):
            print(f"[skip] already exists: {dst_name}")
            processed_names.append(dst_name)
            continue

        try:
            img = prep.imread(src_path)

            # process_without_gb returns: r_img, borders, mask, r_img2, radius_list, centre_w, centre_h
            # Keep the first returned processed image
            r_img, borders, mask, _, radius_list, centre_list_w, centre_list_h = prep.process_without_gb(
                img, img, radius_list, centre_list_w, centre_list_h
            )

            r_img = cv2.resize(r_img, (out_size, out_size), interpolation=cv2.INTER_AREA)
            prep.imwrite(dst_path, r_img)
            processed_names.append(dst_name)
            print(f"[ok] {dst_name}")

        except Exception as e:
            print(f"[fail] {image_path}: {e}")

    return processed_names


def main():
    parser = argparse.ArgumentParser(description="Preprocess fundus images and split metadata by patient.")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to AUTOMORPH_DATA directory (expects subfolders/files: Images/, metadata.csv).",
    )
    parser.add_argument(
        "--out_size",
        type=int,
        default=512,
        help="Output square size for resized images (default: 512).",
    )
    parser.add_argument(
        "--csv_name",
        default="metadata.csv",
        help="Metadata CSV filename inside data_dir (default: metadata.csv).",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.data_dir)
    images_dir = os.path.join(base_dir, "Images")
    save_dir = os.path.join(base_dir, "Images_preprocessed")
    os.makedirs(save_dir, exist_ok=True)

    # Clean any notebook checkpoint clutter in Images
    ckpt_dir = os.path.join(images_dir, ".ipynb_checkpoints")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    # Load metadata
    csv_path = os.path.join(base_dir, args.csv_name)
    df = pd.read_csv(csv_path)

    # Ensure names are normalized to .jpg for the processed output
    if "Image" not in df.columns:
        raise ValueError("Expected column 'Image' in metadata CSV.")

    df["Processed_Image"] = df["Image"].apply(lambda x: os.path.splitext(str(x))[0] + ".jpg")

    # Run processing
    image_list = df["Image"].astype(str).tolist()
    processed_names = process(image_list, base_dir=base_dir, save_dir=save_dir, out_size=args.out_size)

    # Keep only successfully processed rows
    df_success = df[df["Processed_Image"].isin(processed_names)].reset_index(drop=True)

    # Ensure Patient_id exists
    if "Patient_id" not in df_success.columns:
        raise ValueError("Expected column 'Patient_id' in metadata CSV after processing.")

    # Patient-level splits: 60/20/20
    patient_ids = df_success["Patient_id"].unique()
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.4, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    df_train = df_success[df_success["Patient_id"].isin(train_ids)].reset_index(drop=True)
    df_val   = df_success[df_success["Patient_id"].isin(val_ids)].reset_index(drop=True)
    df_test  = df_success[df_success["Patient_id"].isin(test_ids)].reset_index(drop=True)

    # Save splits
    df_train.to_csv(os.path.join(base_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(base_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(base_dir, "test.csv"), index=False)

    print(f"\nDone. Saved to:\n- {os.path.join(base_dir, 'train.csv')}\n- {os.path.join(base_dir, 'val.csv')}\n- {os.path.join(base_dir, 'test.csv')}")
    print(f"Processed images saved to: {save_dir}")
    print(f"Total processed: {len(processed_names)} / {len(image_list)}")


if __name__ == "__main__":
    main()
