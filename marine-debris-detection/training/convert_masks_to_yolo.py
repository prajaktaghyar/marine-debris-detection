import shutil
import random
from pathlib import Path

import cv2


# ==========================================================
# PATHS
# ==========================================================

TRAINING_DIR = Path(__file__).resolve().parent
BASE_DIR = TRAINING_DIR.parent

# Source datasets are inside ROOT dataset folder
SOURCE_DATASET = BASE_DIR / "dataset"

MARINE_DATASET = SOURCE_DATASET / "marine-debris-watertank-release"
SHIPWRECK_DATASET = SOURCE_DATASET / "AI4Shipwrecks"

# Final YOLO dataset
FINAL_DATASET = BASE_DIR / "dataset"

IMAGES_DIR = FINAL_DATASET / "images"
LABELS_DIR = FINAL_DATASET / "labels"

TRAIN_IMAGES = IMAGES_DIR / "train"
VAL_IMAGES = IMAGES_DIR / "val"
TEST_IMAGES = IMAGES_DIR / "test"

TRAIN_LABELS = LABELS_DIR / "train"
VAL_LABELS = LABELS_DIR / "val"
TEST_LABELS = LABELS_DIR / "test"

DATASET_YAML = FINAL_DATASET / "dataset.yaml"


# ==========================================================
# CLASSES
# ==========================================================

MARINE_CLASSES = [
    "bottle",
    "can",
    "chain",
    "drink-carton",
    "hook",
    "propeller",
    "shampoo-bottle",
    "standing-bottle",
    "tire",
    "valve",
]

ALL_CLASSES = MARINE_CLASSES + ["shipwreck"]

SHIPWRECK_CLASS_ID = 10


# ==========================================================
# SETTINGS
# ==========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

LABEL_EXTENSIONS = [
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
]

MIN_AREA = 50

RANDOM_SEED = 42

MARINE_VAL_RATIO = 0.20

MAX_SHIPWRECK_VAL = None


# ==========================================================
# HEADER
# ==========================================================

def print_header():

    print()
    print("=" * 70)
    print("     MARINE DEBRIS + SHIPWRECK YOLO DATASET BUILDER")
    print("=" * 70)
    print()


# ==========================================================
# BASIC HELPERS
# ==========================================================

def is_image_file(path):

    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_images(folder):

    folder = Path(folder)

    if not folder.exists():
        return []

    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and is_image_file(p)
        ]
    )


# ==========================================================
# CHECK SOURCE DATASETS
# ==========================================================

def check_sources():

    print("[1/7] Checking source datasets...")
    print()

    if not SOURCE_DATASET.exists():

        print("[ERROR] Dataset directory does not exist:")
        print(SOURCE_DATASET)
        return False

    marine_found = MARINE_DATASET.exists()
    shipwreck_found = SHIPWRECK_DATASET.exists()

    print("Marine Debris:")
    print(f"  {MARINE_DATASET}")

    if marine_found:
        print("  [FOUND]")
    else:
        print("  [NOT FOUND]")

    print()

    print("AI4Shipwrecks:")
    print(f"  {SHIPWRECK_DATASET}")

    if shipwreck_found:
        print("  [FOUND]")
    else:
        print("  [NOT FOUND]")

    print()

    if not marine_found and not shipwreck_found:

        print("[ERROR] No source dataset found.")
        print()
        print("Available folders:")

        for item in SOURCE_DATASET.iterdir():

            if item.is_dir():
                print(f"  - {item.name}")

        return False

    return True


# ==========================================================
# PREPARE FINAL DATASET
# ==========================================================

def clean_final_dataset():

    print("[2/7] Preparing final YOLO dataset...")
    print()

    # IMPORTANT:
    # Only generated folders are deleted.
    #
    # The following source folders are NOT deleted:
    #
    # dataset/AI4Shipwrecks
    # dataset/marine-debris-watertank-release
    #
    generated_directories = [
        TRAIN_IMAGES,
        VAL_IMAGES,
        TEST_IMAGES,
        TRAIN_LABELS,
        VAL_LABELS,
        TEST_LABELS,
    ]

    for directory in generated_directories:

        if directory.exists():
            shutil.rmtree(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("[OK] Final YOLO folders created.")
    print()


# ==========================================================
# FIND MARINE CLASS FOLDER
# ==========================================================

def find_marine_class_folder(class_name):

    if not MARINE_DATASET.exists():
        return None

    possible_names = {
        class_name.lower(),
        class_name.replace("-", "_").lower(),
        class_name.replace("-", " ").lower(),
    }

    common_paths = [

        MARINE_DATASET / class_name,

        MARINE_DATASET / "class-image-crops" / class_name,

        MARINE_DATASET / "class_image_crops" / class_name,

        MARINE_DATASET / "images" / class_name,

        MARINE_DATASET / "train" / class_name,

    ]

    for path in common_paths:

        if path.exists() and path.is_dir():

            if list_images(path):
                return path

    # Recursive search
    for path in MARINE_DATASET.rglob("*"):

        if not path.is_dir():
            continue

        if path.name.lower() not in possible_names:
            continue

        if list_images(path):
            return path

    return None


# ==========================================================
# FIND MATCHING ANNOTATION
# ==========================================================

def find_matching_annotation(
    image_path,
    search_roots,
):
    """
    Searches for an annotation belonging to an image.

    Supports:
        YOLO .txt
        mask .png
        mask .jpg
        mask .jpeg
        mask .bmp
        mask .tif
        mask .tiff
    """

    image_path = Path(image_path)

    stem = image_path.stem.lower()

    for root in search_roots:

        if root is None:
            continue

        root = Path(root)

        if not root.exists():
            continue

        # --------------------------------------------------
        # Direct matching
        # --------------------------------------------------

        for extension in LABEL_EXTENSIONS:

            candidate = root / f"{image_path.stem}{extension}"

            if candidate.exists():
                return candidate

        # --------------------------------------------------
        # Recursive matching
        # --------------------------------------------------

        for candidate in root.rglob("*"):

            if not candidate.is_file():
                continue

            if candidate.stem.lower() == stem:

                if candidate.suffix.lower() in LABEL_EXTENSIONS:

                    return candidate

    return None


# ==========================================================
# COPY IMAGE
# ==========================================================

def copy_image(
    source_image,
    destination_folder,
    prefix,
    index,
):

    source_image = Path(source_image)

    filename = (
        f"{prefix}_{index:06d}"
        f"{source_image.suffix.lower()}"
    )

    destination = destination_folder / filename

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_image,
        destination,
    )

    return destination


# ==========================================================
# READ YOLO LABEL
# ==========================================================

def read_yolo_label(
    source_label,
    destination_label,
    target_class_id,
):
    """
    Read an existing YOLO annotation.

    Original class IDs are ignored because this converter
    combines different datasets.

    Every valid object becomes target_class_id.

    This is used for marine debris and shipwreck sources
    when existing YOLO .txt annotations are available.
    """

    try:

        with open(
            source_label,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            lines = f.readlines()

    except Exception:

        return False

    output_lines = []

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:

            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

        except ValueError:

            continue

        # Validate YOLO coordinates
        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < bw <= 1
            and 0 < bh <= 1
        ):
            continue

        output_lines.append(
            f"{target_class_id} "
            f"{xc:.6f} "
            f"{yc:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f}\n"
        )

    if not output_lines:
        return False

    with open(
        destination_label,
        "w",
        encoding="utf-8",
    ) as f:

        f.writelines(output_lines)

    return True


# ==========================================================
# MASK TO YOLO
# ==========================================================

def mask_to_yolo(
    mask_path,
    label_path,
    class_id,
):
    """
    Convert image mask to YOLO bounding boxes.
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        return False

    height, width = mask.shape[:2]

    if width <= 0 or height <= 0:
        return False

    # Convert mask to binary
    _, binary = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w <= 0 or h <= 0:
            continue

        xc = (x + w / 2) / width
        yc = (y + h / 2) / height

        bw = w / width
        bh = h / height

        boxes.append(
            (
                class_id,
                xc,
                yc,
                bw,
                bh,
            )
        )

    if not boxes:
        return False

    with open(
        label_path,
        "w",
        encoding="utf-8",
    ) as f:

        for (
            cls,
            xc,
            yc,
            bw,
            bh,
        ) in boxes:

            f.write(
                f"{cls} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}\n"
            )

    return True


# ==========================================================
# FULL IMAGE FALLBACK
# ==========================================================

def create_full_image_label(
    image_path,
    label_path,
    class_id,
):
    """
    Fallback for class-image-crops.

    IMPORTANT:
    This is only used when no annotation exists.

    Since the source is a class-specific object crop,
    the object is assumed to fill the image.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return False

    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        return False

    with open(
        label_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{class_id} "
            f"0.5 0.5 1.0 1.0\n"
        )

    return True


# ==========================================================
# CREATE LABEL FROM ANNOTATION
# ==========================================================

def create_label_from_annotation(
    image_path,
    annotation,
    destination_label,
    class_id,
):
    """
    Decide whether annotation is:

        .txt -> YOLO
        image -> mask
    """

    if annotation is None:
        return False

    suffix = annotation.suffix.lower()

    # Existing YOLO annotation
    if suffix == ".txt":

        return read_yolo_label(
            annotation,
            destination_label,
            class_id,
        )

    # Image mask
    if suffix in IMAGE_EXTENSIONS:

        return mask_to_yolo(
            annotation,
            destination_label,
            class_id,
        )

    return False


# ==========================================================
# PROCESS ONE MARINE IMAGE
# ==========================================================

def process_marine_image(
    source_image,
    class_id,
    class_name,
    destination_folder,
    label_folder,
    index,
):
    """
    Create training example.

    Priority:

    1. Existing YOLO annotation
    2. Existing mask
    3. Full-image crop annotation
    """

    destination_image = copy_image(
        source_image,
        destination_folder,
        f"marine_{class_name}",
        index,
    )

    destination_label = (
        label_folder
        / f"{destination_image.stem}.txt"
    )

    # Search for annotations around the image
    annotation = find_matching_annotation(
        source_image,
        [
            source_image.parent,
            source_image.parent.parent,
            MARINE_DATASET,
            MARINE_DATASET / "labels",
            MARINE_DATASET / "annotations",
            MARINE_DATASET / "masks",
        ],
    )

    # ------------------------------------------------------
    # CASE 1: Real annotation found
    # ------------------------------------------------------

    if annotation is not None:

        success = create_label_from_annotation(
            destination_image,
            annotation,
            destination_label,
            class_id,
        )

        if success:

            return True, "annotation"

    # ------------------------------------------------------
    # CASE 2: No annotation
    #
    # Class-image-crops dataset:
    # use complete image as object box.
    # ------------------------------------------------------

    success = create_full_image_label(
        destination_image,
        destination_label,
        class_id,
    )

    if success:

        return True, "crop"

    # Failed
    if destination_image.exists():
        destination_image.unlink()

    if destination_label.exists():
        destination_label.unlink()

    return False, "failed"


# ==========================================================
# PROCESS MARINE DEBRIS
# ==========================================================

def process_marine_debris():

    print("[3/7] Processing marine debris dataset...")
    print()

    if not MARINE_DATASET.exists():

        print(
            "[WARNING] Marine debris dataset not found:"
        )

        print(
            MARINE_DATASET
        )

        print()

        return 0

    random.seed(RANDOM_SEED)

    total = 0
    train_count = 0
    val_count = 0

    annotation_count = 0
    crop_count = 0

    for class_id, class_name in enumerate(
        MARINE_CLASSES
    ):

        class_folder = find_marine_class_folder(
            class_name
        )

        if class_folder is None:

            print(
                f"[WARNING] Class folder not found: "
                f"{class_name}"
            )

            continue

        images = list_images(
            class_folder
        )

        if not images:

            print(
                f"[WARNING] No images found: "
                f"{class_name}"
            )

            continue

        random.shuffle(images)

        split_index = int(
            len(images)
            * (1 - MARINE_VAL_RATIO)
        )

        if len(images) > 1:

            split_index = max(
                1,
                min(
                    split_index,
                    len(images) - 1,
                ),
            )

        else:

            split_index = len(images)

        train_images = images[:split_index]
        val_images = images[split_index:]

        print(
            f"{class_name:20s} -> "
            f"{len(images):5d} images | "
            f"train={len(train_images):5d} | "
            f"val={len(val_images):5d}"
        )

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------

        for index, source_image in enumerate(
            train_images
        ):

            success, annotation_type = process_marine_image(
                source_image,
                class_id,
                class_name,
                TRAIN_IMAGES,
                TRAIN_LABELS,
                index,
            )

            if success:

                train_count += 1
                total += 1

                if annotation_type == "annotation":
                    annotation_count += 1
                else:
                    crop_count += 1

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        for index, source_image in enumerate(
            val_images
        ):

            success, annotation_type = process_marine_image(
                source_image,
                class_id,
                class_name,
                VAL_IMAGES,
                VAL_LABELS,
                index,
            )

            if success:

                val_count += 1
                total += 1

                if annotation_type == "annotation":
                    annotation_count += 1
                else:
                    crop_count += 1

    print()
    print(
        f"[OK] Marine debris processed: {total}"
    )

    print(
        f"     Training:       {train_count}"
    )

    print(
        f"     Validation:     {val_count}"
    )

    print(
        f"     Real annotations: {annotation_count}"
    )

    print(
        f"     Crop fallback:    {crop_count}"
    )

    print()

    return total


# ==========================================================
# FIND SHIPWRECK SPLIT
# ==========================================================

def find_shipwreck_split(split_names):

    if not SHIPWRECK_DATASET.exists():
        return None, None

    split_names = {
        name.lower()
        for name in split_names
    }

    candidates = []

    # Search for:
    #
    # train/images
    # train/labels
    #
    # test/images
    # test/labels

    for images_dir in SHIPWRECK_DATASET.rglob("*"):

        if not images_dir.is_dir():
            continue

        if images_dir.name.lower() != "images":
            continue

        parent = images_dir.parent

        if parent.name.lower() not in split_names:
            continue

        images = list_images(
            images_dir
        )

        if not images:
            continue

        labels_dir = None

        for label_name in [
            "labels",
            "masks",
            "annotations",
        ]:

            candidate = parent / label_name

            if candidate.exists():

                labels_dir = candidate
                break

        candidates.append(
            (
                images_dir,
                labels_dir,
                len(images),
            )
        )

    if not candidates:

        return None, None

    candidates.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    return (
        candidates[0][0],
        candidates[0][1],
    )


# ==========================================================
# PROCESS SHIPWRECK IMAGE
# ==========================================================

def process_shipwreck_image(
    source_image,
    source_labels,
    destination_folder,
    destination_labels,
    index,
):
    """
    Shipwreck must have a real annotation/mask.

    We do NOT blindly use the complete image as the
    shipwreck bounding box.
    """

    destination_image = copy_image(
        source_image,
        destination_folder,
        "shipwreck",
        index,
    )

    destination_label = (
        destination_labels
        / f"{destination_image.stem}.txt"
    )

    annotation = find_matching_annotation(
        source_image,
        [
            source_labels,
            source_image.parent,
            SHIPWRECK_DATASET,
        ],
    )

    success = create_label_from_annotation(
        destination_image,
        annotation,
        destination_label,
        SHIPWRECK_CLASS_ID,
    )

    if success:

        return True

    # Remove image because annotation is missing
    if destination_image.exists():
        destination_image.unlink()

    if destination_label.exists():
        destination_label.unlink()

    return False


# ==========================================================
# PROCESS SHIPWRECK
# ==========================================================

def process_shipwreck():

    print("[4/7] Processing AI4Shipwrecks dataset...")
    print()

    if not SHIPWRECK_DATASET.exists():

        print(
            "[WARNING] AI4Shipwrecks dataset not found:"
        )

        print(
            SHIPWRECK_DATASET
        )

        print()

        return 0

    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    train_images_dir, train_labels_dir = (
        find_shipwreck_split(
            [
                "train",
                "training",
            ]
        )
    )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    val_images_dir, val_labels_dir = (
        find_shipwreck_split(
            [
                "val",
                "valid",
                "validation",
                "test",
                "testing",
            ]
        )
    )

    if train_images_dir:

        print(
            "[FOUND] Shipwreck training images:"
        )

        print(
            f"  {train_images_dir}"
        )

        if train_labels_dir:

            print(
                "[FOUND] Shipwreck training annotations:"
            )

            print(
                f"  {train_labels_dir}"
            )

        else:

            print(
                "[WARNING] Training annotation folder "
                "not found."
            )

    else:

        print(
            "[WARNING] Shipwreck training images "
            "not found."
        )

    print()

    if val_images_dir:

        print(
            "[FOUND] Shipwreck validation/test images:"
        )

        print(
            f"  {val_images_dir}"
        )

        if val_labels_dir:

            print(
                "[FOUND] Shipwreck validation annotations:"
            )

            print(
                f"  {val_labels_dir}"
            )

        else:

            print(
                "[WARNING] Validation annotation folder "
                "not found."
            )

    else:

        print(
            "[WARNING] Shipwreck validation/test images "
            "not found."
        )

    print()

    train_count = 0
    val_count = 0

    # ======================================================
    # TRAINING
    # ======================================================

    if train_images_dir:

        images = list_images(
            train_images_dir
        )

        print(
            f"Shipwreck training images: "
            f"{len(images)}"
        )

        for index, source_image in enumerate(
            images
        ):

            success = process_shipwreck_image(
                source_image,
                train_labels_dir,
                TRAIN_IMAGES,
                TRAIN_LABELS,
                index,
            )

            if success:
                train_count += 1

    # ======================================================
    # VALIDATION
    # ======================================================

    if val_images_dir:

        images = list_images(
            val_images_dir
        )

        if MAX_SHIPWRECK_VAL is not None:

            images = images[
                :MAX_SHIPWRECK_VAL
            ]

        print(
            f"Shipwreck validation images: "
            f"{len(images)}"
        )

        for index, source_image in enumerate(
            images
        ):

            success = process_shipwreck_image(
                source_image,
                val_labels_dir,
                VAL_IMAGES,
                VAL_LABELS,
                index,
            )

            if success:
                val_count += 1

    print()

    print(
        "[OK] Shipwreck processed:"
    )

    print(
        f"     Training:   {train_count}"
    )

    print(
        f"     Validation: {val_count}"
    )

    print()

    return train_count + val_count


# ==========================================================
# CREATE DATASET YAML
# ==========================================================

def write_dataset_yaml():

    print("[5/7] Creating dataset.yaml...")
    print()

    lines = [
        f"path: {FINAL_DATASET.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(ALL_CLASSES)}",
        "names:",
    ]

    for class_id, class_name in enumerate(
        ALL_CLASSES
    ):

        lines.append(
            f"  {class_id}: {class_name}"
        )

    with open(
        DATASET_YAML,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
            + "\n"
        )

    print(
        "[OK] Dataset YAML created:"
    )

    print(
        DATASET_YAML
    )

    print()


# ==========================================================
# VERIFY LABEL
# ==========================================================

def verify_label_file(label_file):

    try:

        with open(
            label_file,
            "r",
            encoding="utf-8",
        ) as f:

            lines = f.readlines()

    except Exception:

        return False

    if not lines:
        return False

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            return False

        try:

            class_id = int(parts[0])

            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

        except ValueError:

            return False

        if not (
            0 <= class_id < len(ALL_CLASSES)
        ):
            return False

        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < bw <= 1
            and 0 < bh <= 1
        ):
            return False

    return True


# ==========================================================
# VERIFY DATASET
# ==========================================================

def verify_dataset():

    print("[6/7] Verifying final YOLO dataset...")
    print()

    splits = [
        (
            "TRAIN",
            TRAIN_IMAGES,
            TRAIN_LABELS,
        ),
        (
            "VAL",
            VAL_IMAGES,
            VAL_LABELS,
        ),
        (
            "TEST",
            TEST_IMAGES,
            TEST_LABELS,
        ),
    ]

    total_images = 0
    total_labels = 0

    errors = 0

    for name, image_dir, label_dir in splits:

        images = list_images(
            image_dir
        )

        labels = list(
            label_dir.glob("*.txt")
        )

        print(
            f"{name:5s} | "
            f"Images: {len(images):6d} | "
            f"Labels: {len(labels):6d}"
        )

        total_images += len(images)
        total_labels += len(labels)

        # Train and validation are mandatory
        if name in {
            "TRAIN",
            "VAL",
        }:

            if len(images) == 0:

                print(
                    f"  [ERROR] {name} has no images."
                )

                errors += 1

            if len(labels) == 0:

                print(
                    f"  [ERROR] {name} has no labels."
                )

                errors += 1

        # Check image-label correspondence
        label_names = {
            label.stem
            for label in labels
        }

        missing_labels = 0

        for image in images:

            if image.stem not in label_names:

                missing_labels += 1

        if missing_labels > 0:

            print(
                f"  [ERROR] {missing_labels} images "
                f"do not have labels."
            )

            errors += missing_labels

    print()

    # ------------------------------------------------------
    # Validate label contents
    # ------------------------------------------------------

    invalid_labels = 0

    for label_dir in [
        TRAIN_LABELS,
        VAL_LABELS,
        TEST_LABELS,
    ]:

        for label_file in label_dir.glob(
            "*.txt"
        ):

            if not verify_label_file(
                label_file
            ):

                invalid_labels += 1

    if invalid_labels:

        print(
            f"[ERROR] Invalid label files: "
            f"{invalid_labels}"
        )

        errors += invalid_labels

    else:

        print(
            "[OK] All YOLO labels are valid."
        )

    print()

    print(
        f"Total images: {total_images}"
    )

    print(
        f"Total labels: {total_labels}"
    )

    print()

    if errors > 0:

        print(
            f"[ERROR] Dataset verification failed "
            f"with {errors} problem(s)."
        )

        return False

    print(
        "[SUCCESS] Dataset verification passed."
    )

    return True


# ==========================================================
# FINAL STRUCTURE
# ==========================================================

def print_final_structure():

    print("[7/7] Final dataset structure")
    print()

    print("marine-debris-detection/")
    print("└── dataset/")
    print("    ├── images/")
    print("    │   ├── train/")
    print("    │   ├── val/")
    print("    │   └── test/")
    print("    ├── labels/")
    print("    │   ├── train/")
    print("    │   ├── val/")
    print("    │   └── test/")
    print("    └── dataset.yaml")

    print()


# ==========================================================
# MAIN
# ==========================================================

def main():

    print_header()

    print("FINAL YOLO DATASET:")
    print(
        FINAL_DATASET.resolve()
    )

    print()

    print("DATASET YAML:")
    print(
        DATASET_YAML.resolve()
    )

    print()

    print("CLASSES:")

    for class_id, class_name in enumerate(
        ALL_CLASSES
    ):

        print(
            f"  {class_id:2d} -> {class_name}"
        )

    print()

    print("=" * 70)

    print("SOURCE DATASET ROOT:")
    print(
        SOURCE_DATASET.resolve()
    )

    print("=" * 70)

    print()

    # ------------------------------------------------------
    # Check
    # ------------------------------------------------------

    if not check_sources():

        return

    # ------------------------------------------------------
    # Prepare
    # ------------------------------------------------------

    clean_final_dataset()

    # ------------------------------------------------------
    # Marine debris
    # ------------------------------------------------------

    marine_count = (
        process_marine_debris()
    )

    # ------------------------------------------------------
    # Shipwreck
    # ------------------------------------------------------

    shipwreck_count = (
        process_shipwreck()
    )

    # ------------------------------------------------------
    # YAML
    # ------------------------------------------------------

    write_dataset_yaml()

    # ------------------------------------------------------
    # Verify
    # ------------------------------------------------------

    success = verify_dataset()

    # ------------------------------------------------------
    # Structure
    # ------------------------------------------------------

    print_final_structure()

    print("=" * 70)

    if success:

        print(
            "        DATASET READY FOR YOLO TRAINING"
        )

        print("=" * 70)

        print()

        print(
            f"Marine debris images : "
            f"{marine_count}"
        )

        print(
            f"Shipwreck images     : "
            f"{shipwreck_count}"
        )

        print()

        print(
            "Dataset YAML:"
        )

        print(
            DATASET_YAML.resolve()
        )

        print()

        print(
            "Run YOLO training:"
        )

        print(
            "python training\\train.py"
        )

    else:

        print(
            "        DATASET BUILD FAILED"
        )

        print("=" * 70)

        print()

        print(
            "Fix the dataset/annotation problem "
            "before starting training."
        )

    print()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

