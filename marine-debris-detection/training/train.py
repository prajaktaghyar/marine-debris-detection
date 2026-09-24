
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
import torch
from ultralytics import YOLO


# ============================================================
# 1. CLASS DEFINITIONS
# ============================================================

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


# ============================================================
# 2. TRAINING SETTINGS
# ============================================================

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 2
SEED = 42


# ============================================================
# 3. PATH CONFIGURATION
# ============================================================

# Current file:
# marine-debris-detection/training/train.py

TRAINING_DIR = Path(__file__).resolve().parent

# Project root:
# marine-debris-detection/

BASE_DIR = TRAINING_DIR.parent

# Final YOLO dataset:
# marine-debris-detection/dataset/

DATASET_ROOT = BASE_DIR / "dataset"


# Dataset folders

IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

TRAIN_IMAGES = IMAGES_DIR / "train"
VAL_IMAGES = IMAGES_DIR / "val"
TEST_IMAGES = IMAGES_DIR / "test"

TRAIN_LABELS = LABELS_DIR / "train"
VAL_LABELS = LABELS_DIR / "val"
TEST_LABELS = LABELS_DIR / "test"


# Dataset YAML

DATASET_YAML = DATASET_ROOT / "dataset.yaml"


# Model output

MODELS_DIR = BASE_DIR / "models"


# YOLO runs

RUNS_DIR = BASE_DIR / "runs"


# Base YOLO model

BASE_MODEL = BASE_DIR / "yolo11n.pt"


# Dataset converter

CONVERTER_SCRIPT = TRAINING_DIR / "convert_and_train.py"


# ============================================================
# 4. SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 5. PRINT CONFIGURATION
# ============================================================

def print_configuration():

    print("\n" + "=" * 70)
    print("MARINE DEBRIS + AI4SHIPWRECKS YOLO TRAINING")
    print("=" * 70)

    print("\nProject directory:")
    print(BASE_DIR)

    print("\nDataset directory:")
    print(DATASET_ROOT)

    print("\nDataset YAML:")
    print(DATASET_YAML)

    print("\nBase model:")
    print(BASE_MODEL)

    print("\nRuns directory:")
    print(RUNS_DIR)

    print("\nModels directory:")
    print(MODELS_DIR)

    print("\nClasses:")

    for i, class_name in enumerate(ALL_CLASSES):
        print(f"  {i:2d} -> {class_name}")

    print("\nTraining settings:")

    print(f"  Epochs      : {EPOCHS}")
    print(f"  Image size  : {IMAGE_SIZE}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Workers     : {WORKERS}")
    print(f"  Seed        : {SEED}")

    if torch.cuda.is_available():

        print("\nGPU:")
        print(f"  {torch.cuda.get_device_name(0)}")
        print("  Device: CUDA")

    else:

        print("\nGPU:")
        print("  CUDA not available")
        print("  Device: CPU")


# ============================================================
# 6. CREATE REQUIRED DIRECTORIES
# ============================================================

def create_directories():

    print("\n" + "=" * 70)
    print("CREATING DIRECTORIES")
    print("=" * 70)

    directories = [
        DATASET_ROOT,
        IMAGES_DIR,
        LABELS_DIR,

        TRAIN_IMAGES,
        VAL_IMAGES,
        TEST_IMAGES,

        TRAIN_LABELS,
        VAL_LABELS,
        TEST_LABELS,

        MODELS_DIR,
        RUNS_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    print("\nAll required directories are ready.")


# ============================================================
# 7. CREATE DATASET YAML
# ============================================================

def create_dataset_yaml():

    print("\n" + "=" * 70)
    print("CREATING DATASET YAML")
    print("=" * 70)

    data = {
        "path": str(DATASET_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(ALL_CLASSES),
        "names": ALL_CLASSES,
    }

    DATASET_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        DATASET_YAML,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    print("\nDataset YAML created successfully:")

    print(DATASET_YAML)

    print("\nYAML content:")

    with open(
        DATASET_YAML,
        "r",
        encoding="utf-8"
    ) as f:

        print(f.read())


# ============================================================
# 8. COUNT IMAGES
# ============================================================

def count_images(directory):

    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# 9. COUNT LABELS
# ============================================================

def count_labels(directory):

    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".txt"
    )


# ============================================================
# 10. CHECK WHETHER FINAL DATASET EXISTS
# ============================================================

def final_dataset_ready():

    train_images = count_images(
        TRAIN_IMAGES
    )

    val_images = count_images(
        VAL_IMAGES
    )

    train_labels = count_labels(
        TRAIN_LABELS
    )

    val_labels = count_labels(
        VAL_LABELS
    )

    print("\nCurrent final dataset:")

    print(
        f"  Train images : {train_images}"
    )

    print(
        f"  Train labels : {train_labels}"
    )

    print(
        f"  Val images   : {val_images}"
    )

    print(
        f"  Val labels   : {val_labels}"
    )

    return (
        train_images > 0
        and val_images > 0
        and train_labels > 0
        and val_labels > 0
    )


# ============================================================
# 11. RUN DATASET CONVERTER AUTOMATICALLY
# ============================================================

def run_dataset_converter():

    print("\n" + "=" * 70)
    print("DATASET IS EMPTY")
    print("=" * 70)

    print("\nThe final YOLO dataset has not been prepared.")

    print("\nConverter:")

    print(CONVERTER_SCRIPT)

    if not CONVERTER_SCRIPT.exists():

        print(
            "\nERROR: convert_and_train.py was not found."
        )

        print(
            "\nExpected:"
        )

        print(
            CONVERTER_SCRIPT
        )

        return False

    print(
        "\nStarting dataset conversion..."
    )

    print()

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(CONVERTER_SCRIPT)
            ],
            cwd=str(BASE_DIR),
            check=False
        )

    except Exception as e:

        print(
            "\nERROR running converter:"
        )

        print(e)

        return False

    if result.returncode != 0:

        print(
            "\nERROR: Dataset converter failed."
        )

        print(
            "Return code:",
            result.returncode
        )

        return False

    print(
        "\nDataset converter completed."
    )

    return True


# ============================================================
# 12. CHECK DATASET
# ============================================================

def check_dataset():

    print("\n" + "=" * 70)
    print("CHECKING DATASET")
    print("=" * 70)

    if not DATASET_ROOT.exists():

        print("\nERROR:")
        print("Dataset directory does not exist.")

        print("\nExpected:")
        print(DATASET_ROOT)

        return False

    print("\nDataset found:")
    print(DATASET_ROOT)

    splits = {
        "train": (
            TRAIN_IMAGES,
            TRAIN_LABELS
        ),

        "val": (
            VAL_IMAGES,
            VAL_LABELS
        ),

        "test": (
            TEST_IMAGES,
            TEST_LABELS
        ),
    }

    all_ok = True

    for split, (
        image_dir,
        label_dir
    ) in splits.items():

        image_count = count_images(
            image_dir
        )

        label_count = count_labels(
            label_dir
        )

        print(
            f"\n{split.upper():5s} | "
            f"Images: {image_count:6d} | "
            f"Labels: {label_count:6d}"
        )

        if image_count == 0:

            print(
                f"WARNING: No images found in "
                f"{image_dir}"
            )

            if split in ["train", "val"]:
                all_ok = False

        if label_count == 0:

            print(
                f"WARNING: No labels found in "
                f"{label_dir}"
            )

            if split in ["train", "val"]:
                all_ok = False

    return all_ok


# ============================================================
# 13. CHECK IMAGE/LABEL PAIRS
# ============================================================

def check_image_label_pairs():

    print("\n" + "=" * 70)
    print("CHECKING IMAGE/LABEL PAIRS")
    print("=" * 70)

    splits = {
        "train": (
            TRAIN_IMAGES,
            TRAIN_LABELS
        ),

        "val": (
            VAL_IMAGES,
            VAL_LABELS
        ),

        "test": (
            TEST_IMAGES,
            TEST_LABELS
        ),
    }

    total_missing_labels = 0
    total_missing_images = 0

    for split, (
        image_dir,
        label_dir
    ) in splits.items():

        if not image_dir.exists():
            continue

        if not label_dir.exists():
            continue

        images = {
            p.stem
            for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        }

        labels = {
            p.stem
            for p in label_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".txt"
        }

        missing_labels = images - labels

        missing_images = labels - images

        print(f"\n{split.upper()}:")

        print(
            f"  Images       : {len(images)}"
        )

        print(
            f"  Labels       : {len(labels)}"
        )

        print(
            f"  Missing lbls : {len(missing_labels)}"
        )

        print(
            f"  Missing imgs : {len(missing_images)}"
        )

        if missing_labels:

            print("\n  Example missing labels:")

            for name in list(
                sorted(missing_labels)
            )[:5]:

                print(
                    f"    {name}.txt"
                )

        if missing_images:

            print("\n  Example missing images:")

            for name in list(
                sorted(missing_images)
            )[:5]:

                print(
                    f"    {name}"
                )

        total_missing_labels += len(
            missing_labels
        )

        total_missing_images += len(
            missing_images
        )

    print("\n" + "-" * 70)

    if (
        total_missing_labels == 0
        and total_missing_images == 0
    ):

        print(
            "All image/label pairs are correct."
        )

        return True

    print("WARNING:")

    print(
        f"Missing labels: "
        f"{total_missing_labels}"
    )

    print(
        f"Missing images: "
        f"{total_missing_images}"
    )

    return False


# ============================================================
# 14. CHECK LABEL CLASS IDs
# ============================================================

def check_label_class_ids():

    print("\n" + "=" * 70)
    print("CHECKING LABEL CLASS IDs")
    print("=" * 70)

    label_files = []

    for split_dir in [
        TRAIN_LABELS,
        VAL_LABELS,
        TEST_LABELS
    ]:

        if split_dir.exists():

            label_files.extend(
                split_dir.glob("*.txt")
            )

    if not label_files:

        print("\nNo label files found.")

        return False

    invalid_count = 0
    object_count = 0

    class_counts = {
        i: 0
        for i in range(
            len(ALL_CLASSES)
        )
    }

    for label_file in label_files:

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                lines = f.readlines()

        except Exception as e:

            print(
                f"\nWARNING: Could not read "
                f"{label_file}: {e}"
            )

            invalid_count += 1

            continue

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # YOLO:
            # class x_center y_center width height

            if len(parts) != 5:

                print(
                    f"WARNING: Invalid label format: "
                    f"{label_file}:{line_number}"
                )

                invalid_count += 1

                continue

            try:

                class_id = int(
                    parts[0]
                )

                xc = float(
                    parts[1]
                )

                yc = float(
                    parts[2]
                )

                width = float(
                    parts[3]
                )

                height = float(
                    parts[4]
                )

            except ValueError:

                print(
                    f"WARNING: Invalid numeric values: "
                    f"{label_file}:{line_number}"
                )

                invalid_count += 1

                continue

            object_count += 1

            # Class ID
            if (
                class_id < 0
                or class_id >= len(ALL_CLASSES)
            ):

                print(
                    f"WARNING: Invalid class ID "
                    f"{class_id} in "
                    f"{label_file}"
                )

                invalid_count += 1

                continue

            # Bounding box validation
            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):

                print(
                    f"WARNING: Invalid bounding box "
                    f"in {label_file}:{line_number}"
                )

                invalid_count += 1

                continue

            class_counts[
                class_id
            ] += 1

    print("\nObject counts:")

    for class_id, count in class_counts.items():

        print(
            f"  {class_id:2d} -> "
            f"{ALL_CLASSES[class_id]:20s}: "
            f"{count}"
        )

    print(
        "\nTotal objects:",
        object_count
    )

    print(
        "Invalid labels:",
        invalid_count
    )

    if invalid_count == 0:

        print(
            "\nAll label class IDs are valid."
        )

        return True

    print(
        "\nWARNING: Some labels are invalid."
    )

    return False


# ============================================================
# 15. VERIFY DATASET YAML
# ============================================================

def verify_dataset_yaml():

    print("\n" + "=" * 70)
    print("VERIFYING DATASET YAML")
    print("=" * 70)

    if not DATASET_YAML.exists():

        print(
            "\nERROR:"
        )

        print(
            "dataset.yaml does not exist."
        )

        return False

    try:

        with open(
            DATASET_YAML,
            "r",
            encoding="utf-8"
        ) as f:

            data = yaml.safe_load(f)

    except Exception as e:

        print(
            "\nERROR reading YAML:"
        )

        print(e)

        return False

    if not isinstance(
        data,
        dict
    ):

        print(
            "\nERROR: YAML content is invalid."
        )

        return False

    required_keys = [
        "path",
        "train",
        "val",
        "nc",
        "names"
    ]

    for key in required_keys:

        if key not in data:

            print(
                f"\nERROR: Missing YAML key: "
                f"{key}"
            )

            return False

    if data["nc"] != len(
        ALL_CLASSES
    ):

        print(
            "\nERROR: Number of classes "
            "does not match."
        )

        print(
            "Expected:",
            len(ALL_CLASSES)
        )

        print(
            "Found:",
            data["nc"]
        )

        return False

    # YAML names may be a list
    names = data["names"]

    if isinstance(
        names,
        dict
    ):

        names = [
            names[key]
            for key in sorted(
                names,
                key=lambda x: int(x)
            )
        ]

    if list(names) != ALL_CLASSES:

        print(
            "\nERROR: Class names do not match."
        )

        print(
            "Expected:"
        )

        print(
            ALL_CLASSES
        )

        print(
            "\nFound:"
        )

        print(
            names
        )

        return False

    print(
        "\nYAML verified successfully."
    )

    print(
        "\nDataset path:"
    )

    print(
        data["path"]
    )

    print(
        "\nTrain:"
    )

    print(
        data["train"]
    )

    print(
        "\nValidation:"
    )

    print(
        data["val"]
    )

    if "test" in data:

        print(
            "\nTest:"
        )

        print(
            data["test"]
        )

    print(
        "\nNumber of classes:"
    )

    print(
        data["nc"]
    )

    return True


# ============================================================
# 16. FIND BASE MODEL
# ============================================================

def find_base_model():

    print("\n" + "=" * 70)
    print("CHECKING BASE MODEL")
    print("=" * 70)

    if BASE_MODEL.exists():

        print(
            "\nUsing local model:"
        )

        print(
            BASE_MODEL
        )

        return str(
            BASE_MODEL
        )

    print(
        "\nLocal yolo11n.pt not found."
    )

    print(
        "\nUltralytics will try to use/download "
        "yolo11n.pt automatically."
    )

    return "yolo11n.pt"


# ============================================================
# 17. TRAIN YOLO MODEL
# ============================================================

def train_model():

    print("\n" + "=" * 70)
    print("STARTING YOLO TRAINING")
    print("=" * 70)

    model_path = find_base_model()

    print(
        "\nLoading model:"
    )

    print(
        model_path
    )

    try:

        model = YOLO(
            model_path
        )

    except Exception as e:

        print(
            "\nERROR loading YOLO model:"
        )

        print(e)

        return None

    if torch.cuda.is_available():

        device = 0

        print(
            "\nTraining device: GPU"
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = "cpu"

        print(
            "\nTraining device: CPU"
        )

    print(
        "\nDataset used for training:"
    )

    print(
        DATASET_YAML
    )

    print(
        "\nOutput directory:"
    )

    print(
        RUNS_DIR
    )

    print(
        "\nClasses:"
    )

    for i, class_name in enumerate(
        ALL_CLASSES
    ):

        print(
            f"  {i:2d} -> {class_name}"
        )

    try:

        results = model.train(

            # Dataset
            data=str(
                DATASET_YAML
            ),

            # Training
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,

            # Device
            device=device,

            # Output
            project=str(
                RUNS_DIR
            ),

            name="marine_debris_shipwreck",

            exist_ok=True,

            # Model
            pretrained=True,

            optimizer="auto",

            # Early stopping
            patience=15,

            # Saving
            save=True,

            # Plots
            plots=True,

            # Reproducibility
            seed=SEED,

            # Avoid RAM/disk cache issues
            cache=False,

            # Deterministic training
            deterministic=True,

            # Close mosaic near end
            close_mosaic=10,

            # Augmentation
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,

        )

    except Exception as e:

        print(
            "\n" + "=" * 70
        )

        print(
            "TRAINING FAILED"
        )

        print(
            "=" * 70
        )

        print(
            "\nError:"
        )

        print(e)

        return None

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    return results


# ============================================================
# 18. FIND BEST MODEL
# ============================================================

def find_best_model():

    print("\n" + "=" * 70)
    print("SEARCHING FOR BEST MODEL")
    print("=" * 70)

    expected_best = (
        RUNS_DIR
        / "marine_debris_shipwreck"
        / "weights"
        / "best.pt"
    )

    if expected_best.exists():

        print(
            "\nBest model found:"
        )

        print(
            expected_best
        )

        return expected_best

    print(
        "\nExpected best.pt not found."
    )

    candidates = list(
        RUNS_DIR.rglob(
            "best.pt"
        )
    )

    if not candidates:

        print(
            "\nNo best.pt found anywhere "
            "inside runs."
        )

        return None

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    best = candidates[0]

    print(
        "\nBest model found:"
    )

    print(
        best
    )

    return best


# ============================================================
# 19. COPY BEST MODEL
# ============================================================

def copy_best_model():

    print("\n" + "=" * 70)
    print("COPYING BEST MODEL")
    print("=" * 70)

    best_model = find_best_model()

    if best_model is None:

        print(
            "\nERROR:"
        )

        print(
            "Could not find best.pt."
        )

        return False

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        MODELS_DIR / "best.pt"
    )

    try:

        shutil.copy2(
            best_model,
            destination
        )

    except Exception as e:

        print(
            "\nERROR copying best.pt:"
        )

        print(e)

        return False

    print(
        "\nModel copied successfully."
    )

    print(
        "Source:"
    )

    print(
        best_model
    )

    print(
        "\nDestination:"
    )

    print(
        destination
    )

    return True


# ============================================================
# 20. VALIDATE TRAINED MODEL
# ============================================================

def validate_model():

    print("\n" + "=" * 70)
    print("VALIDATING TRAINED MODEL")
    print("=" * 70)

    model_path = (
        MODELS_DIR / "best.pt"
    )

    if not model_path.exists():

        model_path = find_best_model()

    if model_path is None:

        print(
            "\nNo trained model available."
        )

        return None

    print(
        "\nValidation model:"
    )

    print(
        model_path
    )

    try:

        model = YOLO(
            str(model_path)
        )

        if torch.cuda.is_available():

            device = 0

        else:

            device = "cpu"

        metrics = model.val(

            data=str(
                DATASET_YAML
            ),

            imgsz=IMAGE_SIZE,

            batch=BATCH_SIZE,

            device=device,

            split="val",

            plots=True,

        )

    except Exception as e:

        print(
            "\nValidation failed:"
        )

        print(e)

        return None

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATION COMPLETED"
    )

    print(
        "=" * 70
    )

    try:

        print(
            "\nmAP50:",
            metrics.box.map50
        )

        print(
            "mAP50-95:",
            metrics.box.map
        )

    except Exception:

        pass

    return metrics


# ============================================================
# 21. PRINT FINAL INFORMATION
# ============================================================

def print_final_information():

    print("\n" + "=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)

    print(
        "\nDataset:"
    )

    print(
        DATASET_ROOT
    )

    print(
        "\nDataset YAML:"
    )

    print(
        DATASET_YAML
    )

    print(
        "\nTraining runs:"
    )

    print(
        RUNS_DIR
    )

    print(
        "\nTrained model:"
    )

    model_path = (
        MODELS_DIR / "best.pt"
    )

    if model_path.exists():

        print(
            model_path
        )

    else:

        print(
            "best.pt was not copied."
        )

    print(
        "\nClasses:"
    )

    for i, class_name in enumerate(
        ALL_CLASSES
    ):

        print(
            f"  {i:2d} -> {class_name}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )


# ============================================================
# 22. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print_configuration()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Dataset YAML
    # --------------------------------------------------------

    create_dataset_yaml()

    # --------------------------------------------------------
    # Verify YAML
    # --------------------------------------------------------

    if not verify_dataset_yaml():

        print(
            "\nERROR: Dataset YAML verification failed."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check whether dataset already exists
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CHECKING FINAL DATASET STATUS"
    )

    print(
        "=" * 70
    )

    if not final_dataset_ready():

        # ----------------------------------------------------
        # Automatically convert source datasets
        # ----------------------------------------------------

        converter_ok = (
            run_dataset_converter()
        )

        if not converter_ok:

            print(
                "\nERROR: Dataset preparation failed."
            )

            sys.exit(1)

    else:

        print(
            "\nFinal YOLO dataset already exists."
        )

        print(
            "Skipping automatic conversion."
        )

    # --------------------------------------------------------
    # Re-create YAML because converter may update it
    # --------------------------------------------------------

    create_dataset_yaml()

    # --------------------------------------------------------
    # Verify YAML again
    # --------------------------------------------------------

    if not verify_dataset_yaml():

        print(
            "\nERROR: Dataset YAML verification failed."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not check_dataset():

        print(
            "\nERROR: Dataset check failed."
        )

        print(
            "\nExpected:"
        )

        print(
            DATASET_ROOT / "images/train"
        )

        print(
            DATASET_ROOT / "images/val"
        )

        print(
            DATASET_ROOT / "labels/train"
        )

        print(
            DATASET_ROOT / "labels/val"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check image-label pairs
    # --------------------------------------------------------

    pairs_ok = (
        check_image_label_pairs()
    )

    if not pairs_ok:

        print(
            "\nERROR: Image/label pairing problem found."
        )

        print(
            "Fix the dataset before training."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check class IDs
    # --------------------------------------------------------

    labels_ok = (
        check_label_class_ids()
    )

    if not labels_ok:

        print(
            "\nERROR: Invalid YOLO labels found."
        )

        print(
            "Training stopped to prevent "
            "incorrect class mapping."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results = train_model()

    if results is None:

        print(
            "\nTraining did not complete."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Copy best.pt
    # --------------------------------------------------------

    if not copy_best_model():

        print(
            "\nWARNING: best.pt could not be copied."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_model()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_final_information()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

