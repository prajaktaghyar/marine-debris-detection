
import os
import json
import csv
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from werkzeug.utils import secure_filename

from preprocessing.preprocess import preprocess_image
from detection.detector import MarineDebrisDetector


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")


# ============================================================
# TRAINED YOLO MODEL
# ============================================================
#
# NEW TRAINING PIPELINE OUTPUT:
#
# marine-debris-detection/
# └── models/
#     └── best.pt
#
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt"
)


# ============================================================
# YOLO CLASSES
# ============================================================
#
# Class IDs from dataset.yaml:
#
# 0  bottle
# 1  can
# 2  chain
# 3  drink-carton
# 4  hook
# 5  propeller
# 6  shampoo-bottle
# 7  standing-bottle
# 8  tire
# 9  valve
# 10 shipwreck
#
# ============================================================

CLASS_NAMES = [
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
    "shipwreck"
]


SHIPWRECK_CLASS_ID = 10


# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.35
IMAGE_SIZE = 640


# ============================================================
# ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tif",
    "tiff",
    "webp"
}


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ============================================================
# MODEL PATH CHECK
# ============================================================

print("=" * 70)
print("        MARINE DEBRIS + SHIPWRECK DETECTION SYSTEM")
print("=" * 70)

print()
print("Base directory:")
print(BASE_DIR)

print()
print("YOLO model path:")
print(MODEL_PATH)

print()

if os.path.exists(MODEL_PATH):

    print("Model file: FOUND")

    try:
        model_size = os.path.getsize(MODEL_PATH)

        print(
            "Model size:",
            model_size,
            "bytes"
        )

    except Exception:
        pass

else:

    print("Model file: NOT FOUND")

    print()
    print("Expected model location:")
    print(MODEL_PATH)

print()
print("Expected classes:")

for class_id, class_name in enumerate(CLASS_NAMES):
    print(
        f"  {class_id}: {class_name}"
    )

print("=" * 70)


# ============================================================
# YOLO DETECTOR
# ============================================================

try:

    detector = MarineDebrisDetector(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )

    print()
    print("YOLO detector initialized successfully.")

except Exception as e:

    print()
    print("=" * 70)
    print("ERROR WHILE LOADING YOLO MODEL")
    print("=" * 70)

    print(str(e))

    print("=" * 70)

    detector = None


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# NORMALIZE DETECTION RESULT
# ============================================================
#
# This function makes sure detections returned by
# detection/detector.py have a consistent format.
#
# ============================================================

def normalize_detections(result):

    detections = result.get(
        "detections",
        []
    )

    normalized = []

    for detection in detections:

        if not isinstance(
            detection,
            dict
        ):
            continue

        class_id = detection.get(
            "class_id",
            detection.get(
                "cls",
                -1
            )
        )

        try:
            class_id = int(class_id)
        except Exception:
            class_id = -1

        class_name = detection.get(
            "class_name",
            detection.get(
                "class",
                ""
            )
        )

        if (
            not class_name
            and
            0 <= class_id < len(CLASS_NAMES)
        ):
            class_name = CLASS_NAMES[class_id]

        confidence = detection.get(
            "confidence",
            detection.get(
                "conf",
                0
            )
        )

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        # Handle confidence supplied as percentage
        if confidence > 1:
            confidence = confidence / 100.0

        normalized_detection = {
            "class_id": class_id,
            "class_name": class_name,
            "class": class_name,
            "confidence": round(
                confidence,
                4
            ),
            "confidence_percent": round(
                confidence * 100,
                2
            )
        }

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        if "bbox" in detection:

            normalized_detection["bbox"] = detection["bbox"]

        elif all(
            key in detection
            for key in ["x1", "y1", "x2", "y2"]
        ):

            normalized_detection["bbox"] = {
                "x1": detection["x1"],
                "y1": detection["y1"],
                "x2": detection["x2"],
                "y2": detection["y2"]
            }

        normalized.append(
            normalized_detection
        )

    return normalized


# ============================================================
# ADD MODEL CLASS INFORMATION
# ============================================================

def enrich_result(result):

    if not isinstance(
        result,
        dict
    ):
        result = {}

    detections = normalize_detections(
        result
    )

    result["detections"] = detections

    # --------------------------------------------------------
    # Detection count
    # --------------------------------------------------------

    result["detection_count"] = len(
        detections
    )

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if len(detections) == 0:

        result["status"] = "No detections"

        result["best_class"] = ""

        result["best_confidence"] = 0

        result["objects"] = []

        result["shipwreck_detected"] = False

        result["marine_debris_detected"] = False

        return result

    # --------------------------------------------------------
    # Best detection
    # --------------------------------------------------------

    best_detection = max(
        detections,
        key=lambda x: x.get(
            "confidence",
            0
        )
    )

    result["best_class"] = best_detection.get(
        "class_name",
        ""
    )

    result["best_confidence"] = round(
        best_detection.get(
            "confidence",
            0
        ) * 100,
        2
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    result["status"] = "Detection successful"

    # --------------------------------------------------------
    # Shipwreck detection
    # --------------------------------------------------------

    shipwrecks = [
        detection
        for detection in detections
        if (
            detection.get("class_id") == SHIPWRECK_CLASS_ID
            or
            detection.get("class_name", "").lower()
            == "shipwreck"
        )
    ]

    result["shipwreck_detected"] = (
        len(shipwrecks) > 0
    )

    result["shipwreck_count"] = len(
        shipwrecks
    )

    # --------------------------------------------------------
    # Marine debris detection
    # --------------------------------------------------------

    marine_debris = [
        detection
        for detection in detections
        if detection.get(
            "class_name",
            ""
        ).lower() != "shipwreck"
    ]

    result["marine_debris_detected"] = (
        len(marine_debris) > 0
    )

    result["marine_debris_count"] = len(
        marine_debris
    )

    # --------------------------------------------------------
    # Simple object list
    # --------------------------------------------------------

    result["objects"] = [
        {
            "class": detection.get(
                "class_name",
                ""
            ),
            "confidence": detection.get(
                "confidence_percent",
                0
            ),
            "class_id": detection.get(
                "class_id",
                -1
            )
        }
        for detection in detections
    ]

    return result


# ============================================================
# SAVE DETECTION REPORT
# ============================================================

def save_report(result):

    csv_path = os.path.join(
        REPORT_FOLDER,
        "report.csv"
    )

    json_path = os.path.join(
        REPORT_FOLDER,
        "report.json"
    )

    timestamp = datetime.now().isoformat()

    detections = result.get(
        "detections",
        []
    )

    # --------------------------------------------------------
    # CSV DATA
    # --------------------------------------------------------

    row = {
        "timestamp": timestamp,

        "filename": result.get(
            "filename",
            ""
        ),

        "status": result.get(
            "status",
            ""
        ),

        "detections": len(
            detections
        ),

        "best_class": result.get(
            "best_class",
            ""
        ),

        "best_confidence": result.get(
            "best_confidence",
            0
        ),

        "shipwreck_detected": result.get(
            "shipwreck_detected",
            False
        ),

        "shipwreck_count": result.get(
            "shipwreck_count",
            0
        ),

        "marine_debris_count": result.get(
            "marine_debris_count",
            0
        )
    }

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    file_exists = os.path.exists(
        csv_path
    )

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    records = []

    if os.path.exists(json_path):

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                records = json.load(f)

            if not isinstance(
                records,
                list
            ):
                records = []

        except Exception:

            records = []

    json_record = {
        "timestamp": timestamp,
        **result
    }

    records.append(
        json_record
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            indent=4
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    if detector is not None:

        model_loaded = detector.model_loaded

    else:

        model_loaded = False

    return render_template(
        "dashboard.html",
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES
    )


# ============================================================
# MODEL STATUS API
# ============================================================

@app.route("/api/status")
def status():

    if detector is not None:

        model_loaded = detector.model_loaded

        confidence_threshold = (
            detector.confidence_threshold
        )

        iou_threshold = (
            detector.iou_threshold
        )

    else:

        model_loaded = False

        confidence_threshold = (
            CONFIDENCE_THRESHOLD
        )

        iou_threshold = (
            IOU_THRESHOLD
        )

    return jsonify({

        "model_loaded":
            model_loaded,

        "model_exists":
            os.path.exists(
                MODEL_PATH
            ),

        "model_path":
            MODEL_PATH,

        "confidence_threshold":
            confidence_threshold,

        "iou_threshold":
            iou_threshold,

        "image_size":
            IMAGE_SIZE,

        "num_classes":
            len(CLASS_NAMES),

        "classes":
            CLASS_NAMES,

        "time":
            datetime.now().isoformat()

    })


# ============================================================
# IMAGE UPLOAD + DETECTION
# ============================================================

@app.route(
    "/api/detect",
    methods=["POST"]
)
def detect():

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if detector is None:

        return jsonify({

            "success": False,

            "error":
                "YOLO detector could not be initialized."

        }), 500

    if not detector.model_loaded:

        return jsonify({

            "success": False,

            "error":
                "YOLO model is not loaded. "
                "Check the model path: "
                + MODEL_PATH

        }), 500

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image uploaded"

        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No file selected"

        }), 400

    # --------------------------------------------------------
    # CHECK IMAGE FORMAT
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return jsonify({

            "success": False,

            "error":
                "Unsupported image format"

        }), 400

    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    # --------------------------------------------------------
    # CREATE UNIQUE FILE NAME
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    original_name = (
        f"{timestamp}_{filename}"
    )

    # --------------------------------------------------------
    # INPUT IMAGE PATH
    # --------------------------------------------------------

    input_path = os.path.join(
        UPLOAD_FOLDER,
        original_name
    )

    # --------------------------------------------------------
    # SAVE UPLOADED IMAGE
    # --------------------------------------------------------

    file.save(
        input_path
    )

    try:

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        processed_path = preprocess_image(
            input_path,
            UPLOAD_FOLDER
        )

        # ====================================================
        # YOLO DETECTION
        # ====================================================

        result = detector.detect(
            processed_path,
            RESULT_FOLDER
        )

        # ====================================================
        # MAKE RESULT SAFE AND CONSISTENT
        # ====================================================

        result = enrich_result(
            result
        )

        # ====================================================
        # ADD FILE INFORMATION
        # ====================================================

        result["filename"] = filename

        result["input_image"] = (
            "/uploads/"
            + original_name
        )

        # ====================================================
        # RESULT IMAGE URL
        # ====================================================

        if result.get(
            "result_image"
        ):

            result["result_image"] = (
                "/results/"
                +
                os.path.basename(
                    result["result_image"]
                )
            )

        # ====================================================
        # SAVE REPORT
        # ====================================================

        save_report(
            result
        )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({

            "success": True,

            **result

        })

    except Exception as e:

        print()
        print("=" * 70)
        print("DETECTION ERROR")
        print("=" * 70)
        print(str(e))
        print("=" * 70)

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# SERVE RESULT IMAGES
# ============================================================

@app.route(
    "/results/<filename>"
)
def result_file(filename):

    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


# ============================================================
# RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("STARTING MARINE DEBRIS + SHIPWRECK DETECTION SERVER")
    print("=" * 70)

    print()
    print("Model path:")
    print(MODEL_PATH)

    print()

    if detector is not None:

        if detector.model_loaded:

            print(
                "YOLO MODEL: LOADED"
            )

            print()
            print(
                "Number of classes:",
                len(CLASS_NAMES)
            )

            print()
            print("Classes:")

            for class_id, class_name in enumerate(
                CLASS_NAMES
            ):

                print(
                    f"  {class_id}: {class_name}"
                )

        else:

            print(
                "YOLO MODEL: NOT LOADED"
            )

    else:

        print(
            "YOLO MODEL: NOT LOADED"
        )

    print()
    print("Confidence threshold:")
    print(CONFIDENCE_THRESHOLD)

    print()
    print("IoU threshold:")
    print(IOU_THRESHOLD)

    print()
    print("Server:")
    print("http://127.0.0.1:5000")

    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )


