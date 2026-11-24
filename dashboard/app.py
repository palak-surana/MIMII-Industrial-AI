
import os
import sys
import uuid
import json
import traceback

from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_file
)

from scipy.io import wavfile


# =========================================================
# 1. PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = (
    ROOT
    / "results"
    / "active_learning_final"
    / "checkpoints"
    / "seed_1_uncertainty_budget_40pct.pt"
)

MEL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "mel.npz"
)

UPLOAD_FOLDER = (
    ROOT
    / "dashboard"
    / "uploads"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. FLASK CONFIGURATION
# =========================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "mimii-industrial-fault-detection-dev-key"
)

app.config["MAX_CONTENT_LENGTH"] = (
    25 * 1024 * 1024
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

if os.environ.get("RENDER"):
    app.config["SESSION_COOKIE_SECURE"] = True


# =========================================================
# 3. MODEL CONFIGURATION
# =========================================================

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 512

WINDOW_SECONDS = 1.0
OVERLAP = 0.50

LOW_FREQ = 50
HIGH_FREQ = 7500

DEVICE = torch.device("cpu")

model = None
checkpoint = None

threshold = None
mean = None
std = None

mel_filterbank = None

model_load_error = None


# =========================================================
# 4. LOAD CNN-LSTM MODEL
# =========================================================

def load_project_model():

    global model
    global checkpoint
    global threshold
    global mean
    global std
    global mel_filterbank
    global model_load_error

    try:

        from src.trainer import load_model

        print("\n======================================")
        print("Loading MIMII CNN-LSTM Model")
        print("======================================")

        # Check model checkpoint

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {MODEL_PATH}"
            )

        # Check Mel filterbank

        if not MEL_PATH.exists():

            raise FileNotFoundError(
                f"Mel filterbank not found: {MEL_PATH}"
            )

        # Load trained model

        loaded_model, loaded_checkpoint = load_model(
            str(MODEL_PATH),
            device=str(DEVICE)
        )

        # Validate checkpoint

        extra = loaded_checkpoint.get(
            "extra",
            {}
        )

        if "threshold" not in extra:

            raise KeyError(
                "Decision threshold is missing from checkpoint."
            )

        loaded_threshold = float(
            extra["threshold"]
        )

        loaded_mean = np.asarray(
            loaded_checkpoint["mean"],
            dtype=np.float32
        )

        loaded_std = np.asarray(
            loaded_checkpoint["std"],
            dtype=np.float32
        )

        if not np.isfinite(loaded_mean).all():

            raise ValueError(
                "Checkpoint mean contains invalid values."
            )

        if not np.isfinite(loaded_std).all():

            raise ValueError(
                "Checkpoint standard deviation contains invalid values."
            )

        if not 0 <= loaded_threshold <= 1:

            raise ValueError(
                "Invalid decision threshold in checkpoint."
            )

        loaded_std = np.maximum(
            loaded_std,
            1e-8
        )

        # Load Mel filterbank

        with np.load(MEL_PATH) as mel_data:

            if "fb" not in mel_data:

                raise KeyError(
                    "The Mel file does not contain the 'fb' array."
                )

            loaded_filterbank = mel_data["fb"].copy()

        if not np.isfinite(loaded_filterbank).all():

            raise ValueError(
                "Mel filterbank contains invalid values."
            )

        # Set model to evaluation mode

        loaded_model.eval()

        # Update global variables only after successful loading

        model = loaded_model
        checkpoint = loaded_checkpoint

        threshold = loaded_threshold
        mean = loaded_mean
        std = loaded_std

        mel_filterbank = loaded_filterbank

        model_load_error = None

        print("\nCNN-LSTM MODEL LOADED SUCCESSFULLY")
        print("Device:", DEVICE)
        print("Threshold:", threshold)
        print("Mean shape:", mean.shape)
        print("Std shape:", std.shape)
        print("Mel filterbank shape:", mel_filterbank.shape)

        return True

    except Exception as error:

        print("\n======================================")
        print("MODEL LOADING ERROR")
        print("======================================")

        traceback.print_exc()

        model = None
        checkpoint = None

        threshold = None
        mean = None
        std = None

        mel_filterbank = None

        model_load_error = str(error)

        return False


# =========================================================
# 5. AUDIO PREPROCESSING
# =========================================================

def preprocess_audio(audio):

    from src.features import bandpass, log_mel

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # Convert stereo into mono

    if audio.ndim == 2:

        audio = audio[:, 0]

    if audio.size == 0:

        raise ValueError(
            "Uploaded audio is empty."
        )

    if not np.isfinite(audio).all():

        raise ValueError(
            "Audio contains invalid numerical values."
        )

    # Apply bandpass filter

    filtered_audio = bandpass(
        audio,
        SAMPLE_RATE,
        LOW_FREQ,
        HIGH_FREQ
    )

    # Generate Log-Mel spectrogram

    mel = log_mel(
        filtered_audio,
        N_FFT,
        HOP_LENGTH,
        mel_filterbank
    )

    mel = np.asarray(
        mel,
        dtype=np.float32
    )

    if mel.ndim != 2:

        raise ValueError(
            f"Unexpected Log-Mel shape: {mel.shape}"
        )

    # Expected format: (64, time_frames)

    if mel.shape[0] != 64 and mel.shape[1] == 64:

        mel = mel.T

    if mel.shape[0] != 64:

        raise ValueError(
            f"Expected 64 Mel bins, received {mel.shape}"
        )

    if not np.isfinite(mel).all():

        raise ValueError(
            "Generated Mel spectrogram contains invalid values."
        )

    return mel


# =========================================================
# 6. CREATE AUDIO WINDOWS
# =========================================================

def create_audio_windows(mel):

    frames_per_window = int(
        WINDOW_SECONDS * SAMPLE_RATE // HOP_LENGTH
    )

    step_frames = max(
        1,
        int(
            frames_per_window * (1 - OVERLAP)
        )
    )

    windows = []

    for start in range(
        0,
        mel.shape[1] - frames_per_window + 1,
        step_frames
    ):

        end = start + frames_per_window

        windows.append(
            mel[:, start:end]
        )

    if not windows:

        raise ValueError(
            "Audio must be at least 1 second long."
        )

    return np.asarray(
        windows,
        dtype=np.float32
    )


# =========================================================
# 7. NORMALIZATION
# =========================================================

def normalize_windows(X):

    mean_array = np.asarray(
        mean,
        dtype=np.float32
    )

    std_array = np.asarray(
        std,
        dtype=np.float32
    )

    # Support scalar, per-bin and per-bin/per-frame statistics.

    if mean_array.ndim == 1 and mean_array.size == X.shape[1]:

        mean_array = mean_array.reshape(1, -1, 1)

    elif mean_array.ndim == 2:

        mean_array = mean_array.reshape(
            1,
            mean_array.shape[0],
            mean_array.shape[1]
        )

    if std_array.ndim == 1 and std_array.size == X.shape[1]:

        std_array = std_array.reshape(1, -1, 1)

    elif std_array.ndim == 2:

        std_array = std_array.reshape(
            1,
            std_array.shape[0],
            std_array.shape[1]
        )

    mean_tensor = torch.as_tensor(
        mean_array,
        dtype=torch.float32,
        device=DEVICE
    )

    std_tensor = torch.as_tensor(
        std_array,
        dtype=torch.float32,
        device=DEVICE
    )

    return (X - mean_tensor) / std_tensor


# =========================================================
# 8. CNN-LSTM BATCH PREDICTION
# =========================================================

def predict_windows(windows):

    if model is None:

        raise RuntimeError(
            "CNN-LSTM model is not loaded."
        )

    # Convert windows into PyTorch tensor

    X = torch.as_tensor(
        windows,
        dtype=torch.float32,
        device=DEVICE
    )

    # Normalize using checkpoint statistics

    X = normalize_windows(X)

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy().reshape(-1)

    if not np.isfinite(probabilities).all():

        raise ValueError(
            "Model generated invalid prediction probabilities."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions, probabilities


# =========================================================
# 9. COMPLETE AUDIO ANALYSIS
# =========================================================

def analyze_audio_file(file_path):

    if model is None:

        raise RuntimeError(
            "CNN-LSTM model is unavailable."
        )

    # Read WAV file

    sample_rate, audio = wavfile.read(
        file_path
    )

    # Validate sample rate

    if sample_rate != SAMPLE_RATE:

        raise ValueError(
            f"Expected 16000 Hz audio. "
            f"Received {sample_rate} Hz. "
            "Please upload a 16 kHz WAV file."
        )

    if audio.size == 0:

        raise ValueError(
            "Uploaded audio is empty."
        )

    # Normalize integer WAV audio

    if np.issubdtype(audio.dtype, np.integer):

        if np.issubdtype(audio.dtype, np.unsignedinteger):

            audio = (
                audio.astype(np.float32)
                - 128.0
            ) / 128.0

        else:

            max_value = (
                np.iinfo(audio.dtype).max + 1
            )

            audio = (
                audio.astype(np.float32)
                / max_value
            )

    else:

        audio = audio.astype(
            np.float32
        )

    # Convert stereo into mono

    if audio.ndim == 2:

        audio = audio[:, 0]

    if not np.isfinite(audio).all():

        raise ValueError(
            "Audio contains invalid numerical values."
        )

    if len(audio) < SAMPLE_RATE:

        raise ValueError(
            "Audio must be at least 1 second long."
        )

    duration = len(audio) / SAMPLE_RATE

    # Preprocessing

    mel = preprocess_audio(
        audio
    )

    # Create windows

    windows = create_audio_windows(
        mel
    )

    # Model prediction

    predictions, probabilities = predict_windows(
        windows
    )

    # Window statistics

    total_windows = len(predictions)

    healthy_count = int(
        np.sum(predictions == 0)
    )

    faulty_count = int(
        np.sum(predictions == 1)
    )

    average_probability = float(
        np.mean(probabilities)
    )

    maximum_probability = float(
        np.max(probabilities)
    )

    # Overall machine condition

    overall_prediction = (
        "FAULTY"
        if faulty_count > healthy_count
        else "HEALTHY"
    )

    # Waveform display

    waveform_step = max(
        1,
        len(audio) // 1500
    )

    waveform = audio[::waveform_step]

    # Spectrogram display

    display_mel = mel[
        :,
        ::max(1, mel.shape[1] // 180)
    ]

    display_mel = np.nan_to_num(
        display_mel,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    display_min = float(
        np.min(display_mel)
    )

    display_max = float(
        np.max(display_mel)
    )

    if display_max > display_min:

        display_mel = (
            (display_mel - display_min)
            / (display_max - display_min)
        )

    else:

        display_mel = np.zeros_like(
            display_mel
        )

    # Final result

    return {

        "prediction": overall_prediction,

        "duration": round(
            duration,
            2
        ),

        "sample_rate": int(sample_rate),

        "total_windows": int(total_windows),

        "healthy_windows": healthy_count,

        "faulty_windows": faulty_count,

        "average_probability": round(
            average_probability * 100,
            2
        ),

        "maximum_probability": round(
            maximum_probability * 100,
            2
        ),

        "threshold": round(
            threshold,
            6
        ),

        "waveform": waveform.tolist(),

        "spectrogram": display_mel.tolist(),

        "window_probabilities": (
            probabilities * 100
        ).round(2).tolist(),

        "window_predictions": predictions.tolist(),

        "analysis_time": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    }


# =========================================================
# 10. DASHBOARD HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        model_ready=model is not None,
        threshold=threshold
    )


# =========================================================
# 11. AUDIO ANALYSIS API
# =========================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    if model is None:

        return jsonify({

            "success": False,

            "error": (
                "CNN-LSTM model is not loaded. "
                "Please check the deployment logs."
            )

        }), 503

    if "audio" not in request.files:

        return jsonify({

            "success": False,

            "error": "Please select an audio file."

        }), 400

    uploaded_file = request.files["audio"]

    if uploaded_file.filename == "":

        return jsonify({

            "success": False,

            "error": "No audio file selected."

        }), 400

    if not uploaded_file.filename.lower().endswith(".wav"):

        return jsonify({

            "success": False,

            "error": "Only WAV files are supported."

        }), 400

    temporary_path = (
        UPLOAD_FOLDER
        / f"{uuid.uuid4().hex}.wav"
    )

    try:

        # Save uploaded audio

        uploaded_file.save(
            temporary_path
        )

        # Run analysis

        result = analyze_audio_file(
            temporary_path
        )

        analysis_id = uuid.uuid4().hex

        result["analysis_id"] = analysis_id

        # Save history

        history = session.get(
            "history",
            []
        )

        history.insert(
            0,
            {

                "id": analysis_id,

                "filename": uploaded_file.filename,

                "prediction": result["prediction"],

                "duration": result["duration"],

                "analysis_time": result["analysis_time"],

                "average_probability": result["average_probability"],

                "maximum_probability": result["maximum_probability"],

                "total_windows": result["total_windows"],

                "healthy_windows": result["healthy_windows"],

                "faulty_windows": result["faulty_windows"],

                "threshold": result["threshold"]

            }
        )

        session["history"] = history[:20]

        session.modified = True

        return jsonify({

            "success": True,

            "result": result

        })

    except Exception as error:

        print("\nAUDIO ANALYSIS ERROR:")

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500

    finally:

        if temporary_path.exists():

            temporary_path.unlink()


# =========================================================
# 12. ANALYSIS HISTORY API
# =========================================================

@app.route("/api/history")
def get_history():

    return jsonify({

        "success": True,

        "history": session.get(
            "history",
            []
        )

    })


# =========================================================
# 13. MODEL INFORMATION API
# =========================================================

@app.route("/api/model-info")
def model_info():

    return jsonify({

        "success": True,

        "model_name": "CNN-LSTM",

        "dataset": "MIMII Sound Dataset",

        "sample_rate": SAMPLE_RATE,

        "window_seconds": WINDOW_SECONDS,

        "overlap": OVERLAP,

        "threshold": threshold,

        "device": str(DEVICE),

        "model_loaded": model is not None,

        "model_status": (
            "ONLINE"
            if model is not None
            else "OFFLINE"
        ),

        "model_error": model_load_error,

        "checkpoint": MODEL_PATH.name

    })


# =========================================================
# 14. DOWNLOAD ANALYSIS REPORT
# =========================================================

@app.route(
    "/api/report/<analysis_id>"
)
def download_report(analysis_id):

    history = session.get(
        "history",
        []
    )

    matching = next(
        (
            item
            for item in history
            if item["id"] == analysis_id
        ),
        None
    )

    if matching is None:

        return jsonify({

            "success": False,

            "error": "Analysis report not found."

        }), 404

    report_path = (
        UPLOAD_FOLDER
        / f"{analysis_id}_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report_file:

        json.dump(
            matching,
            report_file,
            indent=4
        )

    return send_file(

        report_path,

        as_attachment=True,

        download_name="MIMII_Analysis_Report.json"

    )


# =========================================================
# 15. ERROR HANDLERS
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error": (
            "File is too large. "
            "Maximum size is 25 MB."
        )

    }), 413


# =========================================================
# 16. LOAD MODEL WHEN APPLICATION STARTS
# =========================================================

# IMPORTANT:
# This runs when Gunicorn imports dashboard.app.
# It also runs when Flask is started directly.

print("\n======================================")
print("MIMII INDUSTRIAL FAULT DETECTION")
print("Initializing application...")
print("======================================")

model_loaded = load_project_model()

if model_loaded:

    print("\nMODEL STATUS: ONLINE")

else:

    print("\nMODEL STATUS: OFFLINE")

    print(
        "Model error:",
        model_load_error
    )


# =========================================================
# 17. START LOCAL FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    print(
        "\nStarting MIMII Industrial Fault Detection Dashboard..."
    )

    print(
        "\nDashboard URL: http://127.0.0.1:5000"
    )

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        use_reloader=False

    )
