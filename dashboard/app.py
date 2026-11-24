
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

MEL_PATH = ROOT / "data" / "processed" / "mel.npz"

UPLOAD_FOLDER = ROOT / "dashboard" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. FLASK CONFIGURATION
# =========================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.secret_key = "mimii-industrial-fault-detection"

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


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

    try:

        from src.trainer import load_model

        print("\nLoading CNN-LSTM model...")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {MODEL_PATH}"
            )

        if not MEL_PATH.exists():
            raise FileNotFoundError(
                f"Mel filterbank not found: {MEL_PATH}"
            )

        # Load trained model and checkpoint
        model, checkpoint = load_model(
            str(MODEL_PATH),
            device=str(DEVICE)
        )

        # Load saved decision threshold
        extra = checkpoint.get("extra", {})

        threshold = float(
            extra["threshold"]
        )

        # IMPORTANT:
        # Mean and standard deviation are arrays,
        # not single scalar values.

        mean = np.asarray(
            checkpoint["mean"],
            dtype=np.float32
        )

        std = np.asarray(
            checkpoint["std"],
            dtype=np.float32
        )

        # Avoid division by zero
        std = np.maximum(std, 1e-8)

        # Load Mel filterbank
        with np.load(MEL_PATH) as mel_data:

            mel_filterbank = mel_data["fb"].copy()

        model.eval()

        print("\nCNN-LSTM model loaded successfully.")
        print("Device:", DEVICE)
        print("Decision threshold:", threshold)
        print("Mean shape:", mean.shape)
        print("Std shape:", std.shape)

        return True

    except Exception:

        print("\nMODEL LOADING ERROR:")
        traceback.print_exc()

        model = None
        checkpoint = None

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
        raise ValueError("Uploaded audio is empty.")

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
        int(frames_per_window * (1 - OVERLAP))
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
# 7. CNN-LSTM BATCH PREDICTION
# =========================================================

def predict_windows(windows):

    if model is None:
        raise RuntimeError(
            "CNN-LSTM model is not loaded."
        )

    # Convert windows into PyTorch tensor
    X = torch.tensor(
        windows,
        dtype=torch.float32,
        device=DEVICE
    )

    # Convert saved normalization values into tensors
    mean_tensor = torch.as_tensor(
        mean,
        dtype=torch.float32,
        device=DEVICE
    )

    std_tensor = torch.as_tensor(
        std,
        dtype=torch.float32,
        device=DEVICE
    )

    # Normalize using checkpoint statistics
    X = (X - mean_tensor) / std_tensor

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy().reshape(-1)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions, probabilities


# =========================================================
# 8. COMPLETE AUDIO ANALYSIS
# =========================================================

def analyze_audio_file(file_path):

    if model is None:
        raise RuntimeError(
            "CNN-LSTM model is unavailable."
        )

    sample_rate, audio = wavfile.read(file_path)

    # Validate sample rate
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected 16000 Hz audio. "
            f"Received {sample_rate} Hz. "
            "Please upload a 16 kHz WAV file."
        )

    if audio.size == 0:
        raise ValueError("Uploaded audio is empty.")

    # Normalize integer WAV audio
    if np.issubdtype(audio.dtype, np.integer):

        max_value = (
            np.iinfo(audio.dtype).max + 1
        )

        audio = (
            audio.astype(np.float32) / max_value
        )

    else:

        audio = audio.astype(np.float32)

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
    mel = preprocess_audio(audio)

    # Create windows
    windows = create_audio_windows(mel)

    # Model prediction
    predictions, probabilities = predict_windows(windows)

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

        "duration": round(duration, 2),

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
# 9. DASHBOARD HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        model_ready=model is not None,
        threshold=threshold
    )


# =========================================================
# 10. AUDIO ANALYSIS API
# =========================================================

@app.route("/api/analyze", methods=["POST"])
def analyze():

    if model is None:

        return jsonify({
            "success": False,
            "error": "CNN-LSTM model is not loaded."
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
        uploaded_file.save(temporary_path)

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

        history.insert(0, {

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

        })

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
# 11. ANALYSIS HISTORY API
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
# 12. MODEL INFORMATION API
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

        "checkpoint": MODEL_PATH.name

    })


# =========================================================
# 13. DOWNLOAD ANALYSIS REPORT
# =========================================================

@app.route("/api/report/<analysis_id>")
def download_report(analysis_id):

    history = session.get(
        "history",
        []
    )

    matching = next(
        (
            item for item in history
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
# 14. ERROR HANDLERS
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error": "File is too large. Maximum size is 25 MB."

    }), 413


# =========================================================
# 15. START FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    print(
        "\nStarting MIMII Industrial Fault Detection Dashboard..."
    )

    model_loaded = load_project_model()

    if model_loaded:

        print("\nMODEL STATUS: ONLINE")

    else:

        print("\nMODEL STATUS: OFFLINE")
        print("Check the model loading error above.")

    print(
        "\nDashboard URL: http://127.0.0.1:5000"
    )

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        use_reloader=False

    )
