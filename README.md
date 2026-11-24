# MIMII Industrial AI – Machine Fault Detection Using CNN-LSTM and Agentic Active Learning

An AI-powered industrial machine fault detection system using the **MIMII Sound Dataset**, CNN-LSTM deep learning, and an Agentic Active Learning framework.

The project investigates whether uncertainty-based active learning, managed by a multi-agent system, can achieve cross-machine fault detection performance with fewer labelled audio segments than random sampling.

## 🚀 Live Project

- **Live Dashboard:** https://mimii-industrial-ai.onrender.com
- **GitHub Repository:** https://github.com/palak-surana/MIMII-Industrial-AI

> Note: The dashboard is deployed on Render. Model availability depends on successful loading of the trained checkpoint and Mel filterbank files.

## 🎯 Project Objectives

- Detect abnormal machine operating conditions using machine sound recordings.
- Extract audio features using band-pass filtering and Log-Mel spectrograms.
- Train a CNN-LSTM model for machine fault detection.
- Compare random sampling with uncertainty-based active learning.
- Implement a multi-agent framework to coordinate active learning.
- Evaluate model performance using machine-wise train, validation, and test splits.
- Visualize experiment results and model predictions through a web dashboard.

## 🧠 Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, CNN-LSTM |
| Audio Processing | NumPy, SciPy, Log-Mel Spectrogram |
| Machine Learning | Scikit-learn |
| Active Learning | Random Sampling, Uncertainty Sampling |
| Agentic AI | Multi-Agent Framework |
| Explainable AI | Integrated Gradients |
| Web Development | Flask, HTML, CSS, JavaScript |
| Deployment | Gunicorn, Render |
| Version Control | Git, GitHub |

## 🏗️ Project Architecture

1. **Audio Data Collection:** Machine sound recordings are obtained from the MIMII dataset.
2. **Preprocessing:** Audio signals are filtered and converted into Log-Mel spectrogram features.
3. **Data Splitting:** Machine-wise splitting is performed to prevent data leakage.
4. **Model Training:** CNN-LSTM learns patterns from machine audio features.
5. **Active Learning:** Random and uncertainty-based strategies select samples for annotation.
6. **Agentic Framework:** Multiple agents coordinate sample selection and annotation decisions.
7. **Evaluation:** Model performance is evaluated using classification metrics.
8. **Dashboard:** Users can interact with the deployed Flask application for audio analysis.

## 📁 Project Structure

```text
MIMII-Industrial-AI/
│
├── dashboard/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│   └── uploads/
│
├── data/
│   └── processed/
│       └── mel.npz
│
├── results/
│   ├── active_learning_final/
│   ├── models/
│   └── experiment results
│
├── src/
│   ├── config.py
│   ├── features.py
│   ├── preprocess.py
│   ├── data.py
│   ├── model.py
│   ├── trainer.py
│   ├── agents.py
│   ├── active_learning.py
│   ├── analysis.py
│   ├── run_experiments.py
│   ├── run_agent_demo.py
│   ├── baseline.py
│   ├── metrics.py
│   └── xai.py
│
├── tests/
├── tools/
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/palak-surana/MIMII-Industrial-AI.git

cd MIMII-Industrial-AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Install a compatible PyTorch build depending on your hardware if required.

### Step 4: Prepare the Dataset

Download the MIMII dataset:

https://zenodo.org/records/3384388

Extract the required machine audio recordings into the `data/mimii/` directory.

### Step 5: Run the Flask Dashboard Locally

```bash
python -m dashboard.app
```

Open your browser:

http://127.0.0.1:5000

## 🔬 Experimental Workflow

### Experiment 1: Data Preprocessing and EDA

- Audio preprocessing
- Feature extraction
- Log-Mel spectrogram generation
- Exploratory Data Analysis

### Experiment 2: CNN-LSTM Model

- Train the CNN-LSTM architecture.
- Evaluate machine fault classification.
- Compare different audio window sizes.

### Experiment 3: Active Learning

Compare:
- Random Sampling
- Uncertainty-Based Sampling

Both strategies use the same initial labelled set and annotation budget.

### Experiment 4: Agentic Active Learning

A multi-agent system coordinates the active learning workflow, including sample selection and annotation decisions.

### Experiment 5: Model Evaluation

Evaluate using:

- Accuracy
- Precision
- Recall
- Specificity
- F1-Score
- AUROC

### Experiment 6: Explainable AI

Integrated Gradients is used to understand which input features influence model predictions.

## 📊 Evaluation and Research Focus

The main research question is:

**Does uncertainty-based active learning, managed by a multi-agent framework, achieve comparable cross-machine fault detection performance with fewer labelled audio segments than random sampling?**

The experiments compare performance at different annotation budgets.

The project reports measured results rather than assuming that uncertainty-based sampling always outperforms random sampling.

## 🌐 Web Dashboard

The Flask dashboard provides:

- Machine health monitoring
- Audio recording upload
- Audio preprocessing
- CNN-LSTM fault prediction
- Prediction probabilities
- Fault monitoring
- Analysis history
- Analysis report downloads

The application is deployed using Render and Gunicorn.

**Live Website:** https://mimii-industrial-ai.onrender.com

## ⚠️ Limitations

- The model is evaluated on the selected MIMII machine categories and experimental setup.
- Clip-level labels may not represent every individual audio window correctly.
- The validation machine is used for model selection and the active learning stopping rule.
- Real industrial deployment requires testing with additional machines and operating environments.
- Active learning performance depends on the dataset, initial labelled samples, and experimental settings.

## 👩‍💻 Author

**Palak Surana**

B.Tech – Data Science

MIT Academy of Engineering (MITAOE)

GitHub: https://github.com/palak-surana

---

## 📜 License

This project is intended for academic and research purposes.
