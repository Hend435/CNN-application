# 🌍 Intel Image Classification — CNN Application

A deep learning web application that classifies natural scene images into 6 categories using a fine-tuned **EfficientNetB0** model. The app features a **FastAPI** backend and a **Streamlit** frontend.

---

## 📌 Categories

| Label | Description |
|-----------|-------------|
| 🏢 Buildings | Urban structures and architecture |
| 🌲 Forest | Trees and natural woodland |
| 🧊 Glacier | Ice fields and frozen landscapes |
| 🏔️ Mountain | Mountain peaks and rocky terrain |
| 🌊 Sea | Oceans, lakes, and water bodies |
| 🛣️ Street | Roads and city streets |

---

## 🏗️ Project Structure

```
CNN-application/
├── model/
│   └── final_intel_model.weights.h5   # Pre-trained model weights
├── app.py                              # FastAPI backend server
├── ui.py                               # Streamlit frontend UI
├── requirements.txt                    # Python dependencies
└── .gitignore
```

---

## ⚙️ How It Works

1. The user uploads an image via the **Streamlit UI**.
2. The image is sent as a POST request to the **FastAPI** `/predict` endpoint.
3. The backend preprocesses the image (resizes to 224×224, normalizes using EfficientNet preprocessing).
4. The **EfficientNetB0** model (with frozen base layers + custom classification head) runs inference.
5. The predicted class and confidence score are returned and displayed in the UI.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Hend435/CNN-application.git
cd CNN-application
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Backend (FastAPI)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 4. Run the Frontend (Streamlit)

Open a **new terminal** and run:

```bash
streamlit run ui.py
```

The UI will open in your browser at `http://localhost:8501`.

---

## 🔌 API Reference

### `GET /`

Returns server status and available classes.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"]
}
```

### `POST /predict`

Accepts an image file and returns the predicted class with confidence.

**Request:** `multipart/form-data` with field `file` (JPG, JPEG, or PNG).

**Response:**
```json
{
  "success": true,
  "prediction": "mountain",
  "confidence": "97.43%",
  "confidence_score": 97.43
}
```

---

## 🧠 Model Architecture

- **Base model:** EfficientNetB0 (pre-trained, top layers removed, base frozen)
- **Input shape:** 224 × 224 × 3
- **Custom head:**
  - `GlobalAveragePooling2D`
  - `Dropout(0.5)`
  - `Dense(6, activation='softmax')`
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Weights file:** `model/final_intel_model.weights.h5`

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow >= 2.15.0` | Deep learning framework |
| `numpy >= 1.23.5` | Numerical computation |
| `Pillow >= 10.0.0` | Image processing |
| `fastapi >= 0.100.0` | REST API backend |
| `uvicorn >= 0.23.0` | ASGI server |
| `python-multipart >= 0.0.6` | File upload support |
| `streamlit >= 1.25.0` | Web UI frontend |
| `requests >= 2.31.0` | HTTP client for UI |


