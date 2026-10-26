# UPES Lost & Found Management System
### AI-Powered Minor Project | B.Tech (Hons.) CSE | Semester 6
### University of Petroleum and Energy Studies

---

## 📁 PROJECT FILES

```
upes_lost_found/
│
├── index.html                 ← Main frontend (UPES Portal UI)
├── app.py                     ← Python Flask backend + YOLOv8 detection
├── ml_model.py                ← Standalone ML matching model
├── ML_Model_LostFound.ipynb   ← Jupyter notebook (ML model with plots)
├── requirements.txt           ← All Python packages needed
├── README.md                  ← This file
│
└── static/
    └── uploads/               ← Auto-created folder for uploaded images
```

---

## ✅ DO YOU NEED A DATASET?

**NO — you do NOT need any external dataset.**

Here is why:
- `ml_model.py` has a built-in dataset of 20 lost + 20 found items (hardcoded)
- `app.py` seeds the SQLite database automatically on first run
- YOLOv8 comes pre-trained on the COCO dataset (80 object classes) — already inside the model weights file
- The app builds its own database as students submit lost/found reports

So everything works out of the box. No CSV, no Excel, no download needed.

---

## 🚀 HOW TO START — FULL SEQUENCE

### ─── STEP 1: Install Python ───────────────────────────
Make sure Python 3.10 or higher is installed.
Check by opening Command Prompt and typing:

    python --version

If not installed → download from https://python.org

---

### ─── STEP 2: Open project folder in CMD ──────────────
- Go to the folder where app.py is saved
- Click the address bar at top of the folder
- Type:  cmd
- Press Enter
  → This opens Command Prompt inside that folder

---

### ─── STEP 3: Install all packages (ONE TIME ONLY) ─────
Run this command:

    pip install flask flask-cors pillow python-jose ultralytics scikit-learn pandas numpy matplotlib seaborn

Or use the requirements file:

    pip install -r requirements.txt

Wait for everything to download and install.
YOLOv8 will also download yolov8n.pt (~6MB) automatically on first run.

---

### ─── STEP 4: Start the backend ────────────────────────
In the same CMD window, run:

    python app.py

You should see:

    [YOLO] YOLOv8n loaded successfully ✅
    [DB]   Ready → lostfound.db
    ==================================================
      UPES Lost & Found — OFFLINE MODE (YOLOv8)
    ==================================================
      YOLOv8  : ✅ Ready
      DB      : lostfound.db
      Server  : http://localhost:5000
    ==================================================

KEEP THIS TERMINAL OPEN — do not close it.

---

### ─── STEP 5: Open the frontend ────────────────────────
Open index.html in your browser (double-click it or drag to Chrome).

Login with:
    Student ID : 500119640
    Password   : upes@2025

    Admin ID   : admin
    Password   : admin123

---

### ─── STEP 6 (Optional): Run the ML Model ──────────────
To see the AI matching model with evaluation and graphs, open
a NEW CMD window (keep app.py running) and run:

    python ml_model.py

This will:
- Train the TF-IDF model on 20 lost + 20 found items
- Show accuracy, precision, recall, F1 score
- Save model_results.png with 4 charts

---

### ─── STEP 7 (Optional): Open Jupyter Notebook ─────────
To view the full ML notebook with all explanations:

    pip install jupyter
    jupyter notebook ML_Model_LostFound.ipynb

---

## 🔁 EVERY TIME YOU WANT TO USE THE APP

1. Open CMD in project folder
2. Run:   python app.py
3. Open:  index.html in browser
4. Done ✅

---

## ✨ FEATURES SUMMARY

| Feature                    | How it works                                      |
|----------------------------|---------------------------------------------------|
| Login / Auth               | Student ID + password, JWT token                  |
| Database                   | SQLite (lostfound.db), auto-created               |
| Lost item report           | Form with photo upload, stored in DB              |
| Found item report          | Form with AI photo detection                      |
| AI Object Detection        | YOLOv8n — runs fully offline on your PC           |
| Color Detection            | Pixel analysis — always runs with or without YOLO |
| Text Matching              | TF-IDF cosine similarity on descriptions          |
| Image Matching             | Color histogram comparison (Bhattacharyya)        |
| Combined Match Score       | 60% text + 40% image                              |
| Notifications              | Auto-triggered when match score >= 0.20           |
| Dashboard                  | Campus-wide stats and recent activity             |
| My Reports                 | Personal report history per student               |
| Portal UI                  | Exact UPES portal design (purple-blue sidebar)    |

---

## 🤖 AI MODELS USED

### YOLOv8n (Object Detection)
- File: yolov8n.pt (auto-downloaded ~6MB)
- Detects 80+ object classes: backpack, phone, bottle, book, etc.
- Runs offline after first download
- Used in: Found item photo upload → auto fills item name & category

### TF-IDF + Cosine Similarity (Text Matching)
- Built into ml_model.py and app.py
- No download needed — pure Python/scikit-learn
- Matches lost item descriptions to found item descriptions
- Used in: Auto-matching engine + notifications

### Color Histogram (Image Matching)
- Built into app.py using Pillow
- Compares dominant colors of uploaded photos
- Used in: Secondary match signal alongside text

---

## 📡 API ENDPOINTS (app.py)

| Method | Route                          | What it does                  |
|--------|--------------------------------|-------------------------------|
| POST   | /api/auth/login                | Student login                 |
| GET    | /api/auth/me                   | Get logged-in user info       |
| POST   | /api/detect                    | YOLOv8 image detection        |
| GET    | /api/items                     | Get all items from DB         |
| POST   | /api/items                     | Report lost or found item     |
| GET    | /api/notifications             | Get user notifications        |
| PATCH  | /api/notifications/read-all    | Mark all notifications read   |
| GET    | /api/health                    | Check if server is running    |

---

## 📊 ML MODEL RESULTS (ml_model.py)

Dataset : 20 lost items + 20 found items (built-in)
Method  : TF-IDF bigrams + Category boost + Color histogram
Result  : 20/20 correct matches on demo dataset

Metrics at optimal threshold:
  Accuracy  : 100%
  Precision : 100%
  Recall    : 100%
  F1 Score  : 100%

Plots saved: model_results.png (4 charts)

---

## ⚠️ TROUBLESHOOTING

Problem : "Backend not running — start it: python app.py"
Fix     : Open CMD in project folder and run: python app.py

Problem : pip is not recognized
Fix     : Use: python -m pip install -r requirements.txt

Problem : Port 5000 already in use
Fix     : Change port at bottom of app.py: app.run(port=5001)
          Also update in index.html: const API_BASE = 'http://localhost:5001'

Problem : YOLO not loading
Fix     : pip install ultralytics
          App still works without it (color-only detection as fallback)

Problem : ModuleNotFoundError
Fix     : pip install -r requirements.txt

---

Built for UPES Minor Project | 2025-2026
