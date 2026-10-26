"""
============================================================
UPES Lost & Found Management System -- Backend API
============================================================
Stack   : Python 3.10+, Flask, SQLite, CLIP, JWT Auth
Run     :
    pip install flask flask-cors pillow python-jose
    pip install torch torchvision transformers
    pip install opencv-python-headless easyocr spacy
    python -m spacy download en_core_web_sm
    python app.py

API     : http://localhost:5000

CLIP model (~400MB) downloads automatically on first run.
No internet needed after that.
============================================================
"""

from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
import sqlite3, os, base64, hashlib, hmac, json, time, re, smtplib
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO
from PIL import Image
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Optional JWT ─────────────────────────────────────────
try:
    from jose import jwt as jose_jwt
    USE_JOSE = True
except ImportError:
    USE_JOSE = False

# ── CLIP (OpenAI vision-language model) ──────────────────
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    print("[CLIP] Loading model (downloads ~400MB on first run)...")
    _clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    _clip_model.eval()
    CLIP_READY = True
    print("[CLIP] CLIP ViT-B/32 loaded successfully")
except Exception as e:
    CLIP_READY = False
    print(f"[CLIP] Not available: {e}")
    print("[CLIP] Install with: pip install torch torchvision transformers")

# ── Module 1: GrabCut background removal ─────────────────
try:
    import cv2
    import numpy as np
    CV2_READY = True
    print("[GrabCut] OpenCV loaded ✅")
except ImportError:
    CV2_READY = False
    print("[GrabCut] opencv not found. pip install opencv-python-headless")

# ── Module 2: EasyOCR ────────────────────────────────────
try:
    import easyocr
    _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    OCR_READY = True
    print("[OCR] EasyOCR loaded ✅")
except Exception as e:
    OCR_READY = False
    print(f"[OCR] EasyOCR not available: {e}. pip install easyocr")

# ── Module 3: SpaCy NER ──────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_READY = True
    print("[NER] SpaCy en_core_web_sm loaded ✅")
except Exception as e:
    SPACY_READY = False
    print(f"[NER] SpaCy not available: {e}. pip install spacy && python -m spacy download en_core_web_sm")

# ── Gmail credentials (Module 2) ─────────────────────────
GMAIL_USER     = os.environ.get("GMAIL_USER", "tilaky222@gmail.com")
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD", "uiznqkjnhmlonvop")

# ─────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

SECRET_KEY    = os.environ.get("SECRET_KEY", "upes_lostfound_secret_2025")
DB_PATH       = os.environ.get("DB_PATH", "lostfound.db")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ════════════════════════════════════════════════════════
# CLIP CAMPUS ITEM LABELS
# ════════════════════════════════════════════════════════

CLIP_LABELS = [
    # Keys
    "car key fob",
    "bike key",
    "house door key",
    "bunch of keys on keyring",
    # Electronics
    "Apple Watch smartwatch",
    "Samsung Galaxy Watch",
    "wristwatch analog watch",
    "Apple AirPods earbuds",
    "wireless earbuds charging case",
    "over-ear headphones",
    "mobile phone smartphone",
    "laptop computer",
    "tablet iPad",
    "power bank charger",
    "USB cable charger",
    "earphones wired",
    # Bags
    "backpack school bag",
    "handbag purse",
    "tote bag",
    "suitcase luggage",
    "gym bag duffel bag",
    # Cards & Wallets
    "wallet leather wallet",
    "student ID card",
    "credit card debit card",
    # Clothing
    "jacket hoodie sweatshirt",
    "t-shirt",
    "scarf",
    "cap hat",
    "umbrella",
    # Accessories
    "prescription glasses spectacles",
    "sunglasses",
    "wristband bracelet",
    "ring jewellery",
    "necklace jewellery",
    # Books
    "textbook notebook",
    "diary planner",
    "pen pencil stationery",
    # Other campus items
    "water bottle",
    "lunch box tiffin",
    "sports ball",
    "racket sports equipment",
]

def clip_label_to_category(label: str) -> str:
    label = label.lower()
    if any(k in label for k in ["key", "keyring", "fob"]):
        return "Keys & Cards"
    if any(k in label for k in ["watch", "earbuds", "airpods", "headphones",
                                  "phone", "laptop", "tablet", "power bank",
                                  "charger", "earphones", "usb"]):
        return "Electronics"
    if any(k in label for k in ["backpack", "bag", "handbag", "purse",
                                  "suitcase", "tote", "luggage"]):
        return "Bags & Accessories"
    if any(k in label for k in ["wallet", "id card", "credit card", "debit card"]):
        return "Keys & Cards"
    if any(k in label for k in ["jacket", "hoodie", "t-shirt", "scarf",
                                  "cap", "hat", "umbrella", "sweatshirt"]):
        return "Clothing"
    if any(k in label for k in ["glasses", "spectacles", "sunglasses",
                                  "bracelet", "ring", "necklace", "wristband"]):
        return "Other"
    if any(k in label for k in ["textbook", "notebook", "diary", "pen",
                                  "pencil", "stationery", "planner"]):
        return "Books & Stationery"
    return "Other"

def clip_label_to_friendly(label: str) -> str:
    mapping = {
        "car key fob"                    : "Car Key / Key Fob",
        "bike key"                       : "Bike Key",
        "house door key"                 : "House / Door Key",
        "bunch of keys on keyring"       : "Keys on Keyring",
        "apple watch smartwatch"         : "Apple Watch / Smartwatch",
        "samsung galaxy watch"           : "Samsung Galaxy Watch",
        "wristwatch analog watch"        : "Wristwatch",
        "apple airpods earbuds"          : "Apple AirPods",
        "wireless earbuds charging case" : "Wireless Earbuds",
        "over-ear headphones"            : "Headphones",
        "mobile phone smartphone"        : "Mobile Phone",
        "laptop computer"                : "Laptop",
        "tablet ipad"                    : "Tablet / iPad",
        "power bank charger"             : "Power Bank",
        "usb cable charger"              : "USB Charger / Cable",
        "earphones wired"                : "Earphones (Wired)",
        "backpack school bag"            : "Backpack",
        "handbag purse"                  : "Handbag / Purse",
        "tote bag"                       : "Tote Bag",
        "suitcase luggage"               : "Suitcase",
        "gym bag duffel bag"             : "Gym Bag",
        "wallet leather wallet"          : "Wallet",
        "student id card"                : "Student ID Card",
        "credit card debit card"         : "Bank / ID Card",
        "jacket hoodie sweatshirt"       : "Jacket / Hoodie",
        "t-shirt"                        : "T-Shirt",
        "scarf"                          : "Scarf",
        "cap hat"                        : "Cap / Hat",
        "umbrella"                       : "Umbrella",
        "prescription glasses spectacles": "Prescription Glasses",
        "sunglasses"                     : "Sunglasses",
        "wristband bracelet"             : "Wristband / Bracelet",
        "ring jewellery"                 : "Ring",
        "necklace jewellery"             : "Necklace",
        "textbook notebook"              : "Textbook / Notebook",
        "diary planner"                  : "Diary / Planner",
        "pen pencil stationery"          : "Pen / Stationery",
        "water bottle"                   : "Water Bottle",
        "lunch box tiffin"               : "Lunchbox / Tiffin",
        "sports ball"                    : "Sports Ball",
        "racket sports equipment"        : "Sports Racket",
    }
    return mapping.get(label.lower(), label.title())


# ════════════════════════════════════════════════════════
# MODULE 1 — AUTO BACKGROUND REMOVAL (GrabCut)
# ════════════════════════════════════════════════════════

def remove_background_grabcut(image_bytes: bytes) -> bytes:
    """
    Strip the background from a photo using OpenCV GrabCut.
    Runs entirely on CPU. Falls back silently if opencv is absent.
    """
    if not CV2_READY:
        return image_bytes
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        h, w     = img.shape[:2]
        margin_x = max(int(w * 0.10), 10)
        margin_y = max(int(h * 0.10), 10)
        rect     = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask      = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(img, mask, rect, bgd_model, fgd_model,
                    iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                           255, 0).astype(np.uint8)
        result  = img.copy()
        result[fg_mask == 0] = [255, 255, 255]

        success, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return image_bytes

        print("[GrabCut] Background removed ✅")
        return buf.tobytes()
    except Exception as e:
        print(f"[GrabCut] Skipped: {e}")
        return image_bytes


# ════════════════════════════════════════════════════════
# CLIP DETECTION FUNCTION  (GrabCut pre-processing built in)
# ════════════════════════════════════════════════════════

def detect_with_clip(image_bytes: bytes) -> dict:
    """
    Run GrabCut background removal, then CLIP classification.
    Returns item name, category, color, confidence, description.
    """
    # ── Module 1: strip background before CLIP ───────────
    image_bytes = remove_background_grabcut(image_bytes)

    img  = Image.open(BytesIO(image_bytes)).convert("RGB")
    W, H = img.size

    # ── Color detection ──────────────────────────────────
    cx, cy  = W // 2, H // 2
    crop_w  = max(W // 2, 40)
    crop_h  = max(H // 2, 40)
    crop    = img.crop((cx - crop_w//2, cy - crop_h//2,
                        cx + crop_w//2, cy + crop_h//2))
    small   = crop.resize((80, 80))
    pixels  = list(small.getdata())
    fg      = [p for p in pixels if not (p[0]>220 and p[1]>220 and p[2]>220)]
    if len(fg) < 20:
        fg = pixels
    avg_r = sum(p[0] for p in fg) / len(fg)
    avg_g = sum(p[1] for p in fg) / len(fg)
    avg_b = sum(p[2] for p in fg) / len(fg)

    def color_name(r, g, b):
        brightness = (r + g + b) / 3
        sat = max(r, g, b) - min(r, g, b)
        if brightness < 55:               return "Black"
        if brightness > 210 and sat < 25: return "White"
        if brightness > 170 and sat < 30: return "Silver"
        if r >= g and r >= b:
            if sat < 40: return "Grey"
            if g > 120:  return "Orange" if r > 180 else "Brown"
            return "Red"
        if g >= r and g >= b:
            if sat < 40: return "Grey"
            return "Green" if g > 120 else "Dark Green"
        if b >= r and b >= g:
            if sat < 40: return "Grey"
            if r > 100:  return "Purple"
            return "Blue" if b > 140 else "Dark Blue"
        if r > 150 and g > 120 and b < 90: return "Gold" if r > 180 else "Brown"
        if r > 150 and g > 150 and b < 80: return "Yellow"
        return "Grey"

    color = color_name(avg_r, avg_g, avg_b)

    # ── Fallback if CLIP not installed ───────────────────
    if not CLIP_READY:
        return {
            "item_name"   : f"{color} Item",
            "category"    : "Other",
            "color"       : color,
            "brand"       : "Unknown",
            "condition"   : "Fair",
            "confidence"  : "55%",
            "description" : (f"A {color.lower()} item was found on campus. "
                             "Install CLIP for AI detection: "
                             "pip install torch torchvision transformers"),
            "tags"        : [color.lower(), "item"],
            "yolo_classes": []
        }

    # ── Run CLIP ─────────────────────────────────────────
    try:
        inputs = _clip_processor(
            text=CLIP_LABELS, images=img,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs  = _clip_model(**inputs)
            logits   = outputs.logits_per_image[0]
            probs    = torch.softmax(logits, dim=0)
            top_idx  = int(probs.argmax())
            top_prob = float(probs[top_idx])
            conf_pct = int(top_prob * 100)
            top3_idx = probs.topk(3).indices.tolist()
            top3     = [(CLIP_LABELS[i], round(float(probs[i])*100, 1)) for i in top3_idx]

        best_label = CLIP_LABELS[top_idx]
        friendly   = clip_label_to_friendly(best_label)
        category   = clip_label_to_category(best_label)
        item_name  = f"{color} {friendly}"

        def condition(c):
            if c >= 60: return "Good"
            if c >= 35: return "Fair"
            return "Damaged"

        description = (
            f"A {color.lower()} {friendly.lower()} was found on the UPES campus. "
            f"The item appears to be in {condition(conf_pct)} condition. "
            f"Identified using CLIP AI vision model with {conf_pct}% confidence."
        )
        print(f"[CLIP] Top match: '{best_label}' ({conf_pct}%) | Top3: {top3}")

        return {
            "item_name"   : item_name,
            "category"    : category,
            "color"       : color,
            "brand"       : "Unknown",
            "condition"   : condition(conf_pct),
            "confidence"  : f"{conf_pct}%",
            "description" : description,
            "tags"        : [color.lower(), friendly.lower().split()[0],
                             category.lower().split()[0]],
            "yolo_classes": [best_label]
        }

    except Exception as e:
        print(f"[CLIP] Error: {e}")
        return {
            "item_name"   : f"{color} Item",
            "category"    : "Other",
            "color"       : color,
            "brand"       : "Unknown",
            "condition"   : "Fair",
            "confidence"  : "50%",
            "description" : f"A {color.lower()} item found on campus.",
            "tags"        : [color.lower(), "item"],
            "yolo_classes": []
        }


# ════════════════════════════════════════════════════════
# DATABASE SETUP
# ════════════════════════════════════════════════════════

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db: db.close()

def init_db():
    with app.app_context():
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            email      TEXT UNIQUE NOT NULL,
            phone      TEXT,
            password   TEXT NOT NULL,
            programme  TEXT,
            semester   TEXT,
            role       TEXT DEFAULT 'student',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            type        TEXT NOT NULL CHECK(type IN ('lost','found')),
            status      TEXT NOT NULL DEFAULT 'open'
                            CHECK(status IN ('open','matched','claimed')),
            name        TEXT NOT NULL,
            category    TEXT,
            location    TEXT NOT NULL,
            description TEXT,
            date_event  TEXT,
            image_path  TEXT,
            reported_by TEXT NOT NULL,
            claimed_by  TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(reported_by) REFERENCES users(id),
            FOREIGN KEY(claimed_by)  REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS matches (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            lost_id    INTEGER NOT NULL,
            found_id   INTEGER NOT NULL,
            score      REAL NOT NULL,
            method     TEXT DEFAULT 'text',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(lost_id)  REFERENCES items(id),
            FOREIGN KEY(found_id) REFERENCES items(id)
        );
        CREATE TABLE IF NOT EXISTS notifications (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            title      TEXT NOT NULL,
            body       TEXT NOT NULL,
            type       TEXT DEFAULT 'info',
            is_read    INTEGER DEFAULT 0,
            item_id    INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)

        def hp(pw): return hashlib.sha256(pw.encode()).hexdigest()
        for u in [
            ("500119640","Tilak Yadav","tilak.yadav@upes.ac.in","9876543210",
             hp("upes@2025"),"B.Tech (Hons.) CSE","Semester 6","student"),
            ("admin","Admin","admin@upes.ac.in","",hp("admin123"),"Administration","—","admin"),
        ]:
            try: db.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)", u)
            except sqlite3.IntegrityError: pass

        for it in [
            ("found","open","Black Nike Backpack","Bags & Accessories","Library 2nd floor",
             "Black Nike backpack with laptop compartment","2026-03-12",None,"500119640"),
            ("lost","open","iPhone 14 Blue Case","Electronics","Canteen 2nd floor",
             "iPhone 14 with blue cracked case","2026-03-12",None,"500119640"),
        ]:
            try: db.execute("""INSERT INTO items
                (type,status,name,category,location,description,date_event,image_path,reported_by)
                VALUES (?,?,?,?,?,?,?,?,?)""", it)
            except Exception: pass

        db.commit()
        db.close()
    print("[DB] Ready →", DB_PATH)


def init_extra_tables():
    """
    Create supplementary tables for the new modules.
    Safe to call multiple times — IF NOT EXISTS guards prevent duplication.
    Does NOT touch any existing table.
    """
    with app.app_context():
        db = sqlite3.connect(DB_PATH)
        db.executescript("""
        CREATE TABLE IF NOT EXISTS ocr_resolutions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id     INTEGER,
            sap_id      TEXT,
            ocr_text    TEXT,
            name_found  TEXT,
            email_sent  INTEGER DEFAULT 0,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(item_id) REFERENCES items(id)
        );
        CREATE TABLE IF NOT EXISTS social_ingestions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT,
            raw_text    TEXT NOT NULL,
            item_id     INTEGER,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(item_id) REFERENCES items(id)
        );
        """)
        db.commit()
        # Migration: add claimed_by column if it doesn't exist yet
        try:
            db.execute("ALTER TABLE items ADD COLUMN claimed_by TEXT")
            db.commit()
        except Exception:
            pass  # Column already exists
        db.close()
    print("[DB] Extra tables ready ✅")


# ════════════════════════════════════════════════════════
# AUTH HELPERS
# ════════════════════════════════════════════════════════

def make_token(uid, role):
    payload = {"sub": uid, "role": role, "exp": time.time() + 86400}
    if USE_JOSE:
        return jose_jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    data = json.dumps(payload)
    sig  = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return base64.b64encode(f"{data}|{sig}".encode()).decode()

def decode_token(token):
    try:
        if USE_JOSE:
            return jose_jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        raw  = base64.b64decode(token.encode()).decode()
        data, sig = raw.rsplit("|", 1)
        exp = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, exp): return None
        p = json.loads(data)
        return None if p["exp"] < time.time() else p
    except Exception: return None

def auth_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        token   = request.headers.get("Authorization","").replace("Bearer ","")
        payload = decode_token(token)
        if not payload: return jsonify({"error":"Unauthorized"}), 401
        g.user_id = payload["sub"]
        g.role    = payload.get("role","student")
        return f(*args, **kwargs)
    return wrapper

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()


# ════════════════════════════════════════════════════════
# FRONTEND ROUTE
# ════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(".", "index.html")


# ════════════════════════════════════════════════════════
# ORIGINAL API ROUTES
# ════════════════════════════════════════════════════════

@app.route("/api/detect", methods=["POST"])
def detect_image():
    data = request.get_json()
    if not data or not data.get("image_base64"):
        return jsonify({"error": "image_base64 required"}), 400
    try:
        result = detect_with_clip(base64.b64decode(data["image_base64"]))
        # Flag card detection so the frontend knows to submit with category "Keys & Cards"
        card_keywords = ["id card", "student id", "card", "wallet"]
        result["is_card"] = any(kw in result.get("item_name", "").lower()
                                for kw in card_keywords)
        if result["is_card"]:
            result["ocr_hint"] = ("ID card detected! Submit as a Found item — "
                                  "the system will automatically extract the student's "
                                  "details and notify them by email.")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    uid  = data.get("student_id","").strip()
    pw   = data.get("password","")
    if not uid or not pw: return jsonify({"error":"Credentials required"}), 400
    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not user or user["password"] != hash_pw(pw):
        return jsonify({"error":"Invalid credentials"}), 401
    return jsonify({
        "token": make_token(user["id"], user["role"]),
        "user" : {k: user[k] for k in ["id","name","email","phone","programme","semester","role"]}
    })

@app.route("/api/auth/register", methods=["POST"])
def register():
    data  = request.get_json()
    uid   = data.get("student_id","").strip()
    pw    = data.get("password","")
    name  = data.get("name","").strip()
    email = data.get("email","").strip()
    if not uid or not pw or not name or not email:
        return jsonify({"error":"student_id, password, name and email are required"}), 400
    db = get_db()
    try:
        db.execute("""INSERT INTO users (id,name,email,phone,password,programme,semester,role)
                      VALUES (?,?,?,?,?,?,?,?)""",
                   (uid, name, email, data.get("phone",""),
                    hash_pw(pw), data.get("programme","B.Tech CSE"),
                    data.get("semester","Semester 1"), "student"))
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error":"Student ID or email already registered"}), 409
    token = make_token(uid, "student")
    return jsonify({
        "token": token,
        "user" : {"id":uid,"name":name,"email":email,"phone":data.get("phone",""),
                  "programme":data.get("programme","B.Tech CSE"),
                  "semester":data.get("semester","Semester 1"),"role":"student"}
    }), 201

@app.route("/api/auth/me", methods=["GET"])
@auth_required
def get_me():
    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (g.user_id,)).fetchone()
    if not user: return jsonify({"error":"Not found"}), 404
    return jsonify({k: user[k] for k in ["id","name","email","phone","programme","semester","role"]})

@app.route("/api/items", methods=["GET"])
@auth_required
def get_items():
    db, p, cond = get_db(), [], []
    q = "SELECT i.*, u.name as reporter_name FROM items i JOIN users u ON i.reported_by=u.id"
    if request.args.get("type"):   cond.append("i.type=?");   p.append(request.args["type"])
    if request.args.get("status"): cond.append("i.status=?"); p.append(request.args["status"])
    if cond: q += " WHERE " + " AND ".join(cond)
    return jsonify([dict(r) for r in db.execute(q + " ORDER BY i.created_at DESC", p).fetchall()])

@app.route("/api/items", methods=["POST"])
@auth_required
def create_item():
    data = request.get_json()
    for f in ["type","name","location"]:
        if not data.get(f): return jsonify({"error":f"{f} required"}), 400
    image_path = None
    if data.get("image_base64"):
        try:
            fname = f"{int(time.time())}_{g.user_id}.jpg"
            with open(os.path.join(UPLOAD_FOLDER, fname),"wb") as fh:
                fh.write(base64.b64decode(data["image_base64"]))
            image_path = f"/static/uploads/{fname}"
        except Exception as e: print("Image save:", e)
    db  = get_db()
    cur = db.execute("""INSERT INTO items
        (type,status,name,category,location,description,date_event,image_path,reported_by)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (data["type"],"open",data["name"],data.get("category","Other"),data["location"],
         data.get("description",""),data.get("date_event",""),image_path,g.user_id))
    db.commit()
    item_id  = cur.lastrowid
    matches  = run_matching(item_id, data["type"], db)
    notify_matches(item_id, data["type"], matches, db)

    # ── Auto-OCR: if a FOUND card/ID is uploaded, run OCR immediately ──
    ocr_result = None
    is_card_category = data.get("category", "").lower() in (
        "keys & cards", "cards", "student id", "id card"
    )
    is_card_name = any(kw in data.get("name", "").lower()
                       for kw in ["id card", "student id", "card", "identity"])
    if (data["type"] == "found"
            and data.get("image_base64")
            and (is_card_category or is_card_name)
            and OCR_READY):
        try:
            img_bytes = base64.b64decode(data["image_base64"])
            img_pil   = Image.open(BytesIO(img_bytes)).convert("RGB")

            # ── Try all 4 rotations, pick the one with best OCR result ──
            def ocr_image(pil_img):
                """Run EasyOCR and return lines longer than 2 chars (filters barcode noise)."""
                if CV2_READY:
                    arr     = np.array(pil_img)
                    results = _ocr_reader.readtext(arr, detail=0)
                else:
                    tmp = "/tmp/_ocr_rot.jpg"
                    pil_img.save(tmp)
                    results = _ocr_reader.readtext(tmp, detail=0)
                # Filter out single chars and barcode-like tokens (all digits, len<3)
                lines = []
                for r in results:
                    r = str(r).strip()
                    if len(r) < 2:
                        continue
                    # skip pure digit tokens shorter than 5 (barcode fragments)
                    if r.isdigit() and len(r) < 5:
                        continue
                    lines.append(r)
                return lines

            best_lines = []
            best_score = 0
            for angle in [0, 90, 180, 270]:
                rotated = img_pil.rotate(angle, expand=True)
                lines   = ocr_image(rotated)
                # Score = number of meaningful tokens (words/numbers)
                score   = sum(1 for l in lines if len(l) > 3)
                print(f"[OCR] Rotation {angle}°: {len(lines)} lines, score={score} → {lines[:5]}")
                if score > best_score:
                    best_score = score
                    best_lines = lines

            ocr_lines    = best_lines
            sap_id, name = extract_sap_and_name(ocr_lines)

            # ── Construct UPES email from card details ───────────────────
            # Format: FirstName.last6ofSAPID@stu.upes.ac.in
            # Example: Tilak Yadav + SAP 500119640 → Tilak.119640@stu.upes.ac.in
            # Works even if the student is NOT registered on the portal.
            constructed_email = None
            if sap_id and name:
                first_name        = name.strip().split()[0].capitalize()
                last6 = sap_id[-6:] if sap_id.startswith("500") else sap_id[-5:]
                constructed_email = f"{first_name}.{last6}@stu.upes.ac.in"

            # ── Also check portal DB (for verified email if registered) ──
            owner      = lookup_user_by_sap_or_name(sap_id, name, db)
            owner_dict = dict(owner) if owner else {}

            # Prefer verified DB email, fall back to constructed UPES email
            final_email = owner_dict.get("email") or constructed_email
            final_name  = owner_dict.get("name")  or name or "Student"
            final_sap   = sap_id or owner_dict.get("id", "N/A")

            if final_email:
                subject   = "📢 UPES Lost & Found — Your ID Card Has Been Found!"
                body_html = f"""
                <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;
                            border:1px solid #e0d6f7;border-radius:8px;overflow:hidden;">
                  <div style="background:#6c3a9f;padding:20px 24px;">
                    <h2 style="color:#fff;margin:0;">UPES Lost &amp; Found Portal</h2>
                    <p style="color:#e0d6f7;margin:4px 0 0;">AI-Powered Campus System</p>
                  </div>
                  <div style="padding:24px;">
                    <p>Dear <strong>{final_name}</strong>,</p>
                    <p>
                      Great news! Your student ID card has been found on campus
                      by a fellow student and reported on our portal.
                    </p>
                    <table style="border-collapse:collapse;width:100%;font-size:14px;
                                  margin:16px 0;border:1px solid #e0d6f7;border-radius:6px;">
                      <tr style="background:#f5f0ff;">
                        <td style="padding:10px 14px;font-weight:bold;width:40%;">Your Name</td>
                        <td style="padding:10px 14px;">{final_name}</td>
                      </tr>
                      <tr>
                        <td style="padding:10px 14px;font-weight:bold;">SAP ID</td>
                        <td style="padding:10px 14px;">{final_sap}</td>
                      </tr>
                      <tr style="background:#f5f0ff;">
                        <td style="padding:10px 14px;font-weight:bold;">Found At</td>
                        <td style="padding:10px 14px;">{data.get("location", "Campus")}</td>
                      </tr>
                      <tr>
                        <td style="padding:10px 14px;font-weight:bold;">Reported On</td>
                        <td style="padding:10px 14px;">{datetime.now().strftime("%d %b %Y, %I:%M %p")}</td>
                      </tr>
                      <tr style="background:#f5f0ff;">
                        <td style="padding:10px 14px;font-weight:bold;">Portal Item ID</td>
                        <td style="padding:10px 14px;">#{item_id}</td>
                      </tr>
                    </table>
                    <p>
                      Please visit the
                      <a href="http://localhost:5000" style="color:#6c3a9f;font-weight:bold;">
                        UPES Lost &amp; Found Portal
                      </a>
                      or collect your card from the
                      <strong>campus security office</strong> at your earliest convenience.
                    </p>
                    <p style="color:#888;font-size:12px;margin-top:24px;
                              border-top:1px solid #eee;padding-top:12px;">
                      Automated message from UPES Lost &amp; Found AI System.<br>
                      Sent to: {final_email}
                    </p>
                  </div>
                </div>
                """
                email_sent = send_gmail(final_email, subject, body_html)

                # In-app notification (only if student is registered on portal)
                if owner:
                    db.execute(
                        "INSERT INTO notifications (user_id,title,body,type,item_id) VALUES (?,?,?,?,?)",
                        (owner_dict["id"],
                         "📢 Your ID Card Was Found!",
                         f"Your student ID card (SAP: {final_sap}) was found at "
                         f"{data.get('location', 'campus')}. "
                         + ("Email sent to your college ID." if email_sent
                            else "Please collect it from the security office."),
                         "match", item_id)
                    )

                # Log to ocr_resolutions table
                db.execute(
                    "INSERT INTO ocr_resolutions (item_id,sap_id,ocr_text,name_found,email_sent) VALUES (?,?,?,?,?)",
                    (item_id, final_sap, json.dumps(ocr_lines), final_name, int(email_sent))
                )
                db.commit()

                ocr_result = {
                    "owner_name"        : final_name,
                    "owner_email"       : final_email,
                    "sap_id"            : final_sap,
                    "email_constructed" : not bool(owner_dict.get("email")),
                    "email_sent"        : email_sent
                }
                print(f"[OCR-AUTO] Owner: {final_name} | Email: {final_email} | Sent: {email_sent}")
            else:
                print(f"[OCR-AUTO] Could not extract name/SAP. OCR lines: {ocr_lines}")

        except Exception as e:
            print(f"[OCR-AUTO] Failed: {e}")

    resp = {"id": item_id, "matches": len(matches), "message": "Reported successfully"}
    if ocr_result:
        resp["ocr_auto"] = ocr_result
        resp["message"]  = (
            f"Reported successfully. ID card owner identified: {ocr_result['owner_name']} "
            f"({ocr_result['owner_email']}) — "
            f"{'Email sent ✅' if ocr_result['email_sent'] else 'In-app notification sent.'}"
        )
    return jsonify(resp), 201

@app.route("/api/notifications", methods=["GET"])
@auth_required
def get_notifications():
    rows = get_db().execute(
        "SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC",(g.user_id,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/notifications/read-all", methods=["PATCH"])
@auth_required
def mark_all_read():
    db = get_db()
    db.execute("UPDATE notifications SET is_read=1 WHERE user_id=?",(g.user_id,))
    db.commit()
    return jsonify({"message":"Done"})

@app.route("/api/notifications/clear-all", methods=["DELETE"])
@auth_required
def clear_all_notifications():
    db = get_db()
    db.execute("DELETE FROM notifications WHERE user_id=?",(g.user_id,))
    db.commit()
    return jsonify({"message":"All notifications cleared"})

@app.route("/api/items/<int:item_id>/claim", methods=["PATCH"])
@auth_required
def claim_item(item_id):
    """Mark a found item as claimed. Any logged-in user can claim."""
    db   = get_db()
    item = db.execute("SELECT * FROM items WHERE id=?",(item_id,)).fetchone()
    if not item:
        return jsonify({"error":"Item not found"}), 404
    item = dict(item)
    if item["status"] == "claimed":
        return jsonify({"error":"Item is already claimed"}), 409
    if item["type"] != "found":
        return jsonify({"error":"Only found items can be claimed"}), 400

    db.execute("UPDATE items SET status='claimed', claimed_by=? WHERE id=?",
               (g.user_id, item_id))
    # Notify the finder
    db.execute(
        "INSERT INTO notifications (user_id,title,body,type,item_id) VALUES (?,?,?,?,?)",
        (item["reported_by"],
         "✅ Your Found Item Has Been Claimed",
         f'"{item["name"]}" found at {item["location"]} has been claimed by its owner.',
         "info", item_id)
    )
    db.commit()
    return jsonify({"message":"Item successfully claimed"})

@app.route("/api/items/<int:item_id>", methods=["DELETE"])
@auth_required
def delete_item(item_id):
    """Admin-only: permanently remove an item from the database."""
    if g.role != "admin":
        return jsonify({"error":"Admin access required"}), 403
    db   = get_db()
    item = db.execute("SELECT * FROM items WHERE id=?",(item_id,)).fetchone()
    if not item:
        return jsonify({"error":"Item not found"}), 404
    # Also delete associated notifications and matches
    db.execute("DELETE FROM notifications WHERE item_id=?",(item_id,))
    db.execute("DELETE FROM matches WHERE lost_id=? OR found_id=?",(item_id,item_id))
    db.execute("DELETE FROM items WHERE id=?",(item_id,))
    db.commit()
    return jsonify({"message":"Item deleted successfully"})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status"      : "ok",
        "clip_ready"  : CLIP_READY,
        "grabcut_ready": CV2_READY,
        "ocr_ready"   : OCR_READY,
        "ner_ready"   : SPACY_READY,
        "time"        : datetime.now().isoformat()
    })

# ════════════════════════════════════════════════════════
# FORGOT / RESET PASSWORD
# ════════════════════════════════════════════════════════
# Stores reset tokens in-memory (good enough for local/demo use).
# Token is a 6-digit code valid for 15 minutes.

import random
_reset_tokens = {}   # { sap_id: { "token": "123456", "expires": timestamp } }

def build_upes_email(sap_id: str, name: str) -> str:
    first = name.strip().split()[0].capitalize()
    if sap_id.startswith("500"):
        digits = sap_id[-6:]   # e.g. 500119640 → 119640
    else:
        digits = sap_id[-5:]   # e.g. 590015482 → 15482
    return f"{first}.{digits}@stu.upes.ac.in"


@app.route("/api/auth/forgot_password", methods=["POST"])
def forgot_password():
    """
    Step 1: User submits their SAP ID.
    System looks them up, constructs their UPES email,
    sends a 6-digit reset code, and stores it for 15 minutes.
    """
    data   = request.get_json()
    sap_id = (data.get("sap_id") or "").strip()

    if not sap_id or len(sap_id) != 9:
        return jsonify({"error": "Valid 9-digit SAP ID required"}), 400

    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (sap_id,)).fetchone()

    if not user:
        # Student not registered — still try to email their UPES address
        # but we need at least a name guess; derive from SAP only if impossible
        # Return error so they know to register first
        return jsonify({"error": "SAP ID not found. Please register on the portal first."}), 404

    user  = dict(user)
    email = user.get("email") or build_upes_email(sap_id, user["name"])

    # Generate 6-digit code
    code    = str(random.randint(100000, 999999))
    expires = time.time() + 900   # 15 minutes

    _reset_tokens[sap_id] = {"token": code, "expires": expires}
    print(f"[RESET] Code for {sap_id}: {code}")   # visible in terminal as backup

    subject   = "🔑 UPES Lost & Found — Password Reset Code"
    body_html = f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;
                border:1px solid #e0d6f7;border-radius:8px;overflow:hidden;">
      <div style="background:#6c3a9f;padding:18px 24px;">
        <h2 style="color:#fff;margin:0;font-size:1.1rem;">UPES Lost &amp; Found Portal</h2>
      </div>
      <div style="padding:24px;">
        <p>Dear <strong>{user['name']}</strong>,</p>
        <p>You requested a password reset for your Lost &amp; Found portal account.</p>
        <div style="background:#f5f0ff;border-radius:8px;padding:18px;text-align:center;margin:20px 0;">
          <div style="font-size:.8rem;color:#6c3a9f;font-weight:600;margin-bottom:6px;">YOUR RESET CODE</div>
          <div style="font-size:2.2rem;font-weight:800;color:#3d1f8c;letter-spacing:.3em;">{code}</div>
          <div style="font-size:.72rem;color:#888;margin-top:6px;">Valid for 15 minutes</div>
        </div>
        <p>Enter this code on the portal to set your new password.</p>
        <p>If you did not request this, ignore this email — your password will not change.</p>
        <p style="color:#888;font-size:.72rem;margin-top:20px;border-top:1px solid #eee;padding-top:12px;">
          Automated message from UPES Lost &amp; Found AI System.
        </p>
      </div>
    </div>
    """
    email_sent = send_gmail(email, subject, body_html)

    if not email_sent:
        # If Gmail not configured, print code to terminal so developer can share it
        print(f"[RESET] ⚠️  Gmail not configured. Code for {user['name']}: {code}")

    return jsonify({
        "email"     : email,
        "email_sent": email_sent,
        "message"   : f"Reset code sent to {email}" if email_sent else
                      f"Gmail not configured — check terminal for code."
    })


@app.route("/api/auth/reset_password", methods=["POST"])
def reset_password():
    """
    Step 2: User submits SAP ID + 6-digit code + new password.
    Validates code, updates password hash in DB.
    """
    data     = request.get_json()
    sap_id   = (data.get("sap_id") or "").strip()
    token    = (data.get("token") or "").strip()
    new_pw   = data.get("new_password", "")

    if not sap_id or not token or not new_pw:
        return jsonify({"error": "sap_id, token and new_password are required"}), 400

    if len(new_pw) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    record = _reset_tokens.get(sap_id)
    if not record:
        return jsonify({"error": "No reset request found. Please request a new code."}), 400

    if time.time() > record["expires"]:
        del _reset_tokens[sap_id]
        return jsonify({"error": "Reset code has expired. Please request a new one."}), 400

    if record["token"] != token:
        return jsonify({"error": "Invalid reset code. Please check and try again."}), 400

    # Update password in DB
    db = get_db()
    db.execute("UPDATE users SET password=? WHERE id=?",
               (hashlib.sha256(new_pw.encode()).hexdigest(), sap_id))
    db.commit()
    del _reset_tokens[sap_id]

    print(f"[RESET] Password updated for SAP ID: {sap_id}")
    return jsonify({"message": "Password reset successfully. You can now log in."})


# ════════════════════════════════════════════════════════
# MATCHING ENGINE
# ════════════════════════════════════════════════════════

def text_similarity(a, b):
    def tokens(x):
        return set(w for w in
            f"{x.get('name','')} {x.get('description','')} {x.get('category','')}".lower().split()
            if len(w) > 2)
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def image_color_similarity(path_a, path_b):
    import math
    try:
        def hist(p):
            h = Image.open(p.lstrip("/")).convert("RGB").resize((64,64)).histogram()
            t = sum(h); return [v/t for v in h]
        return round(min(sum(math.sqrt(a*b) for a,b in zip(hist(path_a),hist(path_b))),1.0),3)
    except: return 0.0

def run_matching(new_id, item_type, db):
    new   = dict(db.execute("SELECT * FROM items WHERE id=?",(new_id,)).fetchone())
    opp   = "found" if item_type=="lost" else "lost"
    cands = [dict(r) for r in db.execute(
        "SELECT * FROM items WHERE type=? AND status='open'",(opp,)).fetchall()]
    results = []
    for c in cands:
        t     = text_similarity(new, c)
        i     = image_color_similarity(new["image_path"], c["image_path"]) \
                if new.get("image_path") and c.get("image_path") else 0.0
        score = round(t*0.6 + i*0.4, 3)
        if score >= 0.20:
            lid, fid = (new_id, c["id"]) if item_type=="lost" else (c["id"], new_id)
            try:
                db.execute("INSERT INTO matches (lost_id,found_id,score,method) VALUES (?,?,?,?)",
                           (lid, fid, score, "text+image" if i>0 else "text"))
                db.commit()
                results.append({"item":c,"score":score})
            except: pass
    return sorted(results, key=lambda x: x["score"], reverse=True)[:5]

def notify_matches(item_id, item_type, matches, db):
    if not matches: return
    item = dict(db.execute("SELECT * FROM items WHERE id=?",(item_id,)).fetchone())
    for m in matches[:3]:
        c = m["item"]
        if item_type == "found":
            title = "📦 A Found Item May Match Your Lost Report"
            body  = f'"{item["name"]}" found at {item["location"]} may match your lost "{c["name"]}".'
        else:
            title = "🔍 Someone Lost an Item Similar to What You Found"
            body  = f'"{item["name"]}" lost near {item["location"]} may match "{c["name"]}" you found.'
        db.execute("INSERT INTO notifications (user_id,title,body,type,item_id) VALUES (?,?,?,?,?)",
                   (c["reported_by"], title, body, "match", item_id))
    db.commit()


# ════════════════════════════════════════════════════════
# MODULE 2 — MULTIMODAL OCR + GMAIL AUTO-NOTIFICATION
# POST /api/ocr_card
# ════════════════════════════════════════════════════════

def send_gmail(to_email: str, subject: str, body_html: str) -> bool:
    """Send HTML email via Gmail SMTP. Returns True on success."""
    if not GMAIL_USER or not GMAIL_PASSWORD:
        print("[Gmail] Credentials not set — skipping send.")
        return False
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_email
        msg.attach(MIMEText(body_html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())
        print(f"[Gmail] Sent to {to_email} ✅")
        return True
    except Exception as e:
        print(f"[Gmail] Failed: {e}")
        return False


def extract_sap_and_name(ocr_lines: list) -> tuple:
    """
    Robust extraction of UPES SAP ID and student Name from raw OCR lines.

    Handles:
    - Noisy OCR (spaces inside numbers, O vs 0 confusion)
    - ALL-CAPS names (common on printed ID cards)
    - Name on same line as label ("Name: Tilak Yadav") or next line
    - SAP ID with spaces ("500 119 640") or OCR noise ("5OO119640")
    - Fallback: name-only DB lookup when SAP ID is unreadable

    Returns (sap_id: str | None, name: str | None)
    """
    sap_id = None
    name   = None

    # ── Clean up OCR lines ───────────────────────────────
    # Replace common OCR confusions: O→0, I→1, l→1 inside digit strings
    def clean_line(line):
        # Normalise common OCR digit mistakes in number-like strings
        return re.sub(r'(?<=[0-9OIl])[OIl](?=[0-9OIl])',
                      lambda m: {'O':'0','I':'1','l':'1'}.get(m.group(),'0'),
                      line)

    cleaned   = [clean_line(l) for l in ocr_lines]
    full_text = " ".join(cleaned)
    full_orig = " ".join(ocr_lines)   # keep original for name extraction

    # ── SAP ID: 9-digit number starting with 5 ──────────
    # Also catches numbers with spaces inside, e.g. "500 119 640"
    sap_patterns = [
        re.compile(r'\b(5\d{8})\b'),                          # clean 9-digit
        re.compile(r'\b(5[\d ]{9,12})\b'),                    # with spaces
        re.compile(r'(5[0-9O]{8})'),                          # OCR noise
    ]
    for pat in sap_patterns:
        m = pat.search(full_text)
        if m:
            candidate = re.sub(r'\s', '', m.group(1))         # strip spaces
            candidate = candidate.replace('O', '0').replace('I', '1').replace('l', '1')
            if len(candidate) == 9 and candidate.startswith('5'):
                sap_id = candidate
                break

    # ── Name extraction ──────────────────────────────────
    # Pattern 1: Title Case  "Tilak Yadav" or "Tilak Kumar Yadav"
    title_pat = re.compile(
        r'\b([A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20}){1,3})\b'
    )
    # Pattern 2: ALL CAPS  "TILAK YADAV"  (printed ID cards)
    caps_pat  = re.compile(
        r'\b([A-Z]{2,20}(?:\s[A-Z]{2,20}){1,3})\b'
    )

    # Words to skip (common non-name tokens on ID cards)
    SKIP = {
        "UPES", "UNIVERSITY", "PETROLEUM", "ENERGY", "STUDIES",
        "STUDENT", "NAME", "SAP", "ID", "CARD", "ENROLMENT",
        "PROGRAMME", "BATCH", "SEMESTER", "VALID", "THROUGH",
        "DEHRADUN", "INDIA", "UTTARAKHAND", "COLLEGE",
        "B.TECH", "BTECH", "MBA", "MCA", "PHD", "M.TECH",
        "HONS", "CSE", "ECE", "ME", "CE", "EE",
    }

    def is_valid_name(candidate: str) -> bool:
        parts = candidate.strip().split()
        if len(parts) < 2 or len(parts) > 4:
            return False
        for p in parts:
            if p.upper() in SKIP:
                return False
            if not re.match(r'^[A-Za-z]{2,20}$', p):
                return False
        return True

    def try_extract_name(text_block: str) -> str | None:
        # Try Title Case first
        for m in title_pat.finditer(text_block):
            if is_valid_name(m.group(1)):
                return m.group(1)
        # Try ALL CAPS → convert to Title Case
        for m in caps_pat.finditer(text_block):
            candidate = m.group(1)
            parts = candidate.split()
            if all(p.upper() not in SKIP for p in parts) and len(parts) >= 2:
                return " ".join(p.capitalize() for p in parts)
        return None

    # Strategy A: look for "Name" label and grab what follows
    for i, line in enumerate(ocr_lines):
        if re.search(r'\bname\b', line, re.IGNORECASE):
            # Remove the "Name" / "Name:" prefix and check rest of line
            rest = re.sub(r'(?i)\bname\s*[:\-]?\s*', '', line).strip()
            candidate = try_extract_name(rest)
            if candidate and is_valid_name(candidate):
                name = candidate
                break
            # Try next line
            if i + 1 < len(ocr_lines):
                candidate = try_extract_name(ocr_lines[i + 1])
                if candidate and is_valid_name(candidate):
                    name = candidate
                    break

    # Strategy B: scan all lines for a valid name
    if not name:
        for line in ocr_lines:
            candidate = try_extract_name(line)
            if candidate and is_valid_name(candidate):
                name = candidate
                break

    # Strategy C: full-text fallback
    if not name:
        name = try_extract_name(full_orig)

    print(f"[OCR] Parsed → SAP: {sap_id}  Name: {name}")
    return sap_id, name


def lookup_user_by_sap_or_name(sap_id, name, db):
    """
    Find a user in the DB using SAP ID first, then name fuzzy match.
    Returns a sqlite3.Row or None.
    """
    if sap_id:
        user = db.execute("SELECT * FROM users WHERE id=?", (sap_id,)).fetchone()
        if user:
            return user

    if name:
        # Exact name match (case-insensitive)
        user = db.execute(
            "SELECT * FROM users WHERE LOWER(name)=?", (name.lower(),)
        ).fetchone()
        if user:
            return user

        # Partial: first word of name (first name match)
        first = name.split()[0].lower()
        users = db.execute(
            "SELECT * FROM users WHERE LOWER(name) LIKE ?", (f"{first}%",)
        ).fetchall()
        if len(users) == 1:          # only return if unambiguous
            return users[0]

    return None


@app.route("/api/ocr_card", methods=["POST"])
@auth_required
def ocr_card():
    """
    OCR a card image, identify the student via SAP ID, and auto-notify them.

    Body (JSON):
        image_base64 : str  — base64-encoded image of the ID card
        item_id      : int  — (optional) found item ID this card belongs to
    """
    data = request.get_json()
    if not data or not data.get("image_base64"):
        return jsonify({"error": "image_base64 required"}), 400

    item_id = data.get("item_id")

    try:
        img_bytes = base64.b64decode(data["image_base64"])
        img_pil   = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    if not OCR_READY:
        return jsonify({
            "error"      : "EasyOCR not installed. pip install easyocr",
            "ocr_text"   : [], "sap_id": None, "name": None,
            "user_found" : False, "email_sent": False,
            "message"    : "OCR unavailable"
        }), 503

    # ── Run OCR ──────────────────────────────────────────
    try:
        if CV2_READY:
            img_np  = np.array(img_pil)
            results = _ocr_reader.readtext(img_np, detail=0)
        else:
            tmp = "/tmp/_ocr_tmp.jpg"
            img_pil.save(tmp)
            results = _ocr_reader.readtext(tmp, detail=0)

        ocr_lines = [str(r).strip() for r in results if str(r).strip()]
        print(f"[OCR] {len(ocr_lines)} lines: {ocr_lines}")
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500

    # ── Parse SAP ID + Name ──────────────────────────────
    sap_id, name = extract_sap_and_name(ocr_lines)

    if not sap_id and not name:
        return jsonify({
            "ocr_text"  : ocr_lines,
            "sap_id"    : None, "name": None,
            "user_found": False, "email_sent": False,
            "message"   : "Could not extract SAP ID or Name from image. "
                          "Ensure the card is well-lit and fully visible."
        })

    # ── Look up user by SAP ID first, then by name ───────
    db   = get_db()
    user = lookup_user_by_sap_or_name(sap_id, name, db)

    if not user:
        return jsonify({
            "ocr_text"  : ocr_lines,
            "sap_id"    : sap_id, "name": name,
            "user_found": False, "email_sent": False,
            "message"   : (
                f"Extracted SAP ID: {sap_id or 'N/A'}, Name: {name or 'N/A'}. "
                f"No matching student found in the system."
            )
        })

    user = dict(user)

    # ── Send Gmail ───────────────────────────────────────
    subject   = "📢 UPES Lost & Found — Your ID Card Has Been Found!"
    body_html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
      <h2 style="color:#6c3a9f;">UPES Lost &amp; Found Portal</h2>
      <p>Dear <strong>{user['name']}</strong>,</p>
      <p>
        Good news! An item belonging to you
        <strong>(SAP ID: {sap_id})</strong>
        has been found on campus and logged in our system.
      </p>
      <table style="border-collapse:collapse;width:100%;">
        <tr><td style="padding:6px;font-weight:bold;">Item ID</td>
            <td style="padding:6px;">{item_id or 'N/A'}</td></tr>
        <tr style="background:#f5f0ff;">
            <td style="padding:6px;font-weight:bold;">Your SAP ID</td>
            <td style="padding:6px;">{sap_id}</td></tr>
      </table>
      <p>Please visit the UPES Lost &amp; Found Portal or contact the campus security office to claim your item.</p>
      <p style="color:#888;font-size:12px;">Automated message from UPES Lost &amp; Found AI System.</p>
    </div>
    """
    email_sent = send_gmail(user["email"], subject, body_html)

    # ── In-app notification ──────────────────────────────
    try:
        db.execute(
            "INSERT INTO notifications (user_id, title, body, type, item_id) VALUES (?,?,?,?,?)",
            (sap_id,
             "📢 Your Card Was Found!",
             f"Your student ID (SAP: {sap_id}) was found on campus. "
             f"{'Email sent to ' + user['email'] + '.' if email_sent else 'Please claim it at the security office.'}",
             "match", item_id)
        )
        # ── Log to ocr_resolutions table ─────────────────
        db.execute(
            "INSERT INTO ocr_resolutions (item_id, sap_id, ocr_text, name_found, email_sent) VALUES (?,?,?,?,?)",
            (item_id, sap_id, json.dumps(ocr_lines), name or user["name"], int(email_sent))
        )
        db.commit()
    except Exception as e:
        print(f"[OCR] DB write failed: {e}")

    return jsonify({
        "ocr_text"  : ocr_lines,
        "sap_id"    : sap_id,
        "name"      : name or user["name"],
        "user_found": True,
        "email_sent": email_sent,
        "message"   : (
            f"Owner identified: {user['name']} ({user['email']}). "
            f"{'Email sent ✅' if email_sent else 'In-app notification created (Gmail not configured).'}"
        )
    })


# ════════════════════════════════════════════════════════
# MANUAL NOTIFY — send email to any address about a found item
# POST /api/notify_owner
# ════════════════════════════════════════════════════════

@app.route("/api/notify_owner", methods=["POST"])
@auth_required
def notify_owner():
    """
    Manually send a found-item email to any email address.

    Use this when:
    - The finder knows the owner's email directly
    - Auto-OCR couldn't identify the owner
    - You want to notify someone not registered in the system

    Body (JSON):
        to_email     : str  — recipient email address (required)
        owner_name   : str  — owner's name to personalise the email (optional)
        item_id      : int  — found item ID from the DB (optional)
        item_name    : str  — item description, e.g. "Blue ID Card" (optional)
        location     : str  — where it was found (optional)
        finder_name  : str  — your name, so owner knows who to contact (optional)
        message      : str  — optional custom message to include
    """
    data = request.get_json()
    if not data or not data.get("to_email"):
        return jsonify({"error": "to_email is required"}), 400

    to_email    = data["to_email"].strip()
    owner_name  = data.get("owner_name", "Student").strip()
    item_id     = data.get("item_id")
    item_name   = data.get("item_name", "your item")
    location    = data.get("location", "campus")
    finder_name = data.get("finder_name", "A fellow student")
    custom_msg  = data.get("message", "")

    # Basic email format check
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', to_email):
        return jsonify({"error": "Invalid email address format"}), 400

    # Resolve item details from DB if item_id is given
    if item_id:
        try:
            db   = get_db()
            item = db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
            if item:
                item = dict(item)
                item_name = item_name or item["name"]
                location  = location  or item["location"]
        except Exception:
            pass

    # Also check if this email belongs to a registered user → send in-app notification too
    inapp_sent = False
    try:
        db   = get_db()
        user = db.execute("SELECT * FROM users WHERE email=?", (to_email,)).fetchone()
        if user:
            db.execute(
                "INSERT INTO notifications (user_id,title,body,type,item_id) VALUES (?,?,?,?,?)",
                (user["id"],
                 f"📢 Someone Found: {item_name}",
                 f"{finder_name} found '{item_name}' at {location} and thinks it may be yours. "
                 f"Please check the portal or contact campus security.",
                 "match", item_id)
            )
            db.commit()
            inapp_sent = True
    except Exception as e:
        print(f"[notify_owner] In-app notification failed: {e}")

    # Build email
    subject   = f"📢 UPES Lost & Found — Someone Found: {item_name}"
    body_html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
      <h2 style="color:#6c3a9f;">UPES Lost &amp; Found Portal</h2>
      <p>Dear <strong>{owner_name}</strong>,</p>
      <p>
        <strong>{finder_name}</strong> found an item on campus that may belong to you.
      </p>
      <table style="border-collapse:collapse;width:100%;font-size:14px;margin:12px 0;">
        <tr style="background:#f5f0ff;">
          <td style="padding:8px;font-weight:bold;">Item</td>
          <td style="padding:8px;">{item_name}</td>
        </tr>
        <tr>
          <td style="padding:8px;font-weight:bold;">Found At</td>
          <td style="padding:8px;">{location}</td>
        </tr>
        <tr style="background:#f5f0ff;">
          <td style="padding:8px;font-weight:bold;">Reported On</td>
          <td style="padding:8px;">{datetime.now().strftime('%d %b %Y, %I:%M %p')}</td>
        </tr>
        {"<tr><td style='padding:8px;font-weight:bold;'>Item ID</td><td style='padding:8px;'>#" + str(item_id) + "</td></tr>" if item_id else ""}
      </table>
      {f'<p style="background:#fff8e1;border-left:4px solid #f0c040;padding:10px;">{custom_msg}</p>' if custom_msg else ''}
      <p>
        Please visit the
        <a href="http://localhost:5000" style="color:#6c3a9f;">UPES Lost &amp; Found Portal</a>
        or contact the campus security office to claim your item.
      </p>
      <p style="color:#888;font-size:12px;">
        This message was sent via the UPES Lost &amp; Found AI System on behalf of {finder_name}.
      </p>
    </div>
    """

    email_sent = send_gmail(to_email, subject, body_html)

    if not email_sent and not inapp_sent:
        return jsonify({
            "email_sent"  : False,
            "inapp_sent"  : False,
            "message"     : (
                "Gmail not configured — set GMAIL_USER and GMAIL_APP_PASSWORD "
                "environment variables to enable email sending. "
                + ("In-app notification also skipped (email not registered)." if not inapp_sent else "")
            )
        }), 503

    return jsonify({
        "email_sent" : email_sent,
        "inapp_sent" : inapp_sent,
        "to"         : to_email,
        "message"    : (
            f"{'Email sent ✅' if email_sent else 'Email skipped (Gmail not configured).'} "
            f"{'In-app notification also sent ✅' if inapp_sent else ''}"
        )
    })


# ════════════════════════════════════════════════════════
# MODULE 3 — SOCIAL LISTENING NER
# POST /api/ingest_external
# ════════════════════════════════════════════════════════

CAMPUS_LOCATIONS = [
    "library", "canteen", "cafeteria", "block", "gate",
    "hostel", "lab", "auditorium", "ground", "parking",
    "admin", "academic block", "sports complex", "gym",
    "reception", "corridor", "lecture hall", "seminar"
]

ITEM_KEYWORDS = [
    "wallet", "phone", "bag", "backpack", "laptop", "watch",
    "keys", "key", "bottle", "umbrella", "glasses", "id card",
    "earphones", "airpods", "charger", "notebook", "book",
    "purse", "ring", "necklace", "cap", "jacket", "tiffin"
]


def ner_extract_item_and_location(text: str) -> dict:
    """
    Extract item and campus location from free text using SpaCy NER.
    Falls back to keyword matching when SpaCy is unavailable.
    """
    result = {"item": None, "location": None, "confidence": "low"}

    if not SPACY_READY:
        for kw in ITEM_KEYWORDS:
            if kw in text.lower():
                result["item"] = kw.title()
                break
        for loc in CAMPUS_LOCATIONS:
            if loc in text.lower():
                idx   = text.lower().index(loc)
                result["location"] = text[max(0, idx-5):min(len(text), idx+30)].strip()
                break
        return result

    doc  = _nlp(text)

    # Location from NER
    for ent in doc.ents:
        if ent.label_ in ("LOC", "GPE", "FAC"):
            result["location"] = ent.text
            break

    # Location keyword fallback
    if not result["location"]:
        lower = text.lower()
        for loc in CAMPUS_LOCATIONS:
            if loc in lower:
                idx = lower.index(loc)
                result["location"] = text[max(0, idx-5):min(len(text), idx+35)].strip(" ,.")
                break

    # Item from noun chunks
    stop_words = {"someone", "anybody", "hey", "guys", "please", "help",
                  "found", "lost", "left", "seen", "looking"}
    candidates = []
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if any(s in chunk_text for s in stop_words):
            continue
        if any(kw in chunk_text for kw in ITEM_KEYWORDS):
            candidates.append(chunk.text)
        elif len(chunk.text.split()) <= 4 and chunk.root.pos_ == "NOUN":
            candidates.append(chunk.text)

    if candidates:
        for c in candidates:
            if any(kw in c.lower() for kw in ITEM_KEYWORDS):
                result["item"] = c.strip()
                break
        if not result["item"]:
            result["item"] = candidates[0].strip()

    # Item keyword fallback
    if not result["item"]:
        for kw in ITEM_KEYWORDS:
            if kw in text.lower():
                result["item"] = kw.title()
                break

    if result["item"] and result["location"]:
        result["confidence"] = "high"
    elif result["item"] or result["location"]:
        result["confidence"] = "medium"

    return result


@app.route("/api/ingest_external", methods=["POST"])
@auth_required
def ingest_external():
    """
    Ingest raw text from WhatsApp / Slack / Email.

    Body (JSON):
        text   : str  — raw message text
        source : str  — optional label e.g. "WhatsApp", "Slack"
    """
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "text field required"}), 400

    raw_text = data["text"].strip()
    source   = data.get("source", "External")

    extracted = ner_extract_item_and_location(raw_text)
    item_name = extracted.get("item") or "Unknown Item"
    location  = extracted.get("location") or "Campus (location unclear)"

    db   = get_db()
    desc = (f"[Auto-ingested from {source}] {raw_text}\n\n"
            f"NER confidence: {extracted['confidence']}")

    try:
        cur     = db.execute(
            """INSERT INTO items
               (type, status, name, category, location, description, reported_by)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("found", "open", item_name, "Other", location, desc, g.user_id)
        )
        db.commit()
        item_id = cur.lastrowid

        # Log raw message
        db.execute(
            "INSERT INTO social_ingestions (source, raw_text, item_id) VALUES (?,?,?)",
            (source, raw_text, item_id)
        )
        db.commit()

        # Run standard matching
        matches = run_matching(item_id, "found", db)
        notify_matches(item_id, "found", matches, db)

        return jsonify({
            "extracted": extracted,
            "item_id"  : item_id,
            "matches"  : len(matches),
            "message"  : (
                f"Ingested from {source}. "
                f"Item: '{item_name}' at '{location}'. "
                f"{len(matches)} potential match(es) found."
            )
        }), 201

    except Exception as e:
        return jsonify({"error": f"DB insert failed: {e}"}), 500


# ════════════════════════════════════════════════════════
# MODULE 4 — PREDICTIVE HOTSPOT ANALYTICS
# GET /api/analytics/hotspots
# ════════════════════════════════════════════════════════

def normalise_location(raw: str) -> str:
    """Collapse location spelling variants to a canonical campus label."""
    loc   = raw.strip().lower()
    rules = [
        (r'lib',               "Library"),
        (r'canteen|cafeteria', "Canteen"),
        (r'block[- _]?(\w+)',  "Block-{g1}"),
        (r'hostel',            "Hostel"),
        (r'lab',               "Lab"),
        (r'auditorium|audi',   "Auditorium"),
        (r'parking',           "Parking"),
        (r'admin',             "Admin Block"),
        (r'gate',              "Main Gate"),
        (r'sports|ground',     "Sports Complex"),
        (r'gym',               "Gymnasium"),
    ]
    for pattern, label in rules:
        m = re.search(pattern, loc)
        if m:
            if "{g1}" in label:
                g1 = m.group(1).upper() if m.lastindex else ""
                return label.replace("{g1}", g1)
            return label
    return raw.strip().title()[:40]


@app.route("/api/analytics/hotspots", methods=["GET"])
@auth_required
def hotspot_analytics():
    """
    Predictive hotspot analytics using historical items data.

    Query params:
        days  : int  — lookback window (default 30)
        limit : int  — top N locations to return (default 3)
    """
    days  = int(request.args.get("days",  30))
    limit = int(request.args.get("limit", 3))

    db   = get_db()
    rows = db.execute(
        "SELECT name, category, location, type, created_at FROM items"
    ).fetchall()

    if not rows:
        return jsonify({"top_found_locations": [], "all_locations": [], "summary": "No data yet."})

    now    = datetime.utcnow()
    cutoff = now - timedelta(days=days)

    loc_data: dict = defaultdict(lambda: {
        "total_reports" : 0, "recent_reports": 0,
        "found_count"   : 0, "lost_count"    : 0, "items": []
    })

    for row in rows:
        canonical = normalise_location(row["location"] or "Unknown")
        entry     = loc_data[canonical]
        entry["total_reports"] += 1
        entry["items"].append(row["name"])
        try:
            ts = datetime.strptime(row["created_at"][:19], "%Y-%m-%d %H:%M:%S")
            if ts >= cutoff:
                entry["recent_reports"] += 1
        except Exception:
            pass
        if row["type"] == "found":
            entry["found_count"] += 1
        else:
            entry["lost_count"] += 1

    max_total  = max(v["total_reports"]  for v in loc_data.values()) or 1
    max_recent = max(v["recent_reports"] for v in loc_data.values()) or 1

    scored = []
    for loc, d in loc_data.items():
        prob = round(
            0.6 * (d["total_reports"]  / max_total) +
            0.4 * (d["recent_reports"] / max_recent), 3
        )
        freq = defaultdict(int)
        for itm in d["items"]:
            freq[" ".join(itm.split()[:2])] += 1
        likely_items = [k for k, _ in
                        sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]]
        scored.append({
            "location"        : loc,
            "loss_probability": prob,
            "total_reports"   : d["total_reports"],
            "recent_reports"  : d["recent_reports"],
            "found_count"     : d["found_count"],
            "lost_count"      : d["lost_count"],
            "likely_items"    : likely_items
        })

    scored.sort(key=lambda x: x["loss_probability"], reverse=True)
    top       = scored[:limit]
    top_names = ", ".join(x["location"] for x in top)

    return jsonify({
        "top_found_locations": top,
        "all_locations"      : scored,
        "lookback_days"      : days,
        "summary"            : (
            f"Based on {len(rows)} total reports over the last {days} days, "
            f"the highest-risk zones are: {top_names}. "
            f"Items reported there are most likely to be found at these spots."
        )
    })


# ════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    init_extra_tables()
    print(f"\n{'='*50}")
    print("  UPES Lost & Found — OFFLINE MODE (CLIP AI)")
    print(f"{'='*50}")
    print(f"  CLIP      : {'Ready ✅' if CLIP_READY else 'pip install torch torchvision transformers'}")
    print(f"  GrabCut   : {'Ready ✅' if CV2_READY else 'pip install opencv-python-headless'}")
    print(f"  OCR       : {'Ready ✅' if OCR_READY else 'pip install easyocr'}")
    print(f"  NER       : {'Ready ✅' if SPACY_READY else 'pip install spacy && python -m spacy download en_core_web_sm'}")
    print(f"  Gmail     : {'Configured ✅' if GMAIL_USER else 'Not set (in-app notifications only)'}")
    port = int(os.environ.get("PORT", 5001))
    print(f"  Server    : http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(debug=True, port=port, host='0.0.0.0')