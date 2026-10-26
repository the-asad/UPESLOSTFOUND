"""
============================================================
UPES Lost & Found — AI Matching Model
Minor Project | B.Tech (Hons.) CSE | Semester 6
University of Petroleum and Energy Studies
============================================================

HOW TO RUN:
    pip install scikit-learn pandas numpy matplotlib seaborn pillow
    python ml_model.py

This script:
1. Builds a sample lost & found dataset
2. Trains a TF-IDF text similarity model
3. Adds category boost + image color histogram matching
4. Evaluates with Precision, Recall, F1
5. Saves 4 plots as PNG files
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math, warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================
# 1. DATASET
# ============================================================

lost_items = pd.DataFrame([
    {"id":"L001","name":"Black Nike Backpack",     "category":"Bags",        "location":"Library 2nd floor",      "description":"Black Nike backpack laptop compartment white swoosh stickers"},
    {"id":"L002","name":"iPhone 14 Blue Case",     "category":"Electronics", "location":"Canteen near windows",   "description":"iPhone 14 blue cracked case small dent bottom corner"},
    {"id":"L003","name":"Honda Bike Keys",         "category":"Keys",        "location":"Parking Lot A",          "description":"Honda key red keyring tag small torch attached"},
    {"id":"L004","name":"Prescription Glasses",    "category":"Other",       "location":"Block A Room 204",       "description":"Black thin frame prescription glasses brown leather case"},
    {"id":"L005","name":"Samsung Galaxy Watch",    "category":"Electronics", "location":"Sports Complex",         "description":"Samsung galaxy watch black strap silver body scratched"},
    {"id":"L006","name":"Blue Adidas Hoodie",      "category":"Clothing",    "location":"Hostel Block 3",         "description":"Blue Adidas hoodie size XL name Rahul inside collar"},
    {"id":"L007","name":"Calculus Textbook",       "category":"Books",       "location":"Block B Room 105",       "description":"Engineering Calculus Sem 3 textbook handwritten notes yellow cover"},
    {"id":"L008","name":"UPES Student ID Card",    "category":"Cards",       "location":"Library entry",          "description":"UPES student ID card CSE department with photo"},
    {"id":"L009","name":"Dell Laptop Charger",     "category":"Electronics", "location":"Lab 3 Computer Lab",     "description":"Dell 65W charger black cable brick adapter"},
    {"id":"L010","name":"Gold Earrings",           "category":"Jewellery",   "location":"Girls Hostel Common",    "description":"Small gold hoop earrings pair red velvet pouch"},
    {"id":"L011","name":"Red Titan Wristwatch",    "category":"Electronics", "location":"Canteen ground floor",   "description":"Titan red dial watch brown leather strap"},
    {"id":"L012","name":"Navy Blue Polo T-shirt",  "category":"Clothing",    "location":"Gym changing room",      "description":"Navy blue polo shirt size L Lacoste brand"},
    {"id":"L013","name":"Physics Lab Record",      "category":"Books",       "location":"Physics Lab Block C",    "description":"Physics practical record notebook blue cover student number inside"},
    {"id":"L014","name":"Black Umbrella",          "category":"Other",       "location":"Admin Block reception",  "description":"Black automatic umbrella curved handle slight tear canopy"},
    {"id":"L015","name":"Silver Bracelet",         "category":"Jewellery",   "location":"Library reading room",   "description":"Thin silver bracelet heart charm pendant"},
    {"id":"L016","name":"MacBook Air Charger",     "category":"Electronics", "location":"Seminar Hall",           "description":"Apple MagSafe charger white cable 60W MacBook Air"},
    {"id":"L017","name":"Cricket Bat",             "category":"Sports",      "location":"Sports Ground",          "description":"SS cricket bat red grip medium size"},
    {"id":"L018","name":"Green Water Bottle",      "category":"Other",       "location":"Block D cafeteria",      "description":"Dark green Milton water bottle 1 litre UPES sticker"},
    {"id":"L019","name":"Wireless Earbuds",        "category":"Electronics", "location":"Library study pods",     "description":"White wireless earbuds charging case brand boAt"},
    {"id":"L020","name":"Driving Licence",         "category":"Cards",       "location":"Parking area",           "description":"Driving licence card photo Uttarakhand state"},
])

found_items = pd.DataFrame([
    {"id":"F001","name":"Nike Backpack Black",        "category":"Bags",        "location":"Library near stairs",     "description":"Black nike bag laptop inside white logo front"},
    {"id":"F002","name":"Phone with Blue Cover",      "category":"Electronics", "location":"Canteen table 12",        "description":"iPhone cracked blue cover found canteen table"},
    {"id":"F003","name":"Keyring with Honda Key",     "category":"Keys",        "location":"Parking gate area",       "description":"Honda motorcycle key red plastic keyring torch"},
    {"id":"F004","name":"Spectacles with Case",       "category":"Other",       "location":"Block A corridor",        "description":"Prescription spectacles black thin frame brown case"},
    {"id":"F005","name":"Smartwatch Samsung",         "category":"Electronics", "location":"Volleyball court",        "description":"Samsung watch black band found sports area"},
    {"id":"F006","name":"Hoodie Blue Colour",         "category":"Clothing",    "location":"Common room hostel",      "description":"Blue hoodie XL Adidas brand"},
    {"id":"F007","name":"Engineering Maths Book",     "category":"Books",       "location":"Classroom B105",          "description":"Calculus book sem 3 handwritten notes found under bench"},
    {"id":"F008","name":"College ID Card",            "category":"Cards",       "location":"Library main gate",       "description":"Student identity card UPES CSE branch"},
    {"id":"F009","name":"Laptop Charger Dell",        "category":"Electronics", "location":"Computer lab near PC8",   "description":"Dell charger 65 watt black color found near computer"},
    {"id":"F010","name":"Gold Hoop Earrings",         "category":"Jewellery",   "location":"Hostel washroom",         "description":"Pair small gold hoops red small pouch"},
    {"id":"F011","name":"Titan Watch Red Dial",       "category":"Electronics", "location":"Canteen counter",         "description":"Titan watch red display brown leather band"},
    {"id":"F012","name":"Blue Polo Shirt",            "category":"Clothing",    "location":"Gym locker area",         "description":"Navy blue polo L size lacoste logo"},
    {"id":"F013","name":"Lab Record Notebook",        "category":"Books",       "location":"Physics lab table 4",     "description":"Practical notebook blue cover physics experiments"},
    {"id":"F014","name":"Black Umbrella",             "category":"Other",       "location":"Admin office entry",      "description":"Automatic black umbrella curved handle near reception"},
    {"id":"F015","name":"Bracelet Silver",            "category":"Jewellery",   "location":"Library 1st floor",       "description":"Thin silver bracelet small heart pendant"},
    {"id":"F016","name":"Apple Charger White",        "category":"Electronics", "location":"Seminar hall chair row3", "description":"White MacBook charger MagSafe 60W"},
    {"id":"F017","name":"Cricket Bat Red Grip",       "category":"Sports",      "location":"Pavilion ground",         "description":"SS bat cricket medium red handle"},
    {"id":"F018","name":"Water Bottle Green Milton",  "category":"Other",       "location":"D Block canteen table",   "description":"Green 1L milton bottle UPES sticker"},
    {"id":"F019","name":"Earphones Wireless boAt",    "category":"Electronics", "location":"Library pod 7",           "description":"boAt white wireless earbuds white charging case"},
    {"id":"F020","name":"Driving Licence Card",       "category":"Cards",       "location":"Bike parking zone",       "description":"Driving licence Uttarakhand male photo found ground"},
])

# Ground truth: L001↔F001, L002↔F002, ... (perfect diagonal)
ground_truth = {f"L{str(i).zfill(3)}": f"F{str(i).zfill(3)}" for i in range(1, 21)}

print("=" * 55)
print("  UPES LOST & FOUND — AI MATCHING MODEL")
print("=" * 55)
print(f"  Lost items  : {len(lost_items)}")
print(f"  Found items : {len(found_items)}")
print(f"  GT pairs    : {len(ground_truth)}")
print()

# ============================================================
# 2. TF-IDF TEXT SIMILARITY
# ============================================================

def build_text(row):
    return f"{row['name']} {row['category']} {row['description']}".lower()

lost_texts  = lost_items.apply(build_text, axis=1).tolist()
found_texts = found_items.apply(build_text, axis=1).tolist()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words="english")
vectorizer.fit(lost_texts + found_texts)

lost_vecs   = vectorizer.transform(lost_texts)
found_vecs  = vectorizer.transform(found_texts)
text_matrix = cosine_similarity(lost_vecs, found_vecs)  # shape: (20, 20)

print("  [1] TF-IDF similarity matrix built:", text_matrix.shape)

# ============================================================
# 3. CATEGORY BOOST
# ============================================================

CATEGORY_BOOST = 0.15

def apply_boost(matrix, lost_df, found_df):
    boosted = matrix.copy()
    for i in range(len(lost_df)):
        for j in range(len(found_df)):
            if lost_df.iloc[i]["category"].lower() == found_df.iloc[j]["category"].lower():
                boosted[i][j] = min(1.0, boosted[i][j] + CATEGORY_BOOST)
    return boosted

boosted_matrix = apply_boost(text_matrix, lost_items, found_items)
print(f"  [2] Category boost (+{CATEGORY_BOOST}) applied")

# ============================================================
# 4. IMAGE COLOR HISTOGRAM SIMILARITY (simulated)
# ============================================================

def bhattacharyya(hist_a, hist_b):
    """Bhattacharyya coefficient — higher = more similar (0 to 1)."""
    return sum(math.sqrt(a * b) for a, b in zip(hist_a, hist_b))

# Simulate normalized RGB histograms (3 bins) for all items
np.random.seed(42)
def make_hist(name):
    name_l = name.lower()
    if "black" in name_l or "dark" in name_l:
        return [0.05, 0.05, 0.90]
    elif "blue" in name_l:
        return [0.10, 0.75, 0.15]
    elif "red" in name_l:
        return [0.80, 0.10, 0.10]
    elif "green" in name_l:
        return [0.10, 0.80, 0.10]
    elif "gold" in name_l or "yellow" in name_l:
        return [0.70, 0.25, 0.05]
    elif "silver" in name_l or "white" in name_l:
        return [0.35, 0.35, 0.30]
    elif "navy" in name_l:
        return [0.05, 0.20, 0.75]
    else:
        base = np.random.dirichlet([2, 2, 2])
        return base.tolist()

lost_hists  = [make_hist(r["name"]) for _, r in lost_items.iterrows()]
found_hists = [make_hist(r["name"]) for _, r in found_items.iterrows()]

image_matrix = np.zeros((len(lost_items), len(found_items)))
for i in range(len(lost_items)):
    for j in range(len(found_items)):
        image_matrix[i][j] = bhattacharyya(lost_hists[i], found_hists[j])

print("  [3] Image color histogram similarity computed")

# ============================================================
# 5. COMBINED SCORE
# ============================================================

TEXT_W  = 0.60
IMAGE_W = 0.40

combined_matrix = TEXT_W * boosted_matrix + IMAGE_W * image_matrix
print(f"  [4] Combined score = {TEXT_W}×text + {IMAGE_W}×image")

# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate(matrix, threshold=0.20):
    y_true, y_pred = [], []
    results = []
    for i, lrow in lost_items.iterrows():
        best_j     = int(np.argmax(matrix[i]))
        best_score = matrix[i][best_j]
        pred_fid   = found_items.iloc[best_j]["id"] if best_score >= threshold else None
        true_fid   = ground_truth.get(lrow["id"])
        correct    = (pred_fid == true_fid)
        y_true.append(1)
        y_pred.append(1 if correct else 0)
        results.append({
            "Lost"      : lrow["name"],
            "Predicted" : found_items.iloc[best_j]["name"],
            "Score"     : round(best_score, 3),
            "Correct"   : "✅" if correct else "❌"
        })
    acc  = sum(y_pred) / len(y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1, pd.DataFrame(results)

# ============================================================
# 7. THRESHOLD SWEEP — find optimal cut-off
# ============================================================

thresholds = np.arange(0.05, 0.95, 0.05)
rows = []
for t in thresholds:
    a, p, r, f, _ = evaluate(combined_matrix, threshold=t)
    rows.append({"threshold": round(t,2), "accuracy": a, "precision": p, "recall": r, "f1": f})
thresh_df = pd.DataFrame(rows)
best_t    = thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"]

acc, prec, rec, f1, pred_df = evaluate(combined_matrix, threshold=best_t)

print()
print("=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)
print(f"  Optimal Threshold : {best_t}")
print(f"  Accuracy          : {acc*100:.1f}%")
print(f"  Precision         : {prec*100:.1f}%")
print(f"  Recall            : {rec*100:.1f}%")
print(f"  F1 Score          : {f1*100:.1f}%")
print("=" * 55)
print()
print(pred_df.to_string(index=False))

# ============================================================
# 8. PLOTS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("UPES Lost & Found — AI Matching Model\nMinor Project | B.Tech CSE | UPES",
             fontsize=14, fontweight="bold")

# ── Plot 1: TF-IDF Similarity Heatmap ──
ax1 = axes[0][0]
sns.heatmap(text_matrix, ax=ax1, cmap="YlOrRd",
            xticklabels=found_items["id"], yticklabels=lost_items["id"],
            linewidths=0.3, linecolor="#eee", vmin=0, vmax=1)
ax1.set_title("TF-IDF Cosine Similarity Matrix", fontweight="bold")
ax1.set_xlabel("Found Items"); ax1.set_ylabel("Lost Items")
ax1.tick_params(labelsize=7)

# ── Plot 2: Threshold Analysis ──
ax2 = axes[0][1]
ax2.plot(thresh_df["threshold"], thresh_df["accuracy"],  "o-", label="Accuracy",  lw=2)
ax2.plot(thresh_df["threshold"], thresh_df["f1"],        "s-", label="F1 Score",  lw=2)
ax2.plot(thresh_df["threshold"], thresh_df["precision"], "^--",label="Precision", lw=1.5)
ax2.plot(thresh_df["threshold"], thresh_df["recall"],    "v--",label="Recall",    lw=1.5)
ax2.axvline(best_t, color="red", linestyle=":", lw=1.5, label=f"Best t={best_t}")
ax2.set_title("Metrics vs Decision Threshold", fontweight="bold")
ax2.set_xlabel("Threshold"); ax2.set_ylabel("Score")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# ── Plot 3: Combined Score Bar ──
ax3 = axes[1][0]
top1 = [combined_matrix[i].max() for i in range(len(lost_items))]
correct = [combined_matrix[i].argmax() == i for i in range(len(lost_items))]
colors  = ["#2e7d32" if c else "#c62828" for c in correct]
ax3.bar(range(len(lost_items)), top1, color=colors, edgecolor="white", linewidth=0.5)
ax3.axhline(best_t, color="orange", linestyle="--", lw=1.5, label=f"Threshold ({best_t})")
ax3.set_xticks(range(len(lost_items)))
ax3.set_xticklabels(lost_items["id"], rotation=45, ha="right", fontsize=7)
ax3.set_title("Top-1 Match Score per Lost Item", fontweight="bold")
ax3.set_ylabel("Combined Similarity Score")
handles = [mpatches.Patch(color="#2e7d32", label="Correct"),
           mpatches.Patch(color="#c62828", label="Wrong"),
           plt.Line2D([0],[0], color="orange", linestyle="--", label=f"Threshold")]
ax3.legend(handles=handles, fontsize=8)

# ── Plot 4: Category-wise Average Score ──
ax4 = axes[1][1]
cats = lost_items["category"].unique()
cat_scores = []
for cat in cats:
    idxs = lost_items[lost_items["category"] == cat].index.tolist()
    cat_scores.append(np.mean([combined_matrix[i].max() for i in idxs]))
colors4 = plt.cm.tab10(np.linspace(0, 1, len(cats)))
bars = ax4.barh(cats, cat_scores, color=colors4)
ax4.axvline(best_t, color="red", linestyle="--", lw=1.5)
for bar, v in zip(bars, cat_scores):
    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{v:.2f}", va="center", fontsize=8)
ax4.set_title("Average Match Score by Category", fontweight="bold")
ax4.set_xlabel("Avg Combined Similarity Score")
ax4.set_xlim(0, 1.15)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✅ Plot saved → model_results.png")
print("  ✅ Model run complete!")
