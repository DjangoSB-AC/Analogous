# -*- coding: utf-8 -*-
"""
Spyder Editor
Doer: Alex Koutalistras
Cosmetic Label text recognition
"""
import os
import re
import csv
from collections import Counter
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

# -------------------------------------------------------------------
# 1. CONFIG
# -------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MECCA_FOLDER = r"C:\Users\alexk\OneDrive\Desktop\Coding\Cosmetics\Cosmetics Images\Mecca_Source\Mecca\convertedJPG"
SEPHORA_FOLDER = r"C:\Users\alexk\OneDrive\Desktop\Coding\Cosmetics\Cosmetics Images\Sephora_Source\Sephora"

UNIGRAM_CSV = "skincare_label_terms.csv"
BIGRAM_CSV = "skincare_label_bigrams.csv"
CLAIMS_CSV = "skincare_claim_hits.csv"

# -------------------------------------------------------------------
# 2. IMAGE PREPROCESSING
# -------------------------------------------------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    # grayscale
    img = img.convert("L")
    # upscale
    w, h = img.size
    img = img.resize((int(w * 1.8), int(h * 1.8)), Image.LANCZOS)
    # contrast boost
    img = ImageEnhance.Contrast(img).enhance(1.6)
    # slight sharpen
    img = img.filter(ImageFilter.SHARPEN)
    # binarise
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    return img

# -------------------------------------------------------------------
# 3. OCR ALL IMAGES → CORPUS
# -------------------------------------------------------------------
all_text_chunks = []

for folder in [MECCA_FOLDER, SEPHORA_FOLDER]:
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, file)
            try:
                img = Image.open(img_path)
                img = preprocess_image(img)
                text = pytesseract.image_to_string(img, config="--psm 6")
                all_text_chunks.append(text)
            except Exception as e:
                print(f"Error on {img_path}: {e}")

# single corpus string
raw_corpus = "\n".join(all_text_chunks)

# -------------------------------------------------------------------
# 4. TEXT CLEANING / TOKENISING
# -------------------------------------------------------------------
# base stopwords
base_stopwords = {
    "the","and","for","with","a","an","of","to","on","in","by",
    "from","at","is","it","this","that","as","our","your","be",
    "use","apply","daily","skin","face","cleanse","avoid","area",
    "keep","out","reach","children","if","irritation","occurs"
}

# instructiony words (back-of-pack stuff)
instruction_stopwords = {
    "after","before","then","massage","onto","product","products",
    "water","rinse","clean","morning","night","evening","only",
    "external","eyes","contact","immediately","store","cool","dry"
}

all_stopwords = base_stopwords.union(instruction_stopwords)

# remove non-letters, lowercase
clean_text = re.sub(r"[^a-zA-Z\s]", " ", raw_corpus)
clean_text = clean_text.lower()

# tokens
tokens = [
    w for w in clean_text.split()
    if w not in all_stopwords and len(w) > 2
]

# -------------------------------------------------------------------
# 5. UNIGRAM COUNTS
# -------------------------------------------------------------------
unigram_freq = Counter(tokens)

# -------------------------------------------------------------------
# 6. BIGRAM COUNTS
# -------------------------------------------------------------------
bigrams = Counter(zip(tokens, tokens[1:]))
bigram_freq = {f"{w1} {w2}": count for (w1, w2), count in bigrams.items()}

# -------------------------------------------------------------------
# 7. NORMALISE CLAIM TERMS
#    (so “fragrance-free” → “fragrance free”, like our cleaned text)
# -------------------------------------------------------------------
def normalize_phrase(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

raw_claim_terms = {
    # benefits
    "hydrating","hydration","moisturising","moisturizing","brightening","brighten",
    "glow","radiance","radiant","plumping","firming","soothing","calming",
    "antiaging","anti-ageing","anti-age","antiwrinkle","anti-wrinkle",
    "renew","renewal","repair","restoring","restorative","protect","protection",
    # sun/protection
    "spf","sunscreen","uva","uvb","broad-spectrum","broad spectrum",
    # ingredients
    "vitamin","vitamin c","vitamin-c","niacinamide",
    "hyaluronic","hyaluronic acid","retinol","ceramide","peptide",
    # claims
    "fragrance-free","fragrance free","vegan","clean","dermatologist-tested","dermatologist tested",
    "non-comedogenic","non comedogenic","sensitive","gentle","oil-free","oil free",
    # actives
    "exfoliating","aha","bha","lactic","glycolic","salicylic",
    # barrier/microbiome
    "barrier","microbiome","even tone","pore refining","pore-refining"
}

claim_terms_norm = {normalize_phrase(t) for t in raw_claim_terms}

# -------------------------------------------------------------------
# 8. MATCH CLAIMS AGAINST UNIGRAMS + BIGRAMS
# -------------------------------------------------------------------
claim_hits = []

# 8a. from unigrams
for claim in claim_terms_norm:
    if " " not in claim:  # single word
        count = unigram_freq.get(claim, 0)
        if count > 0:
            claim_hits.append(("unigram", claim, count))

# 8b. from bigrams / phrases
for claim in claim_terms_norm:
    if " " in claim:  # phrase
        count = bigram_freq.get(claim, 0)
        if count > 0:
            claim_hits.append(("bigram", claim, count))

# sort claim hits
claim_hits_sorted = sorted(claim_hits, key=lambda x: x[2], reverse=True)

# -------------------------------------------------------------------
# 9. SAVE TO CSVs
# -------------------------------------------------------------------

# 9a. unigrams
with open(UNIGRAM_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["term", "count"])
    for term, count in unigram_freq.most_common():
        writer.writerow([term, count])

# 9b. bigrams
with open(BIGRAM_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["phrase", "count"])
    for phrase, count in sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([phrase, count])

# 9c. claim hits only
with open(CLAIMS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["type", "claim", "count"])
    for row in claim_hits_sorted:
        writer.writerow(list(row))

# -------------------------------------------------------------------
# 10. PRINT SUMMARY TO CONSOLE
# -------------------------------------------------------------------
print("\nTop 30 label terms:")
for term, count in unigram_freq.most_common(30):
    print(term, count)

print("\nTop 30 label phrases:")
for phrase, count in sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(phrase, count)

print("\nClaim-like terms found:")
for kind, claim, count in claim_hits_sorted:
    print(f"{kind:7} {claim:20} {count}")