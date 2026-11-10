# -*- coding: utf-8 -*-
# ============================================================
# AES SEDERHANA: XLM-RoBERTa + Cosine Similarity
# (versi ringkas dari notebook referensi, output tetap lengkap)
# ------------------------------------------------------------
# CARA PAKAI DI COLAB:
#   1. Upload file "Dataset Biologi.xlsx"
#   2. Runtime > Change runtime type > T4 GPU
#   3. Jalankan cell AES 1 s/d AES 8 berurutan
#   4. Setiap blok "# ====" = satu cell di Colab
# ============================================================

# ============================================================
# AES 1 - LOAD DATA + SPLIT 80:20
# ============================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("Dataset Biologi.xlsx", sheet_name=0)
df.columns = df.columns.str.strip()
df = df.rename(columns={"Jawaban Siswa": "Jawaban_Siswa",
                        "Kunci Guru": "Kunci_Guru",
                        "Nilai": "Skor"})
df["Skor"] = pd.to_numeric(df["Skor"], errors="coerce")
df["Bobot"] = pd.to_numeric(df["Bobot"], errors="coerce")
df = df.dropna(subset=["Jawaban_Siswa", "Kunci_Guru", "Skor", "Bobot"]).reset_index(drop=True)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

total = len(df)
print("=" * 60)
print("AES 1 - DATA SIAP (split 80:20, seed 42)")
print("=" * 60)
print(f"Total jawaban esai : {total}")
print(f"Data latih (train) : {len(train_data)} baris ({len(train_data)/total*100:.1f}%)")
print(f"Data uji (test)    : {len(test_data)} baris ({len(test_data)/total*100:.1f}%)")
print(f"Jumlah siswa       : {df['Nama'].nunique()} | Kelas: {df['Kelas'].iloc[0]}")
print(f"Jumlah soal unik   : {df['Kode_Soal'].nunique()}")
print(f"Sebaran bobot soal : {df['Bobot'].value_counts().sort_index().to_dict()}")
print("\nSebaran skor guru (seluruh data):")
print(df["Skor"].value_counts().sort_index().to_string())

# ============================================================
# AES 2 - PRA-PEMROSESAN (huruf kecil + normalisasi kata)
# ============================================================
import re

def load_kamus(url, sep=","):
    try:
        d = pd.read_csv(url, sep=sep, header=None, encoding="utf-8", on_bad_lines="skip")
        return dict(zip(d[0].astype(str).str.strip().str.lower(),
                        d[1].astype(str).str.strip().str.lower()))
    except Exception:
        return {}

base = "https://raw.githubusercontent.com/evrintobing17/NormalisasiKata/master/"
KAMUS = {**load_kamus(base + "kamus_alay.csv"), **load_kamus(base + "kbba.txt", sep="\t")}
KAMUS.update({"yg": "yang", "dgn": "dengan", "tdk": "tidak", "ga": "tidak",
              "krn": "karena", "dpt": "dapat"})
KAMUS.update({"transpot": "transpor", "dipusi": "difusi", "osmosia": "osmosis",
              "membrane": "membran", "flasid": "flaksid", "gradient": "gradien",
              "semipermiabel": "semipermeabel", "hipertonis": "hipertonik",
              "hipotonis": "hipotonik"})

def bersihkan(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # buang tanda baca
    return " ".join(KAMUS.get(w, w) for w in text.split())

for d in [train_data, test_data]:
    for kol in ["Pertanyaan", "Kunci_Guru", "Jawaban_Siswa"]:
        d[kol + "_clean"] = d[kol].apply(bersihkan)

print("=" * 60)
print("AES 2 - PRA-PEMROSESAN SELESAI")
print("=" * 60)
print(f"Kamus normalisasi aktif : {len(KAMUS)} kata")
print("\nContoh hasil pembersihan (jawaban siswa pertama di data uji):")
print("SEBELUM :", test_data["Jawaban_Siswa"].iloc[0][:110])
print("SESUDAH :", test_data["Jawaban_Siswa_clean"].iloc[0][:110])

# ============================================================
# AES 3 - TOKENISASI (cek panjang token XLM-RoBERTa)
# ============================================================
from transformers import AutoTokenizer

MODEL = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

panjang_siswa = train_data["Jawaban_Siswa_clean"].apply(lambda t: len(tokenizer.tokenize(str(t))))
panjang_guru = train_data["Kunci_Guru_clean"].apply(lambda t: len(tokenizer.tokenize(str(t))))

print("=" * 60)
print("AES 3 - TOKENISASI")
print("=" * 60)
print(f"Tokenizer model     : {MODEL}")
print(f"Token jawaban siswa : rata-rata {panjang_siswa.mean():.0f} | maksimum {panjang_siswa.max()}")
print(f"Token kunci guru    : rata-rata {panjang_guru.mean():.0f} | maksimum {panjang_guru.max()}")
maks_token = max(panjang_siswa.max(), panjang_guru.max())
print(f"Batas MAX_LEN = 512 -> {'SEMUA teks muat, tidak ada yang terpotong' if maks_token <= 512 else 'ADA yang terpotong (truncation aktif)'}")

# ============================================================
# AES 4 - VEKTOR XLM-RoBERTa (token CLS, 768 dimensi)
# Butuh GPU: Runtime > Change runtime type > T4 GPU
# ============================================================
import torch
from transformers import XLMRobertaModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLMRobertaModel.from_pretrained(MODEL).to(device)
model.eval()

def vektor(text):
    masukan = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        keluar = model(**masukan)
    return keluar.last_hidden_state[:, 0, :][0].cpu().numpy()   # ambil token CLS

def vektor_semua(df, kolom, label):
    hasil = []
    for teks in tqdm(df[kolom], desc=label):
        hasil.append(vektor(str(teks)) if str(teks).strip() else np.zeros(768))
    return np.array(hasil)

train_v_siswa = vektor_semua(train_data, "Jawaban_Siswa_clean", "Vektor siswa (train)")
train_v_guru = vektor_semua(train_data, "Kunci_Guru_clean", "Vektor guru (train)")
test_v_siswa = vektor_semua(test_data, "Jawaban_Siswa_clean", "Vektor siswa (test)")
test_v_guru = vektor_semua(test_data, "Kunci_Guru_clean", "Vektor guru (test)")

print("=" * 60)
print("AES 4 - EKSTRAKSI VEKTOR SELESAI")
print("=" * 60)
print(f"Perangkat komputasi  : {device}")
print(f"Dimensi vektor train : {train_v_siswa.shape}  (jumlah baris x dimensi)")
print(f"Dimensi vektor test  : {test_v_siswa.shape}")

# ============================================================
# AES 5 - COSINE SIMILARITY (jawaban siswa vs kunci guru)
# ============================================================
from sklearn.metrics.pairwise import cosine_similarity

def cosine(v1, v2):
    return float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0])

train_data["cosine"] = [cosine(a, b) for a, b in zip(train_v_siswa, train_v_guru)]
test_data["cosine"] = [cosine(a, b) for a, b in zip(test_v_siswa, test_v_guru)]

neg = (test_data["cosine"] < 0).sum()
print("=" * 60)
print("AES 5 - COSINE SIMILARITY (data uji)")
print("=" * 60)
print(f"minimum {test_data['cosine'].min():.4f} | rata-rata {test_data['cosine'].mean():.4f} | maksimum {test_data['cosine'].max():.4f}")
print(f"Nilai negatif: {neg} dari {len(test_data)} (di AES 6 nanti di-clip jadi 0)")
print()
print(test_data[["Kode_Soal", "Jawaban_Siswa_clean", "Kunci_Guru_clean", "cosine"]].head(3).to_string(max_colwidth=40))

# ============================================================
# AES 6 - CARA 1: SKOR = cosine x bobot soal (TANPA regresi)
# Mengacu Lahitani (2022): skor esai = nilai kemiripan x bobot
# ============================================================
for d in [train_data, test_data]:
    d["cosine01"] = d["cosine"].clip(0, 1)                # cosine negatif -> 0
    d["skor_kontinu"] = d["cosine01"] * d["Bobot"]        # skor mentah
    d["skor_prediksi"] = np.maximum(0.0, np.round(d["skor_kontinu"] * 2) / 2)  # bulatkan ke 0.5

sama1 = (test_data["skor_prediksi"] == test_data["Skor"]).sum()
print("=" * 60)
print("AES 6 - SKOR AKHIR CARA 1 (cosine x bobot, dibulatkan 0.5)")
print("=" * 60)
print(test_data[["cosine", "Bobot", "skor_kontinu", "skor_prediksi", "Skor"]].head(10).to_string())
print(f"\nPrediksi PERSIS sama dengan skor guru: {sama1}/{len(test_data)} = {sama1/len(test_data)*100:.1f}%")

# ============================================================
# AES 7 - CARA 2: cosine DISESUAIKAN dengan regresi linear (OLS)
# Mengacu Dhini & Girsang (2022): target skor dinormalisasi 0-1
# ============================================================
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(train_data[["cosine"]], train_data["Skor"] / train_data["Bobot"])

print("=" * 60)
print("AES 7 - SKOR AKHIR CARA 2 (cosine + regresi linear)")
print("=" * 60)
print(f"Persamaan regresi : prediksi_norm = {reg.intercept_:.4f} + {reg.coef_[0]:.4f} x cosine")
print(f"R2 pada data latih: {reg.score(train_data[['cosine']], train_data['Skor'] / train_data['Bobot']):.4f}")

for d in [train_data, test_data]:
    norm = np.clip(reg.predict(d[["cosine"]]), 0, 1)      # batasi 0-1
    d["skor_kontinu_reg"] = norm * d["Bobot"].values
    d["skor_prediksi_reg"] = np.maximum(0.0, np.round(d["skor_kontinu_reg"] * 2) / 2)

sama2 = (test_data["skor_prediksi_reg"] == test_data["Skor"]).sum()
print(test_data[["cosine", "Bobot", "skor_kontinu_reg", "skor_prediksi_reg", "Skor"]].head(10).to_string())
print(f"\nPrediksi PERSIS sama dengan skor guru: {sama2}/{len(test_data)} = {sama2/len(test_data)*100:.1f}%")

# ============================================================
# AES 8 - EVALUASI: MSE, RMSE, MAE, PCC, QWK
# MSE/RMSE/MAE kecil = makin baik | PCC & QWK besar = makin baik
# ============================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr

def evaluasi(df, kolom_kontinu, kolom_bulat, metode, nama_split):
    y = df["Skor"].values
    yp = df[kolom_kontinu].values
    return {
        "Metode": metode, "Data": nama_split, "n": len(df),
        "MSE": mean_squared_error(y, yp),
        "RMSE": np.sqrt(mean_squared_error(y, yp)),
        "MAE": mean_absolute_error(y, yp),
        "PCC": pearsonr(y, yp)[0],
        "QWK": cohen_kappa_score(np.round(y * 2).astype(int),
                                 np.round(df[kolom_bulat].values * 2).astype(int),
                                 weights="quadratic"),
    }

rekap = pd.DataFrame([
    evaluasi(train_data, "skor_kontinu", "skor_prediksi", "Cara 1 (cosine x bobot)", "latih"),
    evaluasi(test_data, "skor_kontinu", "skor_prediksi", "Cara 1 (cosine x bobot)", "uji"),
    evaluasi(train_data, "skor_kontinu_reg", "skor_prediksi_reg", "Cara 2 (cosine + regresi)", "latih"),
    evaluasi(test_data, "skor_kontinu_reg", "skor_prediksi_reg", "Cara 2 (cosine + regresi)", "uji"),
])

print("=" * 60)
print("AES 8 - REKAP EVALUASI (data latih vs data uji)")
print("=" * 60)
print(rekap.round(4).to_string(index=False))

test_data["error"] = (test_data["Skor"] - test_data["skor_prediksi_reg"]).abs()
print("\n5 prediksi dengan kesalahan terbesar (cara 2, data uji):")
print(test_data.nlargest(5, "error")[["Kode_Soal", "Bobot", "Skor", "skor_prediksi_reg", "error"]].to_string(index=False))

test_data.to_excel("hasil_AES_data_uji.xlsx", index=False)
rekap.to_excel("hasil_AES_rekap_evaluasi.xlsx", index=False)
print("\nTersimpan: hasil_AES_data_uji.xlsx & hasil_AES_rekap_evaluasi.xlsx")
