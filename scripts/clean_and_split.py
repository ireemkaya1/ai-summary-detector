#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset temizleme ve train/val/test split.
Girdi: data/raw/dataset_raw.csv (kolonlar: text,label)
Çıktı: data/processed/dataset_clean.csv + train.csv/val.csv/test.csv
"""

import os
import sys
import re
import hashlib
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "dataset_raw.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Temizlik parametreleri
MIN_CHARS = 100
MIN_WORDS = 20


def normalize_text(text: str) -> str:
    """Unicode normalizasyonu ve temel temizlik."""
    if not isinstance(text, str):
        return ""
    
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    
    # HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    
    # LaTeX komutlarını temizle
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\$[^$]+\$", "", text)
    
    # Çoklu boşlukları tek boşluğa
    text = re.sub(r"\s+", " ", text)
    
    # Trim
    text = text.strip()
    
    return text


def is_valid_text(text: str) -> bool:
    """Metnin geçerli olup olmadığını kontrol eder."""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Minimum karakter
    if len(text) < MIN_CHARS:
        return False
    
    # Minimum kelime
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    
    return True


def text_hash(text: str) -> str:
    """Metin için hash oluşturur (deduplication için)."""
    normalized = text.lower().strip()
    # İlk 500 karakteri kullan
    normalized = normalized[:500]
    return hashlib.md5(normalized.encode()).hexdigest()


def main():
    print("=" * 60)
    print("Dataset Cleaner & Splitter")
    print("=" * 60)
    
    # Dosya kontrol
    if not os.path.exists(INPUT_FILE):
        print(f"HATA: Girdi dosyası bulunamadı: {INPUT_FILE}")
        print("Önce 'python scripts/merge_raw.py' çalıştırın.")
        sys.exit(1)
    
    # Output dizini oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Veriyi oku
    print(f"\nOkunuyor: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    original_count = len(df)
    print(f"Orijinal satır sayısı: {original_count}")
    print(f"Kolonlar: {list(df.columns)}")
    
    # Orijinal label dağılımı
    print(f"\nOrijinal label dağılımı:")
    orig_counts = df["label"].value_counts()
    for label, count in orig_counts.items():
        print(f"  {label}: {count}")
    
    # =====================
    # TEMİZLİK
    # =====================
    print(f"\n{'=' * 60}")
    print("TEMİZLİK AŞAMASI")
    print("=" * 60)
    
    # 1. Normalize et
    print("\n1. Unicode normalizasyonu...")
    df["text"] = df["text"].apply(normalize_text)
    
    # 2. Geçersiz metinleri filtrele
    print("2. Kısa/geçersiz metinler filtreleniyor...")
    before_filter = len(df)
    df = df[df["text"].apply(is_valid_text)]
    after_filter = len(df)
    print(f"   Silinen: {before_filter - after_filter}")
    
    # 3. Duplicate temizliği
    print("3. Duplicate temizliği...")
    df["_hash"] = df["text"].apply(text_hash)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["_hash"], keep="first")
    after_dedup = len(df)
    df = df.drop(columns=["_hash"])
    print(f"   Silinen duplicate: {before_dedup - after_dedup}")
    
    # =====================
    # DENGELEME
    # =====================
    print(f"\n{'=' * 60}")
    print("DENGELEME AŞAMASI")
    print("=" * 60)
    
    label_counts = df["label"].value_counts()
    print(f"Temizlik sonrası label dağılımı:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Undersampling ile dengeleme
    min_count = label_counts.min()
    print(f"\nUndersampling: Her sınıftan {min_count} örnek alınacak")
    
    balanced_dfs = []
    for label in df["label"].unique():
        label_df = df[df["label"] == label].sample(n=min_count, random_state=42)
        balanced_dfs.append(label_df)
    
    df = pd.concat(balanced_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Karıştır
    
    final_count = len(df)
    print(f"\nDengeleme sonrası: {final_count} satır")
    
    # Temiz dataseti kaydet
    clean_file = os.path.join(OUTPUT_DIR, "dataset_clean.csv")
    df.to_csv(clean_file, index=False)
    print(f"Kaydedildi: {clean_file}")
    
    # =====================
    # SPLIT
    # =====================
    print(f"\n{'=' * 60}")
    print("TRAIN/VAL/TEST SPLIT (80/10/10)")
    print("=" * 60)
    
    # Stratified split
    X = df["text"]
    y = df["label"]
    
    # İlk önce train ve temp'e ayır (80/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Temp'i val ve test'e ayır (50/50 -> 10/10)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # DataFrame'lere dönüştür
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    val_df = pd.DataFrame({"text": X_val, "label": y_val})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})
    
    # Kaydet
    train_file = os.path.join(OUTPUT_DIR, "train.csv")
    val_file = os.path.join(OUTPUT_DIR, "val.csv")
    test_file = os.path.join(OUTPUT_DIR, "test.csv")
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nKaydedilen dosyalar:")
    print(f"  {train_file}: {len(train_df)} satır")
    print(f"  {val_file}: {len(val_df)} satır")
    print(f"  {test_file}: {len(test_df)} satır")
    
    # =====================
    # RAPOR
    # =====================
    print(f"\n{'=' * 60}")
    print("ÖZET RAPOR")
    print("=" * 60)
    
    print(f"\nVeri Akışı:")
    print(f"  Orijinal satır sayısı: {original_count}")
    print(f"  Kısa/geçersiz silinen: {before_filter - after_filter}")
    print(f"  Duplicate silinen: {before_dedup - after_dedup}")
    print(f"  Dengeleme sonrası: {final_count}")
    
    print(f"\nSplit Oranları:")
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"  Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/total*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/total*100:.1f}%)")
    
    print(f"\nLabel Dağılımı (Train):")
    for label, count in train_df["label"].value_counts().items():
        pct = count / len(train_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\nLabel Dağılımı (Test):")
    for label, count in test_df["label"].value_counts().items():
        pct = count / len(test_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\n{'=' * 60}")
    print("TAMAMLANDI!")


if __name__ == "__main__":
    main()
