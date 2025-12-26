#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human ve AI verilerini birleştirir.
Girdi: data/raw/human_data.csv + data/raw/ai2_data.csv
Çıktı: data/raw/dataset_raw.csv (kolonlar: text,label)
"""

import os
import sys
import pandas as pd

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HUMAN_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "human_data.csv")
AI_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "ai2_data.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "dataset_raw.csv")


def main():
    print("=" * 60)
    print("Dataset Merge Tool")
    print("=" * 60)
    
    # Dosya kontrolleri
    if not os.path.exists(HUMAN_FILE):
        print(f"HATA: Human veri dosyası bulunamadı: {HUMAN_FILE}")
        print("Önce 'python scripts/fetch_arxiv.py' çalıştırın.")
        sys.exit(1)
    
    if not os.path.exists(AI_FILE):
        print(f"HATA: AI veri dosyası bulunamadı: {AI_FILE}")
        print("Önce 'python scripts/generate_ai_gemini.py' çalıştırın.")
        sys.exit(1)
    
    # Dosyaları oku
    print(f"\nOkunuyor: {HUMAN_FILE}")
    human_df = pd.read_csv(HUMAN_FILE)
    print(f"  Satır sayısı: {len(human_df)}")
    print(f"  Kolonlar: {list(human_df.columns)}")
    
    print(f"\nOkunuyor: {AI_FILE}")
    ai_df = pd.read_csv(AI_FILE)
    print(f"  Satır sayısı: {len(ai_df)}")
    print(f"  Kolonlar: {list(ai_df.columns)}")
    
    # Kolon kontrolü
    required_cols = ["text", "label"]
    for col in required_cols:
        if col not in human_df.columns:
            print(f"HATA: '{col}' kolonu human_data.csv'de bulunamadı!")
            sys.exit(1)
        if col not in ai_df.columns:
            print(f"HATA: '{col}' kolonu ai2_data.csv'de bulunamadı!")
            sys.exit(1)
    
    # Sadece text,label kolonlarını al
    human_df = human_df[["text", "label"]]
    ai_df = ai_df[["text", "label"]]
    
    # Label kontrol
    print(f"\nLabel dağılımı (human_data.csv):")
    print(human_df["label"].value_counts().to_string())
    
    print(f"\nLabel dağılımı (ai2_data.csv):")
    print(ai_df["label"].value_counts().to_string())
    
    # Birleştir
    merged_df = pd.concat([human_df, ai_df], ignore_index=True)
    
    # Karıştır
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n{'=' * 60}")
    print(f"Birleşik Dataset:")
    print(f"  Toplam satır: {len(merged_df)}")
    print(f"\nLabel dağılımı:")
    label_counts = merged_df["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(merged_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Validasyon
    print(f"\n{'=' * 60}")
    print("VALİDASYON:")
    
    errors = []
    
    # Toplam satır kontrolü
    if len(merged_df) < 6000:
        errors.append(f"Yetersiz veri: {len(merged_df)} < 6000")
    else:
        print(f"  ✓ Toplam satır yeterli: {len(merged_df)} >= 6000")
    
    # Label dengesi kontrolü
    human_count = label_counts.get("human", 0)
    ai_count = label_counts.get("ai", 0)
    
    if human_count < 2500:
        errors.append(f"Human veri yetersiz: {human_count} < 2500")
    else:
        print(f"  ✓ Human veri yeterli: {human_count}")
    
    if ai_count < 2500:
        errors.append(f"AI veri yetersiz: {ai_count} < 2500")
    else:
        print(f"  ✓ AI veri yeterli: {ai_count}")
    
    # Denge kontrolü
    if human_count > 0 and ai_count > 0:
        ratio = min(human_count, ai_count) / max(human_count, ai_count)
        if ratio < 0.8:
            errors.append(f"Label dengesi bozuk: {ratio:.2f} < 0.8")
        else:
            print(f"  ✓ Label dengesi iyi: {ratio:.2f}")
    
    if errors:
        print(f"\nHATALAR:")
        for err in errors:
            print(f"  ✗ {err}")
        sys.exit(1)
    
    # Kaydet
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'=' * 60}")
    print(f"Kaydedildi: {OUTPUT_FILE}")
    
    # Örnek göster
    print(f"\nİlk 3 örnek:")
    for i, row in merged_df.head(3).iterrows():
        print(f"\n--- Örnek {i+1} (label={row['label']}) ---")
        print(f"{row['text'][:200]}...")
    
    print(f"\n{'=' * 60}")
    print("TAMAMLANDI!")


if __name__ == "__main__":
    main()
