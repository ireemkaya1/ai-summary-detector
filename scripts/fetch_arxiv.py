#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arXiv API ile gerçek makale abstract'ları çeker.
Çıktı: data/raw/human_data.csv (kolonlar: text,label)
Meta bilgiler ayrı dosyada: data/raw/human_meta.csv
"""

import os
import sys
import csv
import time
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "human_data.csv")
META_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "human_meta.csv")

# arXiv API endpoint
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Kategoriler (AI/ML odaklı)
CATEGORIES = [
    "cs.AI",    # Artificial Intelligence
    "cs.LG",    # Machine Learning
    "cs.CL",    # Computation and Language
    "stat.ML",  # Machine Learning (Statistics)
]

# Hedef sayı
TARGET_COUNT = 3000
BATCH_SIZE = 200  # arXiv API max 200
RATE_LIMIT_SECONDS = 3  # arXiv rate limit


def fetch_arxiv_batch(category: str, start: int, max_results: int) -> list:
    """arXiv API'den bir batch abstract çeker."""
    
    query = f"cat:{category}"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"  [HATA] API çağrısı başarısız: {e}")
        return []
    
    # XML parse
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        print(f"  [HATA] XML parse hatası: {e}")
        return []
    
    # Namespace
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    results = []
    for entry in root.findall("atom:entry", ns):
        try:
            arxiv_id_elem = entry.find("atom:id", ns)
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)
            
            if arxiv_id_elem is None or summary_elem is None:
                continue
            
            arxiv_id = arxiv_id_elem.text.strip().split("/abs/")[-1]
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
            abstract = summary_elem.text.strip().replace("\n", " ")
            
            # Boş veya çok kısa abstract'ları atla
            if len(abstract) < 100:
                continue
            
            results.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "category": category
            })
        except Exception:
            continue
    
    return results


def text_hash(text: str) -> str:
    """Metin için hash oluşturur (deduplication için)."""
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def main():
    print("=" * 60)
    print("arXiv API - Human Abstract Fetcher")
    print(f"Hedef: {TARGET_COUNT} abstract")
    print(f"Kategoriler: {', '.join(CATEGORIES)}")
    print("=" * 60)
    
    # Output dizinini oluştur
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    all_abstracts = []
    seen_hashes = set()
    
    # Her kategoriden eşit sayıda çek
    per_category = TARGET_COUNT // len(CATEGORIES) + 200  # Buffer ekle
    
    for category in CATEGORIES:
        print(f"\n[{category}] Çekiliyor...")
        category_count = 0
        start = 0
        
        while category_count < per_category and start < 5000:
            print(f"  Batch: start={start}, max={BATCH_SIZE}")
            
            batch = fetch_arxiv_batch(category, start, BATCH_SIZE)
            
            if not batch:
                print(f"  Batch boş, sonraki kategoriye geçiliyor.")
                break
            
            for item in batch:
                h = text_hash(item["abstract"])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_abstracts.append(item)
                    category_count += 1
            
            print(f"  Toplam (bu kategori): {category_count}")
            
            start += BATCH_SIZE
            time.sleep(RATE_LIMIT_SECONDS)
            
            if len(all_abstracts) >= TARGET_COUNT + 500:
                break
        
        print(f"[{category}] Tamamlandı: {category_count} abstract")
        
        if len(all_abstracts) >= TARGET_COUNT + 500:
            break
    
    # TARGET_COUNT'a düşür
    if len(all_abstracts) > TARGET_COUNT:
        all_abstracts = all_abstracts[:TARGET_COUNT]
    
    print(f"\n{'=' * 60}")
    print(f"Toplam: {len(all_abstracts)} unique abstract")
    
    # Ana CSV (text,label) yaz
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text", "label"])
        for item in all_abstracts:
            writer.writerow([item["abstract"], "human"])
    
    print(f"Kaydedildi: {OUTPUT_FILE}")
    
    # Meta CSV yaz
    with open(META_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["arxiv_id", "title", "category"])
        for item in all_abstracts:
            writer.writerow([item["arxiv_id"], item["title"], item["category"]])
    
    print(f"Meta kaydedildi: {META_FILE}")
    
    # İstatistik
    print(f"\n{'=' * 60}")
    print("Kategori Dağılımı:")
    cat_counts = {}
    for item in all_abstracts:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    
    print(f"\n{'=' * 60}")
    print("TAMAMLANDI!")


if __name__ == "__main__":
    main()
