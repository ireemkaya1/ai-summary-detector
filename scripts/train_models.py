#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model eğitimi - 3 model (LogisticRegression, MultinomialNB, SGDClassifier)
TF-IDF (word + char ngram) kullanır.
Çıktı: models/*.joblib, results/metrics.txt, results/metrics.json, results/confusion_matrix_*.png
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import FeatureUnion
import joblib

warnings.filterwarnings("ignore")

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
VAL_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "val.csv")
TEST_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def load_data():
    """Veriyi yükler."""
    print("Veri yükleniyor...")
    
    if not os.path.exists(TRAIN_FILE):
        print(f"HATA: Train dosyası bulunamadı: {TRAIN_FILE}")
        print("Önce 'python scripts/clean_and_split.py' çalıştırın.")
        sys.exit(1)
    
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    print(f"  Train: {len(train_df)} satır")
    print(f"  Val: {len(val_df)} satır")
    print(f"  Test: {len(test_df)} satır")
    
    return train_df, val_df, test_df


def create_tfidf_vectorizer():
    """Word + Char n-gram TF-IDF vectorizer oluşturur."""
    
    # Word n-grams (1-2)
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=8000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    # Char n-grams (3-5)
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=4000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    # Birleştir
    combined_vectorizer = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])
    
    return combined_vectorizer


def train_models(X_train, y_train, X_val, y_val):
    """3 modeli eğitir ve döndürür."""
    
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        "MultinomialNB": MultinomialNB(
            alpha=0.1
        ),
        "SGDClassifier": SGDClassifier(
            loss="log_loss",  # predict_proba için gerekli
            max_iter=1000,
            random_state=42
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{name} eğitiliyor...")
        
        model.fit(X_train, y_train)
        
        # Validation accuracy
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        trained_models[name] = model
    
    return trained_models


def evaluate_models(models, X_test, y_test, label_names):
    """Modelleri test seti üzerinde değerlendirir."""
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name} değerlendiriliyor...")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="ai")
        rec = recall_score(y_test, y_pred, pos_label="ai")
        f1 = f1_score(y_test, y_pred, pos_label="ai")
        
        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "predictions": y_pred,
            "probabilities": y_proba
        }
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Label leakage uyarısı
        if acc >= 0.99:
            print(f"\n  ⚠️ UYARI: Çok yüksek accuracy ({acc:.4f}) - Label leakage olabilir!")
    
    return results


def save_confusion_matrices(models, X_test, y_test, label_names):
    """Confusion matrix görsellerini kaydeder."""
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=label_names)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix - {name}")
        
        filename = os.path.join(RESULTS_DIR, f"confusion_matrix_{name}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Kaydedildi: {filename}")


def save_results(results, vectorizer, models):
    """Sonuçları dosyalara kaydeder."""
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Vectorizer'ı kaydet
    vectorizer_file = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_file)
    print(f"Kaydedildi: {vectorizer_file}")
    
    # Modelleri kaydet
    for name, model in models.items():
        model_file = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, model_file)
        print(f"Kaydedildi: {model_file}")
    
    # Metrics JSON
    metrics_json = {}
    for name, res in results.items():
        metrics_json[name] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1_score": res["f1_score"]
        }
    
    json_file = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Kaydedildi: {json_file}")
    
    # Metrics TXT
    txt_file = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL DEĞERLENDİRME SONUÇLARI\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for name, res in results.items():
            f.write(f"{name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"  Precision: {res['precision']:.4f}\n")
            f.write(f"  Recall:    {res['recall']:.4f}\n")
            f.write(f"  F1 Score:  {res['f1_score']:.4f}\n")
            f.write("\n")
        
        # En iyi model
        best_model = max(results.items(), key=lambda x: x[1]["f1_score"])
        f.write(f"\nEn İyi Model (F1 Score): {best_model[0]} ({best_model[1]['f1_score']:.4f})\n")
    
    print(f"Kaydedildi: {txt_file}")


def main():
    print("=" * 60)
    print("MODEL EĞİTİMİ")
    print("=" * 60)
    
    # Veri yükle
    train_df, val_df, test_df = load_data()
    
    # Sadece text ve label kolonlarını kullan (label leakage önlemi)
    X_train_raw = train_df["text"].fillna("").astype(str)
    y_train = train_df["label"]
    
    X_val_raw = val_df["text"].fillna("").astype(str)
    y_val = val_df["label"]
    
    X_test_raw = test_df["text"].fillna("").astype(str)
    y_test = test_df["label"]
    
    label_names = ["ai", "human"]
    
    # TF-IDF vectorizer oluştur ve fit et
    print("\nTF-IDF Vectorizer oluşturuluyor...")
    vectorizer = create_tfidf_vectorizer()
    
    print("Vectorizer fit ediliyor...")
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    print(f"Feature sayısı: {X_train.shape[1]}")
    
    # Modelleri eğit
    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ")
    print("=" * 60)
    
    models = train_models(X_train, y_train, X_val, y_val)
    
    # Modelleri değerlendir
    print("\n" + "=" * 60)
    print("MODEL DEĞERLENDİRME (Test Set)")
    print("=" * 60)
    
    results = evaluate_models(models, X_test, y_test, label_names)
    
    # Sonuçları kaydet
    print("\n" + "=" * 60)
    print("SONUÇLARI KAYDET")
    print("=" * 60)
    
    save_results(results, vectorizer, models)
    save_confusion_matrices(models, X_test, y_test, label_names)
    
    # Özet
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    
    print("\nModel Performansları (Test Set):")
    for name, res in results.items():
        print(f"  {name}: Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]["f1_score"])
    print(f"\n🏆 En İyi Model: {best_model[0]} (F1={best_model[1]['f1_score']:.4f})")
    
    print("\n" + "=" * 60)
    print("TAMAMLANDI!")


if __name__ == "__main__":
    main()
