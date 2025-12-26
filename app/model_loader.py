#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Loader Module - Singleton/Caching
Model ve vectorizer'ı tek seferde yükler, sonraki çağrılarda cache'den döner.
"""

import os
import math
import joblib
import numpy as np
from typing import Dict, Tuple, Any

# Debug flag - ilk çalıştırmada True yapın, sonra False
DEBUG_PROBA = os.environ.get("DEBUG_PROBA", "false").lower() == "true"

# Temperature Scaling - olasılıkları yumuşatır
# T=1.0: orijinal, T>1: daha yumuşak (daha az emin), T<1: daha keskin
TEMPERATURE = 6.0  # Dengeli sonuçlar için (%50-70 arası)

# Proje kök dizini
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Singleton cache
_cache: Dict[str, Any] = {}


def temperature_scale(probabilities: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """
    Temperature scaling ile olasılıkları yumuşatır.
    
    T > 1: Olasılıklar daha dengeli (less confident)
    T = 1: Orijinal olasılıklar
    T < 1: Olasılıklar daha keskin (more confident)
    
    Formül: softmax(log(p) / T)
    """
    if temperature == 1.0:
        return probabilities
    
    # Log-space'e çevir (0'a çok yakın değerler için küçük epsilon ekle)
    eps = 1e-10
    log_probs = np.log(probabilities + eps)
    
    # Temperature ile böl
    scaled_logits = log_probs / temperature
    
    # Softmax uygula (numerik stabilite için max çıkar)
    scaled_logits = scaled_logits - np.max(scaled_logits)
    exp_probs = np.exp(scaled_logits)
    scaled_probs = exp_probs / np.sum(exp_probs)
    
    return scaled_probs


def get_vectorizer():
    """TF-IDF vectorizer'ı yükler (singleton)."""
    if "vectorizer" not in _cache:
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer bulunamadı: {vectorizer_path}")
        _cache["vectorizer"] = joblib.load(vectorizer_path)
        print(f"✓ Vectorizer yüklendi: {vectorizer_path}")
    return _cache["vectorizer"]


def get_model(model_name: str):
    """Belirtilen modeli yükler (singleton)."""
    cache_key = f"model_{model_name}"
    if cache_key not in _cache:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        _cache[cache_key] = joblib.load(model_path)
        print(f"✓ Model yüklendi: {model_name}")
    return _cache[cache_key]


def get_all_models() -> Dict[str, Any]:
    """Tüm modelleri yükler ve döner."""
    model_names = ["LogisticRegression", "MultinomialNB", "SGDClassifier"]
    models = {}
    for name in model_names:
        try:
            models[name] = get_model(name)
        except FileNotFoundError:
            print(f"⚠ Model bulunamadı: {name}")
    return models


def load_all() -> Tuple[Any, Dict[str, Any]]:
    """Vectorizer ve tüm modelleri yükler."""
    vectorizer = get_vectorizer()
    models = get_all_models()
    return vectorizer, models


def predict_with_model(text: str, model_name: str) -> Dict[str, Any]:
    """
    Tek bir model ile tahmin yapar.
    
    Returns:
        {
            "model_name": str,
            "prediction": "ai" | "human",
            "ai_probability": float (0-100),
            "human_probability": float (0-100)
        }
    """
    vectorizer = get_vectorizer()
    model = get_model(model_name)
    
    # Metni vektörize et
    X = vectorizer.transform([text])
    
    # Olasılıklar - model.classes_ ile doğru mapping
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        
        # Temperature scaling uygula - olasılıkları yumuşat
        proba_scaled = temperature_scale(np.array(proba), TEMPERATURE)
        
        # numpy string'leri normal string'e çevir
        classes = [str(c) for c in model.classes_]
        
        # class->probability mapping oluştur (scaled probabilities ile)
        class_prob_map = {c: float(p) for c, p in zip(classes, proba_scaled)}
        
        # "ai" ve "human" olasılıklarını mapping'den al
        ai_prob = float(class_prob_map.get("ai", 0.0)) * 100.0
        human_prob = float(class_prob_map.get("human", 0.0)) * 100.0
        
        # Prediction'ı scaled olasılıklara göre belirle
        prediction = "ai" if ai_prob > human_prob else "human"
        
        # Debug logging
        if DEBUG_PROBA:
            print(f"[DEBUG] {model_name}:")
            print(f"  classes_: {classes}")
            print(f"  proba (raw): {[round(p, 4) for p in proba]}")
            print(f"  proba (scaled T={TEMPERATURE}): {[round(p, 4) for p in proba_scaled]}")
            print(f"  ai_prob: {ai_prob:.2f}%, human_prob: {human_prob:.2f}%")
            print(f"  prediction: {prediction}")
    else:
        # decision_function kullanan modeller için (örn: SVM)
        if hasattr(model, "decision_function"):
            decision = model.decision_function(X)[0]
            # decision_function pozitifse ikinci class (genellikle), negatifse birinci
            classes = [str(c) for c in model.classes_]
            sigmoid_prob = 1 / (1 + math.exp(-decision))
            
            # classes_[1] için sigmoid, classes_[0] için 1-sigmoid
            raw_proba = np.array([1 - sigmoid_prob, sigmoid_prob])
            proba_scaled = temperature_scale(raw_proba, TEMPERATURE)
            
            class_prob_map = {
                classes[0]: proba_scaled[0],
                classes[1]: proba_scaled[1]
            }
            ai_prob = float(class_prob_map.get("ai", 0.0)) * 100
            human_prob = float(class_prob_map.get("human", 0.0)) * 100
            prediction = "ai" if ai_prob > human_prob else "human"
            
            if DEBUG_PROBA:
                print(f"[DEBUG] {model_name} (decision_function):")
                print(f"  classes_: {classes}")
                print(f"  decision: {decision:.4f}")
                print(f"  proba (scaled T={TEMPERATURE}): {[round(p, 4) for p in proba_scaled]}")
                print(f"  ai_prob: {ai_prob:.2f}%, human_prob: {human_prob:.2f}%")
        else:
            # Sadece predict varsa, 100/0 olarak döndür
            prediction = str(model.predict(X)[0])
            ai_prob = 100.0 if prediction == "ai" else 0.0
            human_prob = 100.0 - ai_prob
    
    return {
        "model_name": str(model_name),
        "prediction": str(prediction),
        "ai_probability": float(round(ai_prob, 2)),
        "human_probability": float(round(human_prob, 2))
    }


def predict_all_models(text: str) -> Dict[str, Any]:
    """
    Tüm modellerle tahmin yapar.
    """
    model_names = ["LogisticRegression", "MultinomialNB", "SGDClassifier"]
    results = []
    
    for name in model_names:
        try:
            result = predict_with_model(text, name)
            results.append(result)
        except Exception as e:
            print(f"⚠ {name} tahmin hatası: {e}")
    
    if not results:
        raise RuntimeError("Hiçbir model tahmin yapamadı!")
    
    # Ortalamaları hesapla - kesinlikle float
    avg_ai = float(sum(r["ai_probability"] for r in results) / len(results))
    avg_human = float(sum(r["human_probability"] for r in results) / len(results))
    
    # Prediction'ı kesin karşılaştırma ile belirle
    overall_prediction = "ai" if avg_ai > avg_human else "human"
    
    return {
        "overall": {
            "prediction": str(overall_prediction),
            "ai_probability": float(round(avg_ai, 2)),
            "human_probability": float(round(avg_human, 2))
        },
        "models": results,
        "best_model": "SGDClassifier"
    }


def is_loaded() -> bool:
    """Model ve vectorizer yüklü mü kontrol eder."""
    return "vectorizer" in _cache and len([k for k in _cache if k.startswith("model_")]) > 0


def clear_cache():
    """Cache'i temizler."""
    _cache.clear()
