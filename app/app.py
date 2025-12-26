#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web Application - Human vs AI Text Detector
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Model loader'ı import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import predict_all_models, load_all, is_loaded

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Uygulama başlatıldığında modelleri yükle
print("=" * 50)
print("Human vs AI Text Detector")
print("=" * 50)

try:
    load_all()
    print("✓ Modeller başarıyla yüklendi!")
except Exception as e:
    print(f"⚠ Model yükleme hatası: {e}")
    print("Not: Önce 'python scripts/train_models.py' çalıştırın.")


@app.route("/")
def index():
    """Ana sayfa - form göster."""
    return render_template("index.html", result=None, error=None, text="")


@app.route("/predict", methods=["POST"])
def predict():
    """Form submit - tahmin yap ve sonucu göster."""
    text = request.form.get("text", "").strip()
    
    # Debug logging
    print(f"\n[/predict] Text length: {len(text)}")
    if text:
        print(f"[/predict] First 120 chars: {text[:120]}...")
    
    # Validasyon
    if not text:
        print("[/predict] ERROR: Empty text")
        return render_template("index.html", result=None, error="Lütfen bir metin girin!", text="")
    
    if len(text) < 50:
        print(f"[/predict] ERROR: Text too short ({len(text)} chars)")
        return render_template("index.html", result=None, error="Metin en az 50 karakter olmalıdır.", text=text)
    
    try:
        # Tahmin yap
        result = predict_all_models(text)
        print(f"[/predict] Overall: {result['overall']['prediction']} (AI: {result['overall']['ai_probability']}%, Human: {result['overall']['human_probability']}%)")
        return render_template("index.html", result=result, error=None, text=text)
    except Exception as e:
        print(f"[/predict] EXCEPTION: {e}")
        return render_template("index.html", result=None, error=f"Tahmin hatası: {str(e)}", text=text)


import os

# Debug flag - debug bilgisi eklemek için DEBUG_API=true yapın
DEBUG_API = os.environ.get("DEBUG_API", "false").lower() == "true"


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint."""
    data = request.get_json()
    
    if not data or "text" not in data:
        print("[/api/predict] ERROR: Missing 'text' field")
        return jsonify({"error": "JSON body'de 'text' alanı gerekli"}), 400
    
    text = data["text"].strip()
    
    # Debug logging
    print(f"\n[/api/predict] Text length: {len(text)}")
    if text:
        print(f"[/api/predict] First 120 chars: {text[:120]}...")
    
    if not text:
        print("[/api/predict] ERROR: Empty text")
        return jsonify({"error": "Metin boş olamaz"}), 400
    
    if len(text) < 50:
        print(f"[/api/predict] ERROR: Text too short ({len(text)} chars)")
        return jsonify({"error": "Metin en az 50 karakter olmalıdır"}), 400
    
    try:
        result = predict_all_models(text)
        print(f"[/api/predict] Overall: {result['overall']['prediction']} (AI: {result['overall']['ai_probability']}%, Human: {result['overall']['human_probability']}%)")
        
        # Debug modu: ek bilgi ekle
        if DEBUG_API:
            from model_loader import get_model
            result["debug"] = {
                "input_length": len(text),
                "first_120_chars": text[:120],
                "model_classes": {}
            }
            for model_name in ["LogisticRegression", "MultinomialNB", "SGDClassifier"]:
                try:
                    model = get_model(model_name)
                    result["debug"]["model_classes"][model_name] = [str(c) for c in model.classes_]
                except:
                    pass
        
        return jsonify(result)
    except Exception as e:
        print(f"[/api/predict] EXCEPTION: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Sağlık kontrolü."""
    return jsonify({
        "status": "ok",
        "models_loaded": is_loaded()
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
