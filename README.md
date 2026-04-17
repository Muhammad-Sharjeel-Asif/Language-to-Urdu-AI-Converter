---
title: Multilingual to Urdu Translator
emoji: 🌐
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🌐 Multilingual to Urdu Translator

This application translates text from multiple languages into Urdu using **Helsinki-NLP MarianMT models**.

## 🚀 Features

- Supports 10+ languages:
  - English, Hindi, Arabic, French, Spanish, German, Turkish, Persian, Chinese, Russian
- Two-step translation pipeline:
  - Source → English → Urdu
- Direct translation for supported languages (e.g., Hindi → Urdu)
- Clean and modern UI using Gradio
- RTL Urdu output rendering
- Character counter
- Example inputs for quick testing

## 🧠 Models Used

- Helsinki-NLP MarianMT (via Hugging Face Transformers)

## 🛠️ How It Works

1. Select source language
2. Enter text
3. Click "Translate"
4. Get Urdu output instantly

## 💻 Run Locally

```bash
pip install -r requirements.txt
python app.py