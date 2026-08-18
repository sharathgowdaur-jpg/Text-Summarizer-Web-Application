---
title: Text Summarizer Web Application
emoji: 📝
colorFrom: orange
colorTo: amber
sdk: docker
app_port: 7860
---

# 📝 Text Summarizer Web Application

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An AI-powered, full-stack Natural Language Processing (NLP) web application designed to automatically condense long dialogues, paragraphs, or articles into short, accurate summaries. Built with **FastAPI**, **Google T5 Transformers**, **PyTorch**, and a custom **Neumorphic (Soft UI)** frontend.

---

## 🌐 Live Demo

- **Vercel Live Web App:** `https://text-summarizer-web-application.vercel.app` *(Replace with your Vercel URL)*
- **Hugging Face Space Demo:** `https://huggingface.co/spaces/YOUR_USERNAME/Text-Summarizer-Web-Application`
- **Local Host URL:** `http://127.0.0.1:8000`

---

## ✨ Key Features

- 🤖 **Hybrid AI Summarization:** Uses Google's **T5 (Text-to-Text Transfer Transformer)** model fine-tuned for dialogue and document summarization.
- ⚡ **Instant NLP Condensation Engine:** Includes a fallback sentence-density ranking algorithm that condenses input text by 50%–75% instantly.
- 🎨 **Neumorphism (Soft UI) Interface:** Modern, 3D extruded design with smooth rounded corners (`border-radius: 16px`), inset text areas, and tactile interactive button states.
- 📱 **Fully Responsive Layout:** Vertical stacked UI optimized for both desktop monitors and mobile devices.
- 🧼 **Automated Preprocessing:** Pre-cleans input text by stripping HTML tags, redundant line breaks, and whitespace before model inference.
- ☁️ **Serverless Cloud Ready:** Configured with lightweight Vercel serverless handlers (`api/index.py`) and Docker containerization (`Dockerfile`).

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, ES6 JavaScript | Neumorphic Soft UI, responsive design, `fetch` API requests |
| **Backend API** | Python 3.10+, FastAPI, Uvicorn | Async RESTful API endpoints (`GET /`, `POST /summarize/`) |
| **AI / Machine Learning** | PyTorch, Hugging Face `transformers` | Fine-tuned T5 model (`saved_summary_model/`) |
| **Cloud Deployment** | Vercel, Docker, Hugging Face Spaces | Serverless Python functions & Docker containerization |

---

## 📂 Project Directory Structure

```text
Text-Summarizer-Web-Application/
├── api/
│   └── index.py            # Vercel Serverless Python entry point
├── saved_summary_model/    # Fine-tuned T5 Hugging Face model & tokenizer
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── app.py                  # Local FastAPI application runner
├── index.html              # Neumorphic user interface (HTML/CSS/JS)
├── requirements.txt        # Production dependencies
├── vercel.json             # Vercel deployment routing config
├── Dockerfile              # Docker container configuration
├── .gitignore              # Excluded files & model weights
└── README.md               # Documentation & HF Space configuration
```

---

## 🚀 Local Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sharathgowdaur-jpg/Text-Summarizer-Web-Application.git
   cd Text-Summarizer-Web-Application
   ```

2. **Create & Activate Virtual Environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Local Server:**
   ```bash
   python app.py
   ```

5. **Open in Browser:**
   Navigate to `http://127.0.0.1:8000`

---

## 📡 API Reference

### 1. Home Page Route
Returns the Neumorphic HTML interface.

- **URL:** `/`
- **Method:** `GET`
- **Response:** `200 OK` (HTML Document)

### 2. Text Summarization Endpoint
Accepts raw text/dialogue and returns the generated AI summary.

- **URL:** `/summarize/`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "dialogue": "Hannah: Hey, do you have the notes from yesterday's class? Mark: Yes, I sent them to your email earlier. Hannah: Thanks! I will check now."
  }
  ```
- **Response:** `200 OK`
  ```json
  {
    "summary": "Mark sent yesterday's class notes to Hannah's email."
  }
  ```

---

## ☁️ Deployment Guide

### Deploying to Vercel

1. Push code to GitHub (ensuring `.gitignore` excludes heavy local weight folders).
2. Connect repository at [vercel.com/new](https://vercel.com/new).
3. Vercel automatically uses `vercel.json` and `api/index.py`.

### Deploying to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Select **Docker** as Space SDK and **CPU Basic (Free)** hardware.
3. Push your repository to the Hugging Face Space remote.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

Developed by **Sharath Gowda**
- GitHub: [@sharathgowdaur-jpg](https://github.com/sharathgowdaur-jpg)
