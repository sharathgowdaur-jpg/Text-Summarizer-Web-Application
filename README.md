# 🧠 AI Text Summarizer

> **An AI-powered full-stack web application that transforms long text, articles, emails, and conversations into concise, readable summaries.**
>
> Live Demo : https://text-summarizer-web-application.vercel.app/

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.1+-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge\&logo=huggingface\&logoColor=black)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge\&logo=javascript\&logoColor=black)
![Vercel](https://img.shields.io/badge/Vercel-Deployment-000000?style=for-the-badge\&logo=vercel\&logoColor=white)

---

## 📌 Overview

**Text Summarizer** is a full-stack AI application designed to reduce lengthy content into short, meaningful summaries.

It can process:

* 📄 Long documents
* 📰 Articles
* 📧 Emails
* 💬 Conversations
* 📝 General text
* 👥 Multi-person dialogues

The project combines a **modern Neumorphic UI**, **FastAPI backend**, and **AI/NLP summarization engines** to provide a simple and fast summarization experience.

### 🎯 Why This Project?

Reading large amounts of information takes time. This application helps users quickly understand the most important information without reading every sentence.

**Core objectives:**

* ⏱️ Save reading time
* 🎯 Extract important information
* 🧹 Remove unnecessary or repetitive content
* 🧠 Generate meaningful summaries
* 💻 Provide a simple and responsive interface
* ☁️ Support both local and cloud deployment

---

# ✨ Key Features

### 🤖 AI-Powered Summarization

Uses a fine-tuned **Google T5 Transformer model** for abstractive text summarization in local deployment.

### ⚡ Dual Summarization Architecture

The project supports two execution approaches:

| Mode           | Technology              | Purpose                              |
| -------------- | ----------------------- | ------------------------------------ |
| 🖥️ Local Mode | Fine-tuned T5 + PyTorch | Neural abstractive summarization     |
| ☁️ Cloud Mode  | Sentence ranking / NLP  | Lightweight serverless summarization |

### 🎨 Neumorphic UI

The frontend uses a **Soft UI / Neumorphism** design with:

* Smooth rounded cards
* Extruded shadows
* Minimal interface
* Responsive layout
* Clear input/output sections
* Interactive buttons

### ⚡ Asynchronous Requests

JavaScript uses `async/await` and `fetch()` to communicate with the FastAPI backend without refreshing the page.

### 🧹 Text Preprocessing

Before summarization, the backend cleans the input by handling:

* HTML tags
* Extra whitespace
* Unnecessary line breaks
* Formatting noise

### ☁️ Multiple Deployment Options

The project is structured to support:

* Local development
* Vercel
* Hugging Face Spaces
* Docker-based deployment

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      USER           │
                    │  Enters Long Text   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   NEUMORPHIC UI     │
                    │   HTML + CSS + JS    │
                    └──────────┬──────────┘
                               │
                         POST /summarize/
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FASTAPI         │
                    │      BACKEND        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  TEXT PREPROCESSING │
                    │ HTML / Whitespace   │
                    │ Cleaning            │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
            ┌─────────────────┐   ┌──────────────────┐
            │   LOCAL MODE    │   │    CLOUD MODE    │
            │                 │   │                  │
            │ Fine-tuned T5   │   │ Sentence Ranking │
            │ PyTorch         │   │ NLP              │
            └────────┬────────┘   └────────┬─────────┘
                     │                     │
                     └──────────┬──────────┘
                                ▼
                    ┌─────────────────────┐
                    │  GENERATED SUMMARY  │
                    └──────────┬──────────┘
                               │
                         JSON Response
                               │
                               ▼
                    ┌─────────────────────┐
                    │   OUTPUT BOX        │
                    │  Displays Summary   │
                    └─────────────────────┘
```

---

# 🔄 How It Works

### 1️⃣ User Input

The user enters or pastes a long piece of text into the input area.

### 2️⃣ Summarize Request

When the **Summarize** button is clicked, JavaScript sends an asynchronous HTTP `POST` request to:

```text
/summarize/
```

### 3️⃣ Backend Processing

FastAPI receives the request and validates the input using **Pydantic**.

The text is then cleaned and prepared for summarization.

### 4️⃣ AI Processing

Depending on the deployment environment, the application uses one of its summarization engines.

#### 🖥️ Local T5 Pipeline

```text
Input Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Fine-Tuned T5 Model
    ↓
4-Beam Search
    ↓
Generated Tokens
    ↓
Decoded Summary
```

#### ☁️ Cloud NLP Pipeline

```text
Input Text
    ↓
Sentence Splitting
    ↓
Word Frequency / Density Analysis
    ↓
Sentence Weighting
    ↓
Important Sentences
    ↓
Condensed Summary
```

### 5️⃣ API Response

FastAPI returns a JSON response:

```json
{
  "summary": "Generated summary appears here."
}
```

### 6️⃣ UI Rendering

JavaScript receives the response and dynamically displays the generated summary.

---

# 🛠️ Tech Stack

## 🎨 Frontend

* **HTML5** — Page structure
* **CSS3** — Neumorphic / Soft UI design
* **JavaScript ES6+** — API communication and dynamic UI updates
* **Fetch API** — Asynchronous HTTP requests

## ⚡ Backend

* **Python 3.10+**
* **FastAPI** — REST API framework
* **Uvicorn** — ASGI server
* **Pydantic** — Request validation
* **Jinja2** — HTML templating

## 🧠 AI / NLP

* **Google T5**
* **Hugging Face Transformers**
* **PyTorch**
* **Sentence Ranking**
* **Natural Language Processing**

## ☁️ Deployment

* **Vercel**
* **Hugging Face Spaces**
* **Docker**
* **Git**
* **GitHub**

---

# 📂 Project Structure

```text
text-summarizer/
│
├── 📁 api/
│   └── index.py              # Vercel serverless API
│
├── 📁 templates/
│   └── index.html             # Web interface
│
├── 📁 static/
│   ├── style.css              # Neumorphic styling
│   └── script.js              # Frontend JavaScript
│
├── 📁 model/                  # Local model files / weights
│
├── app.py                     # Local FastAPI application
├── Dockerfile                 # Container configuration
├── vercel.json                # Vercel configuration
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

> Adjust the structure above to match your actual repository before publishing.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the FastAPI application with Uvicorn:

```bash
uvicorn app:app --reload
```

Open your browser:

```text
http://127.0.0.1:8000
```

You can now enter text and generate a summary.

---

# 🧪 API Usage

### Endpoint

```http
POST /summarize/
```

### Request

```json
{
  "text": "Your long text goes here..."
}
```

### Response

```json
{
  "summary": "Your generated summary..."
}
```

---

# 🧠 Local T5 Model Pipeline

The local implementation uses a sequence-to-sequence Transformer architecture.

```text
                Input Text
                    │
                    ▼
             T5 Tokenizer
                    │
                    ▼
             Token IDs
                    │
                    ▼
          Fine-Tuned T5 Model
                    │
                    ▼
             Beam Search
             (4 Beams)
                    │
                    ▼
           Generated Tokens
                    │
                    ▼
              Detokenizer
                    │
                    ▼
               Summary
```

This approach can generate **abstractive summaries**, meaning the model can produce new wording rather than simply copying sentences from the original text.

---

# ☁️ Cloud Deployment

The application can be configured for serverless deployment using **Vercel**.

Typical structure:

```text
Vercel
  │
  ├── vercel.json
  │
  └── api/
       └── index.py
```

For environments where running large PyTorch model weights is impractical, the cloud implementation can use a lightweight NLP-based summarization approach.

> ⚠️ Serverless platforms have execution-time, memory, package-size, and runtime limitations. The exact deployment behavior depends on the platform configuration and current service limits.

---

# 🐳 Docker

Build the Docker image:

```bash
docker build -t text-summarizer .
```

Run the container:

```bash
docker run -p 8000:8000 text-summarizer
```

Then open:

```text
http://localhost:8000
```

---

# 📊 Local vs Cloud

| Feature             |           Local |                 Cloud |
| ------------------- | --------------: | --------------------: |
| T5 Transformer      |               ✅ | Depends on deployment |
| PyTorch             |               ✅ |     May be restricted |
| Large Model Weights |               ✅ | ⚠️ Platform dependent |
| Lightweight NLP     |        Optional |                     ✅ |
| Serverless Friendly |               ❌ |                     ✅ |
| Internet Required   | Not necessarily |               Usually |
| Docker Support      |               ✅ |                     ✅ |
| FastAPI             |               ✅ |                     ✅ |

---

# 🎨 UI Design

The interface follows a **Neumorphic / Soft UI** design philosophy.

### Design Characteristics

```css
background: #E0E5EC;
border-radius: 16px;
box-shadow:
    9px 9px 16px rgba(163, 177, 198, 0.6),
    -9px -9px 16px rgba(255, 255, 255, 0.7);
```

The design focuses on:

* Minimal visual noise
* Soft depth
* Consistent spacing
* Rounded components
* Clear input/output separation
* Responsive interaction

---

# 🔐 Input Processing & Validation

The backend validates incoming requests before processing them.

Typical flow:

```text
Request
   ↓
Pydantic Validation
   ↓
Text Cleaning
   ↓
Length / Content Checks
   ↓
Summarization
   ↓
JSON Response
```

This helps prevent malformed requests from reaching the model pipeline.

---

# 📈 Future Improvements

The project can be extended with:

* 🔊 Text-to-Speech summary
* 🎤 Voice input
* 📄 PDF summarization
* 📑 DOCX summarization
* 🌐 URL/article summarization
* 🧠 Larger Transformer models
* 🌍 Multilingual summarization
* 📊 Summary quality evaluation
* 👤 User accounts
* 💾 Summary history
* 📱 Improved mobile UI
* ⚡ Streaming summary generation
* 🔍 Key-point extraction
* 🏷️ Automatic topic detection

---

# 🧪 Evaluation

Possible evaluation metrics for future model benchmarking include:

* **ROUGE-1**
* **ROUGE-2**
* **ROUGE-L**
* **BLEU**
* **BERTScore**
* Human readability evaluation
* Compression ratio
* Inference latency

Example:

```text
Original Text
      │
      ├── Length: 1000 words
      │
      ▼
Generated Summary
      │
      └── Length: 180 words

Compression Ratio ≈ 18%
```

---

# 🌟 Project Highlights

| Area          | Highlight                           |
| ------------- | ----------------------------------- |
| 🤖 AI         | Fine-tuned T5 Transformer           |
| 🧠 NLP        | Extractive + abstractive approaches |
| ⚡ Backend     | FastAPI                             |
| 🎨 Frontend   | Vanilla HTML/CSS/JavaScript         |
| 🎭 Design     | Neumorphic Soft UI                  |
| ☁️ Deployment | Vercel + Docker + Hugging Face      |
| 🔄 API        | REST-based JSON communication       |
| 🐍 Language   | Python 3.10+                        |

---

# 💡 What Makes This Project Interesting?

### 🔹 1. Full-Stack AI

The project combines frontend development, backend APIs, NLP, deep learning, and deployment into a single application.

### 🔹 2. Dual Execution Strategy

Instead of relying exclusively on a large Transformer model, the architecture provides a lightweight alternative for constrained cloud environments.

### 🔹 3. Practical Use Case

Text summarization has applications in:

* 📚 Education
* 📰 News
* 💼 Business
* 📧 Email management
* 📝 Research
* 💬 Meeting/conversation analysis
* 📄 Document processing

### 🔹 4. Lightweight Frontend

The interface is built using vanilla technologies instead of heavy frontend frameworks, keeping the client-side architecture simple.

---

# 🛡️ Limitations

The quality of generated summaries depends on:

* Input length
* Model quality
* Training data
* Domain-specific terminology
* Deployment resources
* Cloud/serverless limitations

Very long or highly specialized documents may require chunking, larger context models, or document-level summarization pipelines.

---

# 🔮 Future Vision

The long-term goal is to evolve this project from a simple text summarizer into a **multi-format AI information compression platform** capable of understanding documents, conversations, webpages, and multimedia content.

```text
                    ┌─────────────────────┐
                    │     AI SUMMARIZER   │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
       Documents           Websites          Conversations
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
                       🧠 AI Processing
                               │
                               ▼
                       🎯 Key Information
                               │
                               ▼
                       ✨ Smart Summary
```

---

# 🤝 Contributing

Contributions are welcome!

```bash
# Fork the repository
# Create a feature branch
git checkout -b feature/new-feature

# Make your changes
git add .

# Commit
git commit -m "Add new feature"

# Push
git push origin feature/new-feature
```

Then open a Pull Request.

---

# 📜 License

This project is intended for educational and development purposes.

Add your preferred license here, such as **MIT License**, before publishing the repository.

---

# 👨‍💻 Author

### **Sharath U R**

AI/ML Engineering Student passionate about:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 💻 Software Development
* 📊 Data Science
* 🧩 Problem Solving
* 🚀 Building practical AI applications

---

# 🤝 Contributors

A huge thank you to everyone who has contributed to this project! 💛

---

## 🌟 Krupa B T — *Briefly AI Redesign*

| | |
|---|---|
| **GitHub** | [@krupabt](https://github.com/krupabt) |
| **Contribution** | [PR #1 — `feature/briefly-ai-redesign`](https://github.com/sharathgowdaur-jpg/Text-Summarizer-Web-Application/pull/1) |
| **Merged** | ✅ October 2026 |

### 🛠️ What Krupa contributed:

* 🎨 **Briefly AI Redesign** — Completely redesigned the workspace with a modern **"Briefly AI"** branding and enhanced layout
* ⚙️ **Enhanced Summarization Controls** — Added advanced controls for summarization length, style, and output configuration
* 🧪 **Test Suite** — Added comprehensive test coverage for the summarization features
* 🔧 **Improved UX** — Better user experience with cleaner interaction patterns and feedback mechanisms

> *"A fresh and modern take on the summarizer — the Briefly AI redesign brings a polished, professional feel to the application."*

---

### 🤝 Want to Be Listed Here?

We’d love your help! Check the [Contributing](#-contributing) section and open a PR — all contributors get credited right here. 🌟

---

# ⭐ Support

If you find this project useful:

**⭐ Star the repository**
**🍴 Fork the project**
**💡 Share your feedback**
**🤝 Contribute improvements**

---

## 🚀 Built with Python, FastAPI, T5, PyTorch & Vanilla JavaScript

> **Turn long text into clear ideas — faster.** 🧠✨
