# 🌟 AI Resume Analyzer 🔘

The **AI Resume Analyzer** is an intelligent, interactive Streamlit-based application that helps job seekers evaluate and improve their resumes by comparing them to specific job descriptions using state-of-the-art AI models like **LLAMA-3 (via Groq)** and **BERT embeddings**.

---

## 🔍 What Does This Project Do?

### 1. 📝 Resume Text Extraction
- Upload your resume in PDF format.
- The app extracts raw text using `pdfminer.six`.

### 2. 📄 Job Description Input
- Paste a job description for your targeted job role (optional).
- If skipped, a generic Software Engineer JD is used.

### 3. 📊 ATS Similarity Score
- Uses **BERT-based sentence transformers** to measure how well your resume matches the job description.
- A percentage similarity score is shown.

### 4. 🤖 AI-Powered Evaluation
- Uses **Groq's LLAMA3-70B** model to deeply analyze your resume.
- Provides detailed review with scores, emoji indicators (✅, ❌, ⚠️), and helpful suggestions.

### 5. 📈 Skill Radar Chart
- Extracts skills from both the JD and resume using the LLM.
- Displays a radar chart showing overlapping and missing skills.

### 6. 🧠 Voice-Powered Q&A
- Ask questions about your resume via text input or microphone.
- Uses **faster-whisper** to transcribe audio.
- LLAMA3 gives instant intelligent feedback based on your resume.

### 7. 📥 Downloadable Report
- Receive a complete text report with recommendations.
- Download and keep track of improvements over time.

---

## 🎯 Why Use This Tool?

- ✅ **ATS Optimization** – Improve your chances of getting past recruiters and bots.
- 🤖 **AI-Driven Feedback** – No need for expensive human consultants.
- 📌 **Visual Skill Gaps** – Instantly see what skills you’re missing.
- 🔁 **Iterative Improvements** – Update and reanalyze your resume anytime.

---

## ⚙️ Installation Steps

Make sure you have **Python 3.9+**, **pip**, and **Git** installed.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
