# 📄 Resume Evaluator (FastAPI + LLM Vision)

This is a web application that evaluates resume **images** for different professions using **FastAPI** and **Unsloth's Vision-Language model (Qwen2.5-VL-7B)**.


## 🚀 Features

- Upload a CV/resume as an image (PNG, JPG, etc.)
- Choose a profession (SDE, AI Engineer, Lawyer, etc.)
- Get a score and AI-generated feedback in JSON format
- Saves evaluation history in a local file (`CV_evaluations.txt`)


## 🔧 Tech Stack

- **FastAPI** — Web backend
- **Unsloth Vision LLM** — for multimodal evaluation
- **Transformers / Torch** — for model execution
- **Pillow** — to handle uploaded images


## 🛠 How to Run Locally

1. **Clone the repo** (if not uploading manually)
2. 
```bash
git clone https://github.com/yourusername/resume-evaluator.git
cd resume-evaluator
