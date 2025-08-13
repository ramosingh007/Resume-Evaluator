from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image
from datetime import datetime
import torch
import os
import io

# ==== CONFIG ====
MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
output_file = "CV_evaluations.txt"

# ==== LOAD MODEL ====
print(" Loading model... this may take a while.")
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
FastVisionModel.for_inference(model)
print("✅ Model loaded.")

# ==== PROFESSIONS & PROMPTS ====
PROFESSION_PROMPTS = {
    "SDE": """You are an experienced Software Development Engineer recruiter in a top MNC.
            
            Output strictly in JSON format with the following keys:
            {
            "score": <score_out_of_100>,
            "selection_status": "Selected" or "Not Selected",
            "feedback": "<short constructive feedback>"
            }
            Evaluate the provided CV for this role.
            Consider:
            - Technical skills (programming, frameworks, tools)
            - Projects and achievements
            - Education background
            - Work experience or internships
            - Problem-solving ability
            - Communication & leadership skills
            Output strictly in JSON format: { "score": <0-100>, "selection_status": "Selected"/"Not Selected", "feedback": "<reason>" }""",
    "AI Engineer": """You are an experienced AI/ML recruiter.
            Output strictly in JSON format with the following keys:
            {
            "score": <score_out_of_100>,
            "selection_status": "Selected" or "Not Selected",
            "feedback": "<short constructive feedback>"
            }
            Evaluate the provided CV for this AI Engineer role.
            Consider:
            - AI/ML skills, frameworks (TensorFlow, PyTorch)
            - Relevant projects & research
            - Education
            - Experience in deploying AI systems
            - Publications or Kaggle competitions
            Output strictly in JSON format: { "score": <0-100>, "selection_status": "Selected"/"Not Selected", "feedback": "<reason>" }""",
    "Full Stack": """You are a senior recruiter for a Full Stack Developer position.
            Output strictly in JSON format with the following keys:
            {
            "score": <score_out_of_100>,
            "selection_status": "Selected" or "Not Selected",
            "feedback": "<short constructive feedback>"
            }
            Evaluate the CV considering:
            - Frontend & Backend skills
            - Projects & deployments
            - Database & API experience
            - UI/UX awareness
            Output strictly in JSON format: { "score": <0-100>, "selection_status": "Selected"/"Not Selected", "feedback": "<reason>" }""",
    "Management": """You are hiring for a Management position.
            Output strictly in JSON format with the following keys:
            {
            "score": <score_out_of_100>,
            "selection_status": "Selected" or "Not Selected",
            "feedback": "<short constructive feedback>"
            }
            Evaluate the CV based on:
            - Leadership experience
            - Strategic planning
            - Decision making
            - Team management
            Output strictly in JSON format: { "score": <0-100>, "selection_status": "Selected"/"Not Selected", "feedback": "<reason>" }""",
    "Reporter": """You are hiring for a News Reporter role.
            Output strictly in JSON format with the following keys:
            {
            "score": <score_out_of_100>,
            "selection_status": "Selected" or "Not Selected",
            "feedback": "<short constructive feedback>"
            }
            Evaluate the CV based on:
            - Communication skills
            - Journalism background
            - Investigative skills
            - Prior reporting experience
            Output strictly in JSON format: { "score": <0-100>, "selection_status": "Selected"/"Not Selected", "feedback": "<reason>" }""",
    "Anchor": """You are hiring for a TV Anchor role.
            Output strictly in JSON format with the following keys:
            {
            "score": <score_out_of_100>,
            "selection_status": "Selected" or "Not Selected",
            "feedback": "<short constructive feedback>"
            }
            Evaluate based on:
            - Presentation skills
            - Voice modulation
            - On-camera experience
            - Language fluency
            Output strictly in JSON format: { "score": <0-100>, "selection_status": "Selected"/"Not Selected", "feedback": "<reason>" }""",
    "Actor": """You are hiring for an Acting role.
            Output strictly in JSON format with the following keys:
            {
            "score": <score_out_of_100>,
            "selection_status": "Selected" or "Not Selected",
            "feedback": "<short constructive feedback>"
            }
            Evaluate the CV based on:
            - Acting experience
            - Theatre/Film background
            - Awards or recognitions
            Output strictly in JSON format: { "score": <0-100>, "selection_status": "Selected"/"Not Selected", "feedback": "<reason>" }""",
    "Lawyer": """You are hiring for a Lawyer position.
            Output strictly in JSON format with the following keys:
            {
            "score": <score_out_of_100>,
            "selection_status": "Selected" or "Not Selected",
            "feedback": "<short constructive feedback>"
            }
            Evaluate the CV considering:
            - Legal education
            - Case experience
            - Court exposure
            - Client handling skills
            Output strictly in JSON format: { "score": <0-100>, "selection_status": "Selected"/"Not Selected", "feedback": "<reason>" }"""
}

# ==== FASTAPI APP ====
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def form_page():
    professions = "".join([f'<option value="{p}">{p}</option>' for p in PROFESSION_PROMPTS])
    return f"""
    <html>
    <head>
        <title>CV Evaluation</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(to right, #6a11cb, #2575fc);
                color: white;
                text-align: center;
                padding: 50px;
            }}
            h2 {{
                font-size: 2.5em;
                margin-bottom: 20px;
                color: #FFD700;
            }}
            form {{
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                display: inline-block;
                text-align: left;
                box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
            }}
            label {{
                font-size: 1.2em;
                font-weight: bold;
                display: block;
                margin-top: 15px;
            }}
            select, input[type="file"] {{
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                border-radius: 8px;
                border: none;
                font-size: 1em;
            }}
            button {{
                background-color: #FFD700;
                color: black;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                margin-top: 20px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: background 0.3s ease;
            }}
            button:hover {{
                background-color: #ffcc00;
            }}
        </style>
    </head>
    <body>
        <h2>📄 CV Evaluation Portal</h2>
        <form action="/evaluate" method="post" enctype="multipart/form-data">
            <label for="profession">Select Profession:</label>
            <select name="profession" required>{professions}</select>

            <label for="file">Upload CV Image:</label>
            <input type="file" name="file" accept="image/*" required>

            <button type="submit">🚀 Submit</button>
        </form>
    </body>
    </html>
    """

@app.post("/evaluate")
async def evaluate(profession: str = Form(...), file: UploadFile = File(...)):
    if profession not in PROFESSION_PROMPTS:
        return JSONResponse({"error": "Invalid profession selected"}, status_code=400)

    # Load image
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = image.resize((512, 512))

    # Prompt
    instruction = PROFESSION_PROMPTS[profession]

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction.strip()}
        ]}
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate output
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output_tokens = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=4096,
        use_cache=True,
        temperature=0.3,
        min_p=0.1
    )

    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Save result
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Profession: {profession}\n")
        f.write("Result:\n")
        f.write(output_text.strip() + "\n")
        f.write("-" * 80 + "\n\n")

    return HTMLResponse(f"""
        <html>
        <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(to right, #00c6ff, #0072ff);
                color: white;
                padding: 50px;
                text-align: center;
            }}
            pre {{
                background-color: rgba(0,0,0,0.5);
                padding: 20px;
                border-radius: 10px;
                text-align: left;
                color: #FFD700;
            }}
            a {{
                color: #FFD700;
                text-decoration: none;
                font-weight: bold;
            }}
        </style>
        </head>
        <body>
            <h3>Result for {profession} Role</h3>
            <p>{output_text}</p>
            <a href="/">⬅ Go Back</a>
        </body>
        </html>
    """)

# Run: uvicorn app:app --reload
