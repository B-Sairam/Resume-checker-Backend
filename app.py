from flask import Flask, json, request, jsonify
import pdfplumber
from flask_cors import CORS
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from os import getenv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes





# Configure Gemini API
genai.configure(api_key=getenv("GEMINI_API_KEY"))

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip() if text else None

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Missing file or job description"}), 400

    resume_file = request.files['resume']
    job_description = request.form['job_description']

    if resume_file:
        try:
            if resume_file.filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(resume_file)
                if not resume_text:
                    return jsonify({"error": "Could not extract text from PDF"}), 400
            else:
                resume_text = resume_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            return jsonify({"error": f"Failed to process resume: {str(e)}"}), 500
    else:
        return jsonify({'error': 'No resume file uploaded'}), 400  # Bad Request

    # Create a prompt for the AI model
    prompt = f"""
    Analyze the following resume against this job description.

    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Provide:
    - A relevance score (0-100%)
    - Key missing skills
    - Suggestions to improve the resume
    """
    print(prompt)
    try:
        # Create the model
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type = content.Type.OBJECT,
            enum = [],
            required = ["matching_score", "missing_skill", "Suggestions"],
            properties = {
            "matching_score": content.Schema(
                type = content.Type.NUMBER,
            ),
            "missing_skill": content.Schema(
                type = content.Type.ARRAY,
                items = content.Schema(
                type = content.Type.STRING,
                ),
            ),
            "Suggestions": content.Schema(
                type = content.Type.STRING,
            ),
            },
        ),
        "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        system_instruction="you are resume review",
        )

        chat_session = model.start_chat(
        history=[
            
        ]
        )

        response = chat_session.send_message(prompt)
        parsed_data = json.loads(response.text)
        
        return jsonify({"analysis":parsed_data }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
        print(getenv("GEMINI_API_KEY"))
        return "Hello, World!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True, use_reloader=False)

