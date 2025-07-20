# app.py

import gradio as gr
from PIL import Image
from src.model_utils import load_model, predict_image
from src.eeg_utils import extract_signal_from_image, predict_from_signal
from src.report_utils import generate_pdf
from ragbot import EEGAlzheimersRAG  # ✅ Import FAISS-based RAG chatbot

# ===============================
# Load Models
# ===============================
alz_model = load_model()
rag_bot = EEGAlzheimersRAG("neurology_faq.csv", use_sentence_transformers=True)

# ===============================
# 📚 FAISS Chatbot Interface
# ===============================
def rag_chatbot(user_question):
    return rag_bot.answer_question(user_question)

rag_chatbot_tab = gr.Interface(
    fn=rag_chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask about EEG, Alzheimer's, seizures...", label="Your Question"),
    outputs=gr.Textbox(label="Neurobot's Answer", lines=12, show_copy_button=True),
    title="📚 Neurology Q&A Chatbot (FAISS)"
)

# ===============================
# 🎥 Learn Tab
# ===============================
def show_local_video(disease):
    if disease == "Alzheimer's":
        return "videos/alz_video.mp4"
    elif disease == "Epilepsy":
        return "videos/epilepsy_video.mp4"
    return None

learn_tab = gr.Interface(
    fn=show_local_video,
    inputs=gr.Radio(["Alzheimer's", "Epilepsy"], label="Select Disease to Learn More"),
    outputs=gr.Video(label="Explainer Video"),
    title="🎥 Learn About Brain Disorders"
)

# ===============================
# 🧠 Alzheimer’s MRI Diagnosis
# ===============================
def diagnose_alz(img, name):
    pred = predict_image(alz_model, img)

    measures = {
        "Very Mild": "Engage in mental exercises and maintain healthy sleep.",
        "Mild": "Use memory aids and follow routines.",
        "Moderate": "Supervision needed, ensure emotional support.",
        "Severe": "Full-time care required. Focus on comfort.",
        "No Impairment": "Maintain a healthy mental and physical lifestyle."
    }

    meds = {
        "Very Mild": "Monitor only.",
        "Mild": "Donepezil",
        "Moderate": "Memantine + Donepezil",
        "Severe": "Supervised medication.",
        "No Impairment": "None"
    }

    file_path = f"{name.replace(' ', '_')}_alz_report.pdf"
    generate_pdf(pred, 0.95, measures.get(pred, "N/A"), meds.get(pred, "N/A"), name, filename=file_path)
    return f"🧠 Prediction: {pred}", file_path

alz_tab = gr.Interface(
    fn=diagnose_alz,
    inputs=[
        gr.Image(type="filepath", label="Upload MRI Image"),
        gr.Text(label="Patient Name")
    ],
    outputs=[
        gr.Text(label="Diagnosis Result"),
        gr.File(label="Download Report")
    ],
    title="🧠 Alzheimer's Stage Detector"
)

# ===============================
# ⚡ Epilepsy Seizure Prediction
# ===============================
def diagnose_eeg(img, name):
    signal = extract_signal_from_image(Image.open(img))
    pred = predict_from_signal(signal)
    label = "SEIZURE" if pred else "NON-SEIZURE"

    info = {
        "SEIZURE": {
            "measures": "Avoid stress, bright lights, and ensure regular sleep.",
            "medicine": "Levetiracetam or Lamotrigine"
        },
        "NON-SEIZURE": {
            "measures": "No action needed. Maintain regular lifestyle.",
            "medicine": "None"
        }
    }

    file_path = f"{name.replace(' ', '_')}_epilepsy_report.pdf"
    generate_pdf(label, 0.91, info[label]["measures"], info[label]["medicine"], name, filename=file_path)
    return f"⚡ Prediction: {label}", file_path

eeg_tab = gr.Interface(
    fn=diagnose_eeg,
    inputs=[
        gr.Image(type="filepath", label="Upload EEG Graph Image"),
        gr.Text(label="Patient Name")
    ],
    outputs=[
        gr.Text(label="Diagnosis Result"),
        gr.File(label="Download Report")
    ],
    title="⚡ Epileptic Seizure Detector"
)

# ===============================
# 🧠 Final App with All Tabs
# ===============================
gr.TabbedInterface(
    interface_list=[learn_tab, alz_tab, eeg_tab, rag_chatbot_tab],
    tab_names=["📘 Learn", "🧠 Alzheimer's", "⚡ Epilepsy", "📚 Chatbot"],
    title="🛡️ Neuro Shield: Brain Health AI"
).launch()
