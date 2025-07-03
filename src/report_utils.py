from fpdf import FPDF
from datetime import datetime

def generate_pdf(prediction, confidence, measures, medicine, patient_name="Unknown", filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Neuro Shield Health Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Diagnosis: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}", ln=True)

    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Precautionary Measures:\n{measures}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Suggested Medications:\n{medicine}")

    pdf.output(filename)
    return filename
