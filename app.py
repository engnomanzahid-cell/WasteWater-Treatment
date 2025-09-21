import gradio as gr
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from groq import Groq
import pandas as pd
import os

# ✅ Setup Groq API client (reads key from environment variable on HF)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "demo_key_replace"))

# ---------------------------
# Mock pollutant database
# ---------------------------
treatment_db = {
    "lead": {
        "methods": ["Chemical Precipitation", "Ion Exchange", "Membrane Filtration"],
        "efficiency": [85, 90, 95],
        "details": "Lead can be effectively treated using lime precipitation, ion-exchange resins, or RO membranes.",
        "cost_per_m3": [0.8, 1.5, 2.2],
        "sludge_kg_per_m3": [0.12, 0.05, 0.02]
    },
    "arsenic": {
        "methods": ["Coagulation–Filtration", "Adsorption", "Reverse Osmosis"],
        "efficiency": [80, 88, 96],
        "details": "Arsenic removal requires multi-barrier approaches including iron-based adsorbents and RO systems.",
        "cost_per_m3": [0.9, 1.2, 2.5],
        "sludge_kg_per_m3": [0.15, 0.07, 0.03]
    },
    "chromium": {
        "methods": ["Reduction + Precipitation", "Activated Carbon Adsorption", "Electrochemical Treatment"],
        "efficiency": [82, 87, 93],
        "details": "Chromium (VI) is reduced to Cr (III) before precipitation, or handled with advanced adsorption/EC.",
        "cost_per_m3": [1.0, 1.4, 2.0],
        "sludge_kg_per_m3": [0.18, 0.06, 0.04]
    }
}

# ---------------------------
# Core functions
# ---------------------------
def recommend_treatment(pollutant, flow_rate):
    pollutant = pollutant.lower().strip()
    if pollutant not in treatment_db:
        return f"❌ No data for '{pollutant}'. Try lead, arsenic, chromium.", None, None

    rec = treatment_db[pollutant]

    # 📊 Efficiency Bar Chart
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(rec["methods"], rec["efficiency"], color=["#5DADE2", "#48C9B0", "#F5B041"])
    ax.set_ylabel("Removal Efficiency (%)")
    ax.set_title(f"Treatment methods for {pollutant.capitalize()}")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)

    # 📑 Text summary
    text = f"""
<div style="background:#f5f5f5; padding:15px; border-radius:10px; color:#2C3E50">
<h3>💧 Recommended Treatment for {pollutant.capitalize()}</h3>
<ul>
<li><b>Methods:</b> {', '.join(rec['methods'])}</li>
<li><b>Efficiency (%):</b> {rec['efficiency']}</li>
<li><b>Technical Note:</b> {rec['details']}</li>
<li><b>Flow Rate Consideration:</b> For {flow_rate} m³/day, costs and sludge must be considered.</li>
</ul>
</div>
"""

    # 📊 Cost & Sludge Estimation Table
    data = {
        "Method": rec["methods"],
        "Efficiency (%)": rec["efficiency"],
        "Cost ($/m³)": rec["cost_per_m3"],
        "Daily Cost ($)": [round(c * flow_rate, 2) for c in rec["cost_per_m3"]],
        "Sludge (kg/m³)": rec["sludge_kg_per_m3"],
        "Sludge (kg/day)": [round(s * flow_rate, 2) for s in rec["sludge_kg_per_m3"]],
    }
    df = pd.DataFrame(data)

    return text, fig, df


def explain_recommendation(pollutant, flow_rate):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user", "content": f"Explain in technical detail how to treat {pollutant} in wastewater with flow rate {flow_rate} m³/day."}],
            temperature=0.6,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI explanation unavailable: {e}"


def extract_pdf_text(pdf_file):
    if not pdf_file:
        return "No file uploaded."
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text[:2000] + "..."


# ---------------------------
# Dashboard Layout
# ---------------------------
with gr.Blocks(css="""
.gradio-container { font-family: 'Segoe UI', sans-serif; }
h1 { font-size: 38px; text-align: center; color:#154360; margin-bottom: 20px; }
h2, h3 { color: #2C3E50; }
.sidebar { background:#ECF0F1; padding:15px; border-radius:12px; }
.card { background:#FDFEFE; padding:12px; border-radius:10px; margin:10px; border: 1px solid #D5DBDB; }
""") as demo:

    gr.Markdown("<h1>🌊 Wastewater AI – Industrial Treatment Advisor</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div class='sidebar'><h2>⚙️ Input Panel</h2><p>Provide pollutant & flow rate</p></div>")
            pollutant = gr.Textbox(label="Enter pollutant", placeholder="e.g., lead, arsenic, chromium")
            flow_rate = gr.Number(label="Flow rate (m³/day)", value=100)
            btn = gr.Button("🔍 Get Recommendation")
            explain_btn = gr.Button("📘 AI Explanation")
            pdf_input = gr.File(label="Upload WHO/EPA PDF", type="filepath")
        with gr.Column(scale=2):
            output_text = gr.HTML()
            output_plot = gr.Plot()
            output_table = gr.Dataframe(interactive=False, label="Cost & Sludge Estimation")
            explanation_out = gr.Markdown()
            pdf_output = gr.Textbox(label="Extracted Guideline", lines=10)

    btn.click(recommend_treatment, inputs=[pollutant, flow_rate], outputs=[output_text, output_plot, output_table])
    explain_btn.click(explain_recommendation, inputs=[pollutant, flow_rate], outputs=explanation_out)
    pdf_input.change(extract_pdf_text, inputs=pdf_input, outputs=pdf_output)

demo.launch()
