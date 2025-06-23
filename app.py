import os
import csv
import time
import random
from PIL import Image
from datetime import datetime
from fpdf import FPDF
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st

# Cached model loader
@st.cache_resource
def load_model_once():
    return load_model("models/best_model.h5")

model = load_model_once()

# Streamlit config
st.set_page_config(page_title="Solar Panel Dust Detection", layout="wide")
st.title("Solar Panel Dust Detection")
st.caption("Model version: v1.0 – EfficientNet-B3 F1-optimised")

# Helper Functions
def preprocess(img):
    img = img.resize((256, 256))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def estimate_dust_coverage(uploaded_image):
    image = Image.open(uploaded_image).convert("L")
    image = np.array(image.resize((256, 256)))
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dust_percent = (np.sum(thresh == 0) / thresh.size) * 100
    return round(dust_percent, 2)

def estimate_impact(dust_percent):
    energy_loss_pct = min(dust_percent * 1.25, 100)
    weekly_loss_rm = energy_loss_pct * 0.5 * 7
    return round(energy_loss_pct, 2), round(weekly_loss_rm, 2)

def cleaning_suggestion(dust_percent):
    if dust_percent >= 35:
        return "Immediate cleaning required!"
    elif dust_percent >= 20:
        return "Cleaning recommended within 3 days."
    elif dust_percent >= 10:
        return "Monitor regularly. Clean in 5–7 days."
    else:
        return "Clean. No action needed."

def cost_comparison(dust_percent):
    manual_cost = 100 * 4
    ai_freq = 2 if dust_percent > 30 else 1
    ai_cost = 100 * ai_freq
    energy_loss_pct = min(dust_percent * 1.25, 100)
    monthly_energy_cost = energy_loss_pct * 0.5 * 7 * 4
    return {
        "manual_cost": manual_cost,
        "ai_cost": ai_cost,
        "energy_loss": round(monthly_energy_cost, 2),
        "savings": manual_cost - ai_cost
    }

def plot_costs(costs):
    fig, ax = plt.subplots()
    bars = ax.bar(['Manual', 'AI-Based'], [costs['manual_cost'], costs['ai_cost']])
    ax.set_title("Monthly Cleaning Cost Comparison")
    ax.set_ylabel("Cost (RM)")
    for bar in bars:
        ax.annotate(f"RM {bar.get_height()}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    return fig

def analyze_trend(values):
    if len(values) < 2:
        return "Insufficient data"
    avg_diff = sum([values[i+1]-values[i] for i in range(len(values)-1)]) / (len(values)-1)
    return "Dust is rapidly accumulating" if avg_diff > 2 else "Dust is decreasing" if avg_diff < -2 else "Dust level is stable"

def simulate_vibration_cleaning(dust_percent):
    efficiency = random.uniform(0.60, 0.70)
    return round(dust_percent * (1 - efficiency), 2)

def log_cleaning_action(panel_name, before, after):
    record = {
        "Panel": panel_name,
        "Dust Before": before,
        "Dust After": after,
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Method": "Vibration"
    }
    file = "vibration_cleaning_log.csv"
    with open(file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if os.path.getsize(file) == 0:
            writer.writeheader()
        writer.writerow(record)

def generate_pdf_report(panel_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Solar Panel Dust Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    for panel in panel_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, f"{panel['Panel']} Summary", ln=True)
        pdf.set_font("Arial", size=11)
        for k in ['Dust %', 'Energy Loss %', 'Cost Impact (RM)', 'Cleaning Suggestion']:
            pdf.cell(200, 10, f"{k}: {panel[k]}", ln=True)
        pdf.ln(8)
    name = f"Solar_Report_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(10,99)}.pdf"
    pdf.output(name)
    return name

# UI sections with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Single Panel Check", "Panel Dashboard", "Real-Time Feed", "Cleaning Logs"])

with tab1:
    st.header("Single Panel Prediction")
    uploaded_file = st.file_uploader("Choose a solar panel image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        try:
            img_input = preprocess(image_pil)
            prediction = model.predict(img_input)[0][0]
            st.write(f"Predicted Dust Level: **{prediction:.2%}**")
            if prediction > 0.30:
                st.error("WARNING: Your panel likely needs cleaning.")
                after_cleaning = simulate_vibration_cleaning(prediction * 100)
                st.success(f"Vibration Cleaning Simulated: {round(prediction*100,2)}% → {after_cleaning}%")
                log_cleaning_action("Single Upload", round(prediction*100,2), after_cleaning)
            else:
                st.success("Your panel is clean.")
        except Exception as e:
            st.exception("Prediction failed. Please check the model and input.")

with tab2:
    st.header("Multiple Panel Dashboard")
    num_panels = st.number_input("Number of panels to monitor", min_value=1, max_value=10, value=3)
    panel_data = []
    for i in range(int(num_panels)):
        panel_name = st.text_input(f"Panel {i+1} name", value=f"Panel {i+1}")
        uploaded_img = st.file_uploader(f"Upload image for {panel_name}", type=["jpg", "png", "jpeg"], key=f"panel{i}")
        if uploaded_img:
            dust_percent = estimate_dust_coverage(uploaded_img)
            eff_loss, cost_loss = estimate_impact(dust_percent)
            suggestion = cleaning_suggestion(dust_percent)
            panel_data.append({
                "Panel": panel_name,
                "Dust %": dust_percent,
                "Energy Loss %": eff_loss,
                "Cost Impact (RM)": cost_loss,
                "Cleaning Suggestion": suggestion
            })
    if panel_data:
        df = pd.DataFrame(panel_data)
        st.dataframe(df)
        if st.button("Generate PDF Report"):
            report_file = generate_pdf_report(panel_data)
            with open(report_file, "rb") as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name=report_file,
                    mime="application/pdf"
                )

with tab3:
    st.header("Simulated Real-Time Monitoring")

    image_folder = "realtime_images"

    # create the folder if it doesn't exist
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        st.info(f"Folder '{image_folder}' created. "
                "Drop JPG/PNG images into it and press **Restart Simulation**.")
        st.stop()                     # stop execution until images are added

    # refresh file list
    image_files = sorted(
        [f for f in os.listdir(image_folder)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    # handle empty folder
    if not image_files:
        st.warning("No images found in the 'realtime_images' folder.")
        if st.button("Retry"):
            st.experimental_rerun()
        st.stop()

    # session-state index initialisation
    if "index" not in st.session_state:
        st.session_state.index = 0

    # show next image
    if st.session_state.index < len(image_files):
        image_path = os.path.join(image_folder, image_files[st.session_state.index])
        image      = Image.open(image_path).convert("RGB")
        st.image(image, caption=f"Image {st.session_state.index + 1}", use_column_width=True)

        dust_percent = estimate_dust_coverage(image)
        eff_loss, cost_loss = estimate_impact(dust_percent)
        suggestion = cleaning_suggestion(dust_percent)

        st.markdown(f"**Estimated Dust Coverage:** {dust_percent}%")
        st.markdown(f"**Predicted Energy Loss:** {eff_loss}%")
        st.markdown(f"**Estimated Cost Impact:** RM {cost_loss}")
        st.markdown(f"**Cleaning Suggestion:** {suggestion}")

        if dust_percent > 30:
            after_cleaning = simulate_vibration_cleaning(dust_percent)
            st.success(f"Vibration Cleaning Triggered – Dust reduced to {after_cleaning}%")
            log_cleaning_action(f"Image {st.session_state.index+1}", dust_percent, after_cleaning)

        time.sleep(3)
        st.session_state.index += 1
        st.experimental_rerun()
    else:
        st.success("Monitoring completed.")
        if st.button("Restart Simulation"):
            st.session_state.index = 0
            st.experimental_rerun()

with tab4:
    st.header("Vibration Cleaning Logs")
    if os.path.exists("vibration_cleaning_log.csv"):
        df = pd.read_csv("vibration_cleaning_log.csv")
        st.dataframe(df)
        fig, ax = plt.subplots()
        ax.plot(df['Time'], df['Dust Before'], label='Before')
        ax.plot(df['Time'], df['Dust After'], label='After')
        ax.set_title("Dust Reduction from Vibration Cleaning")
        ax.set_xlabel("Time")
        ax.set_ylabel("Dust %")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No cleaning logs found. Run a simulation first.")
