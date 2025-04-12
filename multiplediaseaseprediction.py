import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import google.generativeai as genai

# Load models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# Initialize Gemini
import google.generativeai as genai

# Configure Gemini with your API key
genai.configure(api_key=st.secrets["gemini_api_key"])  # or paste your key directly

# Create Gemini model instance
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

def get_chatbot_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


# Sidebar menu
with st.sidebar:
    selected = option_menu(
    'AI MEDICAL ASSISTANT',
    ['AI Medical Chatbot', 'AI Image Analyzer', 'PDF Report Summarizer','Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson’s Prediction'],
    icons=['robot', 'image', 'file-earmark-text','activity', 'heart', 'person'],
    default_index=0
)
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=200)

# Diabetes
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')

    st.subheader("Enter Patient Details:")
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input('🤰 Number of Pregnancies', min_value=0)
        Glucose = st.number_input('🧪 Glucose Level', min_value=0)
        BloodPressure = st.number_input('🩺 Blood Pressure', min_value=0)
        SkinThickness = st.number_input('🧍 Skin Thickness', min_value=0)

    with col2:
        Insulin = st.number_input('💉 Insulin Level', min_value=0)
        BMI = st.number_input('⚖️ BMI', min_value=0.0)
        DiabetesPedigreeFunction = st.number_input('🧬 Pedigree Function', min_value=0.0)
        Age = st.number_input('🎂 Age', min_value=0)

    if st.button('🔍 Predict Diabetes'):
        diabetes_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                                        BMI, DiabetesPedigreeFunction, Age]])
        if diabetes_prediction[0] == 1:
            st.error('🚨 The person is likely **diabetic**.')
        else:
            st.success('✅ The person is **not diabetic**.')

# Heart
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')
    st.subheader("Fill in the patient’s details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('🎂 Age', min_value=0)
        sex = st.selectbox('🚻 Sex', ['Female', 'Male'])
        sex = 1 if sex == 'Male' else 0
        cp = st.selectbox('💓 Chest Pain Type', [0, 1, 2, 3])

    with col2:
        trestbps = st.number_input('🩸 Resting BP', min_value=0)
        chol = st.number_input('🧪 Cholesterol (mg/dl)', min_value=0)
        fbs = st.selectbox('🍬 Fasting Sugar > 120 mg/dl?', ['No', 'Yes'])
        fbs = 1 if fbs == 'Yes' else 0

    with col3:
        restecg = st.selectbox('📊 ECG Results', [0, 1, 2])
        thalach = st.number_input('💗 Max Heart Rate', min_value=0)
        exang = st.selectbox('🏃‍♂️ Exercise Angina?', ['No', 'Yes'])
        exang = 1 if exang == 'Yes' else 0

    oldpeak = st.number_input('📉 ST Depression', min_value=0.0)
    slope = st.selectbox('📈 ST Slope', [0, 1, 2])
    ca = st.selectbox('🔬 Major Vessels (0–3)', [0, 1, 2, 3])
    thal = st.selectbox('🧬 Thalassemia Type', [0, 1, 2, 3])

    if st.button('🔍 Predict Heart Disease'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,
                                                 thalach, exang, oldpeak, slope, ca, thal]])
        if heart_prediction[0] == 1:
            st.error("✅ The person **does not have heart disease**.")
        else:
            st.success("🚨 The person **has heart disease**.")

# Parkinson’s
if selected == 'Parkinson’s Prediction':
    st.title("🧬 Parkinson’s Disease Prediction using ML")
    st.subheader("Provide acoustic details below:")

    input_labels = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    user_inputs = []
    for i in range(0, len(input_labels), 2):
        col1, col2 = st.columns(2)
        with col1:
            user_inputs.append(st.number_input(input_labels[i]))
        if i+1 < len(input_labels):
            with col2:
                user_inputs.append(st.number_input(input_labels[i+1]))

    if st.button("🔍 Predict Parkinson’s"):
        parkinsons_prediction = parkinsons_model.predict([user_inputs])
        if parkinsons_prediction[0] == 1:
            st.error("✅ The person **does not have Parkinson’s disease**.")
        else:
            st.success("🚨 The person **has Parkinson’s disease**.")
            
                
# AI Image Analyzer (Image + Text)
if selected == 'AI Image Analyzer':
    st.title("🖼️ AI Image Analyzer")
    st.markdown("Upload a medical image and describe the symptoms. The AI will analyze both and provide a diagnosis-oriented response.")

    uploaded_image = st.file_uploader("📤 Upload a Medical Image", type=["jpg", "jpeg", "png"])
    symptom_description = st.text_area("📝 Describe the patient's symptoms or your concern")

    if uploaded_image and symptom_description:
        st.image(uploaded_image, caption="Uploaded Image")

        with st.spinner("Analyzing image and context... 🤖"):

            try:
                from PIL import Image
                import io

                image = Image.open(uploaded_image).convert('RGB')

                # Send multimodal input to Gemini
                response = model.generate_content([
                    symptom_description,
                    image
                ])

                st.success("**AI Medical Insight:**")
                st.write(response.text)

            except Exception as e:
                st.error(f"Error during analysis: {e}")

    elif uploaded_image and not symptom_description:
        st.info("✍️ Please describe the symptoms or issue for better analysis.")
    elif symptom_description and not uploaded_image:
        st.info("📸 Please upload an image to analyze.")



# Medical Chatbot
if selected == 'AI Medical Chatbot':
    st.title("🤖 AI Medical Assistant")
    st.markdown("Ask any health-related question and get AI-generated responses!")

    user_input = st.text_input("💬 Ask your medical question:")

    if user_input:
        with st.spinner("Thinking... 💭"):
            reply = get_chatbot_response(user_input)
        st.success("**AI Assistant:**")
        st.write(reply)

# PDF Report Summarizer
if selected == 'PDF Report Summarizer':
    st.title("📄 PDF Medical Report Summarizer")
    st.markdown("Upload a medical report in PDF format. The AI will extract and summarize key insights.")

    uploaded_pdf = st.file_uploader("📤 Upload PDF Report", type=["pdf"])

    if uploaded_pdf:
        import fitz  # PyMuPDF
        from transformers import pipeline

        try:
            # Extract text from PDF
            with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()

            st.info("📑 **Extracted Text:**")
            st.write(text[:1000] + "...")  # Show first 1000 characters

            # Summarization
            with st.spinner("Summarizing report... 🧠"):
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                summary = summarizer(text, max_length=300, min_length=30, do_sample=False)

            st.success("📝 **Summary:**")
            st.write(summary[0]['summary_text'])

        except Exception as e:
            st.error(f"Error: {e}")


st.markdown(
    """
    <style>
    .developer-note {
        position: fixed;
        bottom: 10px;
        right: 15px;
        font-size: 14px;
        color: gray;
        z-index: 9999;
    }
    </style>
    <div class="developer-note">
        Developed by <strong>Arshiya Nandy</strong>
    </div>
    """,
    unsafe_allow_html=True
)



