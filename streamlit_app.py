
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8080"

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide"
)

# -------------------------------
# Custom Styling
# -------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0E1117;
}

/* Headers */
.main-title {
    text-align: center;
    color: white;
    font-size: 3rem;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #B0B0B0;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Card */
.card {
    background-color: #1A1F2B;
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid #2D3748;
}

/* Result Card */
.result-card {
    background-color: #1F77B4;
    padding: 1rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}

/* Metrics */
.metric-card {
    background-color: #1A1F2B;
    padding: 1rem;
    border-radius: 12px;
    border-left: 5px solid #00D4AA;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Backend Connection
# -------------------------------

try:
    methods_response = requests.get(
        f"{API_BASE_URL}/methods",
        timeout=5
    )
    methods_response.raise_for_status()
    METHODS = methods_response.json()["methods"]

except Exception as e:
    st.error(f"Backend unavailable: {e}")
    st.stop()

# -------------------------------
# Header
# -------------------------------

st.markdown(
    '<div class="main-title">🔍 Deepfake Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered image authenticity verification</div>',
    unsafe_allow_html=True
)

# -------------------------------
# Layout
# -------------------------------

left, right = st.columns([1, 1])

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    selected_method = st.selectbox(
        "Detection Method",
        METHODS
    )

    predict_button = st.button(
        "🚀 Analyze Image",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right:

    if uploaded_file:
        st.image(
            uploaded_file,
            caption="Preview",
            use_container_width=True
        )

# -------------------------------
# Prediction
# -------------------------------

if predict_button and uploaded_file:

    with st.spinner("Running Deepfake Analysis..."):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        data = {
            "method": selected_method
        }

        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            data=data,
            timeout=300
        )

        if response.status_code == 200:

            result = response.json()

            st.markdown("---")

            st.markdown(
                f"""
                <div class="result-card">
                    <h2>{result['label'].upper()}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Prediction",
                result["label"].upper()
            )

            c2.metric(
                "Confidence",
                f"{result['confidence']*100:.2f}%"
            )

            c3.metric(
                "Accuracy",
                f"{result['reported_accuracy']:.2f}%"
            )

            with st.expander("Raw Response"):
                st.json(result)

        else:
            st.error(response.text)

