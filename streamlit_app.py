import requests
import streamlit as st
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8080"

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #0a0f1e 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem; max-width: 100% !important; }

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg, #0f1729 0%, #131d35 60%, #0c1526 100%);
    border-bottom: 1px solid rgba(99,179,237,0.15);
    padding: 2.5rem 2rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.25);
    color: #63b3ed;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0 0 0.5rem;
}
.hero-title span {
    background: linear-gradient(90deg, #63b3ed, #76e4f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    color: #718096;
    font-size: 0.95rem;
    margin: 0 0 1.5rem;
}
.team-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    padding: 1.25rem 2rem 1.5rem;
    background: rgba(255,255,255,0.02);
    border-top: 1px solid rgba(255,255,255,0.05);
}
.team-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
}
.team-members {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    justify-content: center;
}
.member-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #a0aec0;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.28rem 0.75rem;
    border-radius: 100px;
}
.member-chip::before {
    content: '';
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #63b3ed;
    flex-shrink: 0;
}
.mentor-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
}
.mentor-label { color: #4a6fa5; font-weight: 400; margin-right: 0.15rem; }

/* ── Tabs ── */
[data-testid="stTabs"] { margin-top: 0.5rem; }
[data-testid="stTabs"] > div:first-child {
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0 2rem;
    gap: 0;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #4a5568 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.9rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #63b3ed !important;
    border-bottom: 2px solid #63b3ed !important;
    background: transparent !important;
}
[data-testid="stTabContent"] { padding: 2rem; }

/* ── Cards ── */
.panel {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.6rem;
}
.card-heading {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 1rem;
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(99,179,237,0.25) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,179,237,0.5) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
[data-testid="stSelectbox"] label {
    color: #718096 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
[data-testid="stMultiSelect"] label {
    color: #718096 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(99,179,237,0.15) !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    color: #63b3ed !important;
    border-radius: 6px !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2b6cb0, #2c7a7b) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.4rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #3182ce, #319795) !important;
    transform: translateY(-1px) !important;
}

/* ── Preview empty state ── */
.preview-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    border: 1.5px dashed rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 3rem;
    min-height: 220px;
    text-align: center;
}

/* ── Result cards ── */
.result-fake {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 1px solid rgba(245,101,101,0.3);
    border-left: 4px solid #e53e3e;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
}
.result-real {
    background: linear-gradient(135deg, #0a1a0f, #0d2714);
    border: 1px solid rgba(72,187,120,0.3);
    border-left: 4px solid #38a169;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
}
.result-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.3rem; }
.result-label.fake { color: #fc8181; }
.result-label.real { color: #68d391; }
.result-verdict { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
.result-verdict.fake { color: #feb2b2; }
.result-verdict.real { color: #9ae6b4; }
.result-desc { font-size: 0.8rem; margin-top: 0.3rem; opacity: 0.8; }
.result-desc.fake { color: #fc8181; }
.result-desc.real { color: #68d391; }

/* ── Metric row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 10px;
    margin-top: 0.9rem;
}
.metric-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-value { font-size: 1.35rem; font-weight: 700; color: #e2e8f0; line-height: 1.2; }
.metric-name { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #4a5568; margin-top: 0.2rem; }

/* ── Comparison table ── */
.compare-table-wrap {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    overflow: hidden;
    margin-top: 1.5rem;
}
.compare-table-header {
    padding: 1.2rem 1.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
}
table.ctable {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
table.ctable thead tr {
    background: rgba(255,255,255,0.03);
}
table.ctable thead th {
    padding: 0.8rem 1.2rem;
    text-align: left;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #718096;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
table.ctable tbody tr {
    border-bottom: 1px solid rgba(255,255,255,0.04);
    transition: background 0.15s;
}
table.ctable tbody tr:hover { background: rgba(255,255,255,0.02); }
table.ctable tbody tr.winner-row {
    background: rgba(99,179,237,0.05);
    border-left: 3px solid #63b3ed;
}
table.ctable tbody td {
    padding: 0.85rem 1.2rem;
    color: #a0aec0;
    vertical-align: middle;
}
table.ctable tbody td.model-name { color: #e2e8f0; font-weight: 600; }
.badge-fake {
    display: inline-block;
    background: rgba(229,62,62,0.15);
    border: 1px solid rgba(229,62,62,0.3);
    color: #fc8181;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}
.badge-real {
    display: inline-block;
    background: rgba(56,161,105,0.15);
    border: 1px solid rgba(56,161,105,0.3);
    color: #68d391;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}
.badge-winner {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    margin-left: 0.5rem;
    letter-spacing: 0.05em;
}
.conf-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.conf-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
    min-width: 60px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #2b6cb0, #63b3ed);
}
.conf-bar-fill.high { background: linear-gradient(90deg, #276749, #68d391); }
.summary-box {
    margin: 1rem 1.6rem 1.4rem;
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 10px;
}
.summary-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.summary-stat .val { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; }
.summary-stat .lbl { font-size: 0.62rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #4a5568; margin-top: 0.2rem; }

[data-testid="stImage"] img { border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.07) !important; }
[data-testid="stExpander"] { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 10px !important; }
[data-testid="stAlert"] { border-radius: 10px !important; border-left-width: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🔍 AI-Powered Forensics</div>
    <h1 class="hero-title">Deepfake <span>Detection</span> System</h1>
    <p class="hero-subtitle">Upload an image to verify its authenticity using advanced neural network models</p>
    <div class="team-section">
        <div class="team-label">Project Team</div>
        <div class="team-members">
            <span class="member-chip">Liyakat Hussain</span>
            <span class="member-chip">Vanshika Vaidya</span>
            <span class="member-chip">Gunuru Hemanth Kumar</span>
            <span class="member-chip">Navroop Singh</span>
            <span class="member-chip">Anupam Rathore</span>
        </div>
        <span class="mentor-chip">
            <span class="mentor-label">Mentor</span> Mr. Bappaditya Chakraborty
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Backend ──
try:
    methods_response = requests.get(f"{API_BASE_URL}/methods", timeout=5)
    methods_response.raise_for_status()
    METHODS = methods_response.json()["methods"]
except Exception as e:
    st.error(f"⚠️ Cannot reach backend — make sure the server is running at {API_BASE_URL}")
    st.stop()

# ── Tabs ──
tab1, tab2 = st.tabs(["🔍  Single Model Analysis", "⚖️  Compare Models"])

# ═══════════════════════════════════════════════
# TAB 1 — Single Model
# ═══════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">📁 Upload Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "drop", type=["jpg","jpeg","png"], label_visibility="collapsed", key="single_upload"
        )
        st.markdown("<div style='margin-top:0.3rem;color:#4a5568;font-size:0.7rem;text-align:center'>JPG, PNG · Max 200 MB</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card-heading">⚙️ Detection Model</div>', unsafe_allow_html=True)
        selected_method = st.selectbox("Model", METHODS, label_visibility="collapsed", key="single_method")

        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Analyze Image", use_container_width=True, key="single_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🖼️ Image Preview</div>', unsafe_allow_html=True)
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
        else:
            st.markdown("""
            <div class="preview-empty">
                <div style="font-size:2.2rem;opacity:0.25">🖼️</div>
                <div style="color:#4a5568;font-size:0.82rem">No image uploaded yet</div>
                <div style="color:#2d3748;font-size:0.74rem">Preview will appear here</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn and uploaded_file:
        with st.spinner("Running deepfake analysis…"):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"method": selected_method},
                    timeout=300
                )
                if resp.status_code == 200:
                    result = resp.json()
                    is_fake = result["label"].lower() == "fake"
                    cc = "result-fake" if is_fake else "result-real"
                    lc = "fake" if is_fake else "real"
                    icon = "⚠️" if is_fake else "✅"
                    verdict = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC IMAGE"
                    desc = "Signs of AI manipulation or synthetic generation detected." if is_fake else "No manipulation artifacts found. Image appears genuine."

                    st.markdown(f"""
                    <div class="{cc}">
                        <div class="result-label {lc}">{icon} Verdict</div>
                        <div class="result-verdict {lc}">{verdict}</div>
                        <div class="result-desc {lc}">{desc}</div>
                    </div>
                    <div class="metrics-row">
                        <div class="metric-box">
                            <div class="metric-value">{result['label'].upper()}</div>
                            <div class="metric-name">Prediction</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{result['confidence']*100:.1f}%</div>
                            <div class="metric-name">Confidence</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{result['reported_accuracy']:.1f}%</div>
                            <div class="metric-name">Model Accuracy</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                    with st.expander("View Raw API Response"):
                        st.json(result)
                else:
                    st.error(f"Analysis failed: {resp.text}")
            except Exception as e:
                st.error(f"Request error: {e}")
    elif predict_btn:
        st.warning("⚠️ Please upload an image before running analysis.")

# ═══════════════════════════════════════════════
# TAB 2 — Compare Models
# ═══════════════════════════════════════════════
with tab2:
    cl, cr = st.columns([1, 1], gap="large")

    with cl:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">📁 Upload Image</div>', unsafe_allow_html=True)

        cmp_file = st.file_uploader(
            "drop2", type=["jpg","jpeg","png"], label_visibility="collapsed", key="cmp_upload"
        )
        st.markdown("<div style='margin-top:0.3rem;color:#4a5568;font-size:0.7rem;text-align:center'>JPG, PNG · Max 200 MB</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card-heading">⚙️ Select Models to Compare</div>', unsafe_allow_html=True)
        selected_models = st.multiselect(
            "Models", METHODS,
            default=METHODS[:3] if len(METHODS) >= 3 else METHODS,
            label_visibility="collapsed",
            key="cmp_models"
        )
        st.markdown("<div style='margin-top:0.3rem;color:#4a5568;font-size:0.7rem'>Select 2 or more models to compare side-by-side</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
        cmp_btn = st.button("⚖️ Run Comparison", use_container_width=True, key="cmp_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    with cr:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🖼️ Image Preview</div>', unsafe_allow_html=True)
        if cmp_file:
            st.image(cmp_file, use_container_width=True)
        else:
            st.markdown("""
            <div class="preview-empty">
                <div style="font-size:2.2rem;opacity:0.25">🖼️</div>
                <div style="color:#4a5568;font-size:0.82rem">No image uploaded yet</div>
                <div style="color:#2d3748;font-size:0.74rem">Preview will appear here</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if cmp_btn and cmp_file and len(selected_models) >= 2:
        results = []
        errors = []
        progress = st.progress(0, text="Starting comparison…")

        for i, model in enumerate(selected_models):
            progress.progress((i) / len(selected_models), text=f"Running {model}…")
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/predict",
                    files={"file": (cmp_file.name, cmp_file.getvalue(), cmp_file.type)},
                    data={"method": model},
                    timeout=300
                )
                if resp.status_code == 200:
                    r = resp.json()
                    results.append({
                        "model": model,
                        "label": r["label"].upper(),
                        "confidence": round(r["confidence"] * 100, 1),
                        "accuracy": round(r["reported_accuracy"], 1),
                    })
                else:
                    errors.append(f"{model}: {resp.text}")
            except Exception as e:
                errors.append(f"{model}: {e}")

        progress.progress(1.0, text="Done!")
        progress.empty()

        if errors:
            for err in errors:
                st.warning(f"⚠️ {err}")

        if results:
            best = max(results, key=lambda x: x["confidence"])
            fake_count = sum(1 for r in results if r["label"] == "FAKE")
            real_count = len(results) - fake_count
            avg_conf = round(sum(r["confidence"] for r in results) / len(results), 1)
            consensus = "FAKE" if fake_count > real_count else ("REAL" if real_count > fake_count else "SPLIT")

            cons_color = "#fc8181" if consensus == "FAKE" else ("#68d391" if consensus == "REAL" else "#f6ad55")

            rows_html = ""
            for r in results:
                is_winner = r["model"] == best["model"]
                row_class = "winner-row" if is_winner else ""
                badge = '<span class="badge-winner">★ BEST</span>' if is_winner else ""
                label_badge = f'<span class="badge-fake">FAKE</span>' if r["label"] == "FAKE" else f'<span class="badge-real">REAL</span>'
                bar_pct = r["confidence"]
                bar_class = "high" if bar_pct >= 70 else ""
                rows_html += f"""
                <tr class="{row_class}">
                    <td class="model-name">{r['model']}{badge}</td>
                    <td>{label_badge}</td>
                    <td>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-bg">
                                <div class="conf-bar-fill {bar_class}" style="width:{bar_pct}%"></div>
                            </div>
                            <span style="color:#e2e8f0;font-weight:600;min-width:42px">{bar_pct}%</span>
                        </div>
                    </td>
                    <td style="color:#e2e8f0;font-weight:500">{r['accuracy']}%</td>
                </tr>
                """

            st.markdown(f"""
            <div class="compare-table-wrap">
                <div class="compare-table-header">📊 Model Comparison Results</div>
                <div class="summary-box">
                    <div class="summary-stat">
                        <div class="val">{len(results)}</div>
                        <div class="lbl">Models Tested</div>
                    </div>
                    <div class="summary-stat">
                        <div class="val" style="color:{cons_color}">{consensus}</div>
                        <div class="lbl">Consensus</div>
                    </div>
                    <div class="summary-stat">
                        <div class="val">{avg_conf}%</div>
                        <div class="lbl">Avg Confidence</div>
                    </div>
                    <div class="summary-stat">
                        <div class="val" style="color:#63b3ed">{best['model']}</div>
                        <div class="lbl">Most Confident</div>
                    </div>
                </div>
                <table class="ctable">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Verdict</th>
                            <th>Confidence</th>
                            <th>Model Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            with st.expander("📋 Export raw comparison data"):
                df = pd.DataFrame(results)
                df.columns = ["Model", "Verdict", "Confidence (%)", "Model Accuracy (%)"]
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download as CSV",
                    data=csv,
                    file_name="model_comparison.csv",
                    mime="text/csv"
                )

    elif cmp_btn and not cmp_file:
        st.warning("⚠️ Please upload an image before running the comparison.")
    elif cmp_btn and len(selected_models) < 2:
        st.warning("⚠️ Please select at least 2 models to compare.")

# ── Footer ──
st.markdown("""
<div style="text-align:center;padding:2rem 1rem 1rem;border-top:1px solid rgba(255,255,255,0.04);margin-top:2rem;">
    <div style="color:#2d3748;font-size:0.7rem;letter-spacing:0.05em">
        Deepfake Detection System · Built with Streamlit & PyTorch
    </div>
</div>
""", unsafe_allow_html=True)