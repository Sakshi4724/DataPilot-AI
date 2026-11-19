
import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st

from preprocessing_agent import show_preprocessing_agent
from ai_analyzer import generate_ai_insights
from data_quality import show_data_quality


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DataPilot AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. DYNAMIC NEURAL NETWORK BACKGROUND
# --------------------------------------------------

st.html("""
<style>
.stApp {
    background: #03121e;
    color: white;
}

#neural-particles {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
}

#neural-particles canvas {
    width: 100% !important;
    height: 100% !important;
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stSidebar"] {
    position: relative;
    z-index: 1;
}
</style>

<div id="neural-particles"></div>

<script>
(async function () {
    if (window.__datapilotParticlesStarted) return;
    window.__datapilotParticlesStarted = true;

    function loadScript(url) {
        return new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = url;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    try {
        await loadScript(
            "https://cdn.jsdelivr.net/npm/@tsparticles/engine@4/tsparticles.engine.min.js"
        );

        await loadScript(
            "https://cdn.jsdelivr.net/npm/@tsparticles/slim@4/tsparticles.slim.bundle.min.js"
        );

        await loadSlim(tsParticles);

        await tsParticles.load({
            id: "neural-particles",
            options: {
                fullScreen: {
                    enable: false
                },
                background: {
                    color: "#03121e"
                },
                fpsLimit: 45,
                particles: {
                    number: {
                        value: 85,
                        density: {
                            enable: true,
                            area: 900
                        }
                    },
                    color: {
                        value: ["#00fff0", "#35cfff", "#76fff8"]
                    },
                    shape: {
                        type: "circle"
                    },
                    opacity: {
                        value: {
                            min: 0.35,
                            max: 0.85
                        }
                    },
                    size: {
                        value: {
                            min: 1,
                            max: 3
                        }
                    },
                    links: {
                        enable: true,
                        distance: 155,
                        color: "#00e5e5",
                        opacity: 0.35,
                        width: 1
                    },
                    move: {
                        enable: true,
                        speed: 2.5,
                        direction: "none",
                        random: true,
                        straight: false,
                        outModes: {
                            default: "bounce"
                        }
                    }
                },
                interactivity: {
                    events: {
                        onHover: {
                            enable: true,
                            mode: "grab"
                        }
                    },
                    modes: {
                        grab: {
                            distance: 190,
                            links: {
                                opacity: 0.8
                            }
                        }
                    }
                },
                detectRetina: true
            }
        });

    } catch (error) {
        console.error("DataPilot animation failed:", error);
        window.__datapilotParticlesStarted = false;
    }
})();
</script>
""", unsafe_allow_javascript=True)


# --------------------------------------------------
# 3. DATASET LOADING
# --------------------------------------------------

@st.cache_data(show_spinner="Reading dataset...")
def load_dataset(file_bytes, filename):

    stream = BytesIO(file_bytes)

    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(stream, engine="openpyxl")

    return pd.read_csv(stream, encoding="utf-8-sig")


# --------------------------------------------------
# 4. DATASET SUMMARY
# --------------------------------------------------

@st.cache_data(show_spinner="Preparing dataset summary...")
def get_dataset_summary(df):

    numeric_df = df.select_dtypes(include="number")

    return {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "statistical_summary": (
            numeric_df.describe().round(2).to_dict()
            if not numeric_df.empty
            else {}
        ),
    }


# --------------------------------------------------
# 5. CUSTOM DASHBOARD STYLING
# --------------------------------------------------

st.markdown("""
<style>

:root {
    color-scheme: dark;
}

.stApp {
    background: #03121e;
    color: #e6f8fc;
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stSidebar"] {
    position: relative;
    z-index: 1;
}

[data-testid="stSidebar"] {
    background: rgba(3,20,33,.93);
    border-right: 1px solid rgba(63,228,225,.24);
}

[data-testid="stHeader"] {
    background: rgba(3,20,33,.30);
}

[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    max-width: 1380px;
}

h1, h2, h3 {
    color: #d7fcff !important;
    letter-spacing: .015em;
}

h1 {
    text-shadow: 0 0 25px rgba(0,229,224,.25);
}

[data-testid="stMetric"] {
    background: rgba(5,30,46,.82);
    border: 1px solid rgba(54,216,218,.28);
    border-radius: 15px;
    padding: 16px;
    backdrop-filter: blur(9px);
}

[data-testid="stMetricValue"] {
    color: #64f5ed;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: rgba(4,24,38,.85);
    border-radius: 12px;
}

[data-testid="stForm"] {
    background: rgba(5,28,43,.85);
    border: 1px solid rgba(56,205,209,.25);
    border-radius: 16px;
    padding: 16px;
}

div.stButton > button[kind="primary"],
div.stFormSubmitButton > button {
    background: #087d8a;
    color: white;
    border: 1px solid #44d9dc;
    border-radius: 9px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    border-color: #7bffff;
    color: #7bffff;
}

.hero {
    background: linear-gradient(
        110deg,
        rgba(2,20,36,.87),
        rgba(0,91,104,.46)
    );
    border: 1px solid rgba(73,239,234,.33);
    border-radius: 20px;
    padding: 25px 28px;
    margin-bottom: 22px;
    box-shadow: 0 14px 40px rgba(0,0,0,.18);
    backdrop-filter: blur(8px);
}

.hero .eyebrow {
    color: #65efe8;
    font-size: .85rem;
    letter-spacing: .16em;
    font-weight: 700;
}

.hero h1 {
    margin: .3rem 0;
    font-size: 2.45rem;
}

.hero p {
    color: #c0e5ed;
    margin: 0;
}

@media (prefers-reduced-motion: reduce) {
    #neural-particles {
        display: none;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">INTELLIGENT ANALYTICS PLATFORM</div>
    <h1>◈ DataPilot AI</h1>
    <p>
        Transform datasets into clarity with interactive
        analytics and local AI insights.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 7. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.title("◈ DataPilot AI")
    st.caption("NEURAL ANALYTICS WORKSPACE")

    section = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🛡️ Data Quality",
            "🤖 Data Preprocessing Agent",
            "🤖 AI Insights"
        ],
        key="dashboard_section"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="dataset_upload"
    )

    st.caption("All processing stays on your local application.")


# --------------------------------------------------
# 8. MAIN DASHBOARD
# --------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel dataset using the sidebar to begin."
    )

else:

    try:

        file_bytes = uploaded_file.getvalue()

        dataset_key = hashlib.sha256(file_bytes).hexdigest()

        if st.session_state.get("active_dataset_key") != dataset_key:

            st.session_state["active_dataset_key"] = dataset_key

            for key in (
                "ai_insights",
                "cleaning_result",
                "cleaning_operations",
                "cleaning_settings"
            ):
                st.session_state.pop(key, None)

            # Clear previous preprocessing agent results
            for key in list(st.session_state.keys()):
                if key.startswith((
                    "preprocessing_result_",
                    "preprocessing_log_"
                )):
                    st.session_state.pop(key, None)

        df = load_dataset(file_bytes, uploaded_file.name)

        st.caption(f"Active dataset: {uploaded_file.name}")


        # ------------------------------------------
        # OVERVIEW
        # ------------------------------------------

        if section == "🏠 Overview":

            st.header("Dataset Overview")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Records",
                f"{df.shape[0]:,}"
            )

            c2.metric(
                "Columns",
                df.shape[1]
            )

            c3.metric(
                "Missing Values",
                f"{int(df.isna().sum().sum()):,}"
            )

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.subheader("Column Details")

            column_info = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values
            })

            st.dataframe(
                column_info,
                use_container_width=True
            )

            
            st.subheader("Statistical Summary")

            # Select numerical columns
            numeric_summary_df = df.select_dtypes(include="number")

            # Exclude identifier columns
            identifier_columns = {
                "id",
                "customerid",
                "invoiceid",
                "invoiceno",
                "stockcode",
                "productid"
            }

            numeric_summary_df = numeric_summary_df.loc[
                :,
                [
                    col for col in numeric_summary_df.columns
                    if str(col).lower().replace("_", "").replace(" ", "")
                    not in identifier_columns
                ]
            ]

            if not numeric_summary_df.empty:

                summary = numeric_summary_df.describe().round(2)

                st.dataframe(
                    summary,
                    use_container_width=True
                )

            else:
                st.info("No numerical measurement columns available.")




        # ------------------------------------------
        # DATA QUALITY
        # ------------------------------------------

        elif section == "🛡️ Data Quality":

            show_data_quality(df)


        # ------------------------------------------
        # DATA PREPROCESSING AGENT
        # ------------------------------------------

        elif section == "🤖 Data Preprocessing Agent":

            show_preprocessing_agent(
                df,
                dataset_key=dataset_key
            )

        


        # ------------------------------------------
        # AI INSIGHTS
        # ------------------------------------------

        elif section == "🤖 AI Insights":

            st.header("🤖 AI-Powered Dataset Insights")

            st.write(
                "Generate a natural-language report "
                "using the local pretrained model."
            )

            if st.button(
                "Generate AI Insights",
                key="generate_ai_insights"
            ):

                with st.spinner("AI is analyzing your dataset..."):

                    st.session_state["ai_insights"] = (
                        generate_ai_insights(
                            get_dataset_summary(df),
                            df=df
                        )
                    )


            if st.session_state.get("ai_insights"):

                st.markdown(
                    st.session_state["ai_insights"]
                )


    except Exception as exc:

        st.error(
            f"An error occurred while processing the dataset: {exc}"
        )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DataPilot AI | Final Year AI & DS Project"
)
