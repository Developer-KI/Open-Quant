"""Terminal-style CSS shared across all app pages."""
import streamlit as st

_CSS = """
<style>
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    button[data-baseweb="tab"] {
        font-size: 12px !important;
        height: 30px !important;
        padding: 0px 12px !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #9ba3b8;
    }
    [data-testid="stMetricValue"] {
        font-size: 17px !important;
        font-weight: 700 !important;
        font-family: 'Courier New', Courier, monospace;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    h3 {
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 250px !important;
    }
    .stMainBlockContainer {
        transition: margin-left 0.3s ease-in-out;
    }
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
