import streamlit as st

def render_footer():
    """Render consistent footer across pages"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        Made with ❤️ for language learners and researchers
    </div>
    """, unsafe_allow_html=True)

def render_help(text: str):
    """Render help text with consistent styling"""
    with st.expander("ℹ️ Help"):
        st.markdown(text)

def render_warning(condition: bool, message: str):
    """Render warning with consistent styling"""
    if condition:
        st.warning(message, icon="⚠️")

def render_error(condition: bool, message: str):
    """Render error with consistent styling"""
    if condition:
        st.error(message, icon="🚨")

def render_success(message: str):
    """Render success message with consistent styling"""
    st.success(message, icon="✅")

def render_info(message: str):
    """Render info message with consistent styling"""
    st.info(message, icon="ℹ️")

def render_page_config(title: str, icon: str):
    """Set consistent page configuration"""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide"
    )