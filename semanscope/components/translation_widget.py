import streamlit as st
from typing import Optional, Tuple

class TranslationWidget:
    def __init__(self):
        self.supported_languages = ["English", "Chinese", "Spanish", "French"]  # Add more as needed
        
    def render(self) -> Tuple[str, str, str]:
        """Render the translation widget and return source text and languages"""
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "From:",
                options=self.supported_languages,
                index=0
            )
            source_text = st.text_area(
                "Enter text to translate:",
                height=150
            )
            
        with col2:
            target_lang = st.selectbox(
                "To:",
                options=[lang for lang in self.supported_languages if lang != source_lang],
                index=0
            )
            
        return source_text, source_lang, target_lang