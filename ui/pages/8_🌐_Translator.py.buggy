import streamlit as st
import os
import deepl
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from typing import Tuple, Optional, Dict

from semanscope.config import check_login, SRC_DIR, DATA_PATH

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Translator",
    page_icon="🌐",
    layout="wide"
)

class DeepLTranslator:
    """DeepL Translation service for Semanscope"""
    
    def __init__(self, auth_key: str):
        try:
            self.translator = deepl.Translator(auth_key)
        except Exception as e:
            raise Exception(f"Failed to initialize DeepL: {e}")

    def get_source_languages(self) -> Dict[str, str]:
        """Get available source languages"""
        try:
            return {lang.code: lang.name for lang in self.translator.get_source_languages()}
        except Exception as e:
            raise Exception(f"Error fetching source languages: {e}")

    def get_target_languages(self) -> Dict[str, str]:
        """Get available target languages"""
        try:
            return {lang.code: lang.name for lang in self.translator.get_target_languages()}
        except Exception as e:
            raise Exception(f"Error fetching target languages: {e}")

    def translate(self, text: str, target_lang: str, source_lang: str = "auto", line_by_line: bool = True) -> Tuple[str, str]:
        """Translate text and return (translated_text, detected_language)

        Args:
            text: Text to translate
            target_lang: Target language code
            source_lang: Source language code (or "auto" for auto-detection)
            line_by_line: If True, translate each line independently. If False, translate as block.
        """
        try:
            # Detect language first
            if source_lang == "auto":
                # Use first line or whole text for detection
                detection_text = text.strip().split('\n')[0] if text.strip() else text
                temp_result = self.translator.translate_text(detection_text, target_lang="EN-US")
                detected_lang = temp_result.detected_source_lang
            else:
                detected_lang = source_lang

            # Block translation mode (original behavior)
            if not line_by_line:
                print(f"[DEBUG] Block translation mode")
                result = self.translator.translate_text(
                    text,
                    target_lang=target_lang,
                    source_lang=detected_lang if detected_lang != "auto" else None
                )
                print(f"[DEBUG] Block translation result: '{result.text}'")
                return result.text, detected_lang

            # Line-by-line translation mode
            print(f"[DEBUG] Line-by-line translation mode")
            print(f"[DEBUG] Raw text repr: {repr(text)}")
            lines = text.strip().split('\n')
            print(f"[DEBUG] Total lines to translate: {len(lines)}")
            print(f"[DEBUG] Lines: {lines}")

            # Filter out empty lines but preserve their positions
            non_empty_lines = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]
            print(f"[DEBUG] Non-empty lines: {len(non_empty_lines)}")
            print(f"[DEBUG] Non-empty lines content: {non_empty_lines}")

            if not non_empty_lines:
                return "", detected_lang

            # Translate each line separately for better quality
            translated_lines = [''] * len(lines)

            for idx, line in non_empty_lines:
                print(f"[DEBUG] Translating line {idx}: '{line}'")

                result = self.translator.translate_text(
                    line,
                    target_lang=target_lang,
                    source_lang=detected_lang if detected_lang != "auto" else None
                )

                print(f"[DEBUG] Translation result: '{result.text}'")
                translated_lines[idx] = result.text

            # Rejoin with newlines
            translated_text = '\n'.join(translated_lines)
            print(f"[DEBUG] Final joined translation:\n{translated_text}")

            return translated_text, detected_lang

        except Exception as e:
            raise Exception(f"Translation failed: {e}")

def main():
    # Check login status
    check_login()
    
    st.markdown("### 🌐 Professional Translation by [DeepL](https://www.deepl.com/en/translator)")
    
    # Check for DeepL API key
    DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY')
    if not DEEPL_AUTH_KEY:
        st.error("🔑 DeepL API key not found!")
        st.info("Please add your DeepL API key to your .env file:")
        st.code("DEEPL_AUTH_KEY=your_api_key_here", language="bash")
        st.markdown("Get your API key at: https://www.deepl.com/pro-api")
        return
    
    # Initialize translator
    try:
        translator = DeepLTranslator(DEEPL_AUTH_KEY)
        source_languages = translator.get_source_languages()
        target_languages = translator.get_target_languages()
    except Exception as e:
        st.error(f"❌ Error initializing DeepL translator: {e}")
        return
    
    # Language selection
    col_source, col_target = st.columns(2)
    
    with col_source:
        # st.subheader("Source Language")
        
        # Source language selection
        source_langs = ['auto'] + list(source_languages.keys())
        source_display_names = ['Auto-detect'] + [source_languages[lang] for lang in source_languages.keys()]

        # Default to Chinese if available
        default_source_idx = 0
        if 'ZH' in source_languages:
            default_source_idx = source_langs.index('ZH')
        
        selected_source_idx = st.selectbox(
            "Select source language:",
            range(len(source_langs)),
            index=default_source_idx,
            format_func=lambda x: source_display_names[x],
            key="source_lang_select"
        )
        source_lang = source_langs[selected_source_idx]
        
    with col_target:
        # st.subheader("Target Language")
        
        # Target language selection
        target_langs = list(target_languages.keys())
        target_display_names = [target_languages[lang] for lang in target_langs]

        # Default to English (American) if available
        default_target_idx = 0
        if 'EN-US' in target_languages:
            default_target_idx = target_langs.index('EN-US')
        elif 'EN' in target_languages:
            default_target_idx = target_langs.index('EN')
            
        selected_target_idx = st.selectbox(
            "Select target language:",
            range(len(target_langs)),
            index=default_target_idx,
            format_func=lambda x: target_display_names[x],
            key="target_lang_select"
        )
        target_lang = target_langs[selected_target_idx]

    # Display current translation direction
    source_name = source_languages.get(source_lang, source_lang) if source_lang != "auto" else "Auto-detect"
    target_name = target_languages.get(target_lang, target_lang)
    st.caption(f"🔍 Translation direction: {source_name} ({source_lang}) → {target_name} ({target_lang})")

    # Clear cached translations if language selection changed
    if 'last_source_lang' not in st.session_state:
        st.session_state.last_source_lang = source_lang
    if 'last_target_lang' not in st.session_state:
        st.session_state.last_target_lang = target_lang

    if (st.session_state.last_source_lang != source_lang or
        st.session_state.last_target_lang != target_lang):
        # Language changed - clear old translations
        if 'translated_text' in st.session_state:
            del st.session_state['translated_text']
        if 'detected_language' in st.session_state:
            del st.session_state['detected_language']
        st.session_state.last_source_lang = source_lang
        st.session_state.last_target_lang = target_lang

    # st.divider()

    # Translation interface
    col1, col2 = st.columns(2)
    
    with col1:
        # st.subheader("📝 Source Text")

        # Initialize with default text if not already set
        if 'source_text_input' not in st.session_state:
            st.session_state.source_text_input = "丈夫\n妻子"

        source_text = st.text_area(
            "📝 Enter text to translate:",
            height=500,
            placeholder="Type or paste your text here...",
            key="source_text_input"
        )


        # Action buttons - Translate first, then Clear and Example
        col_checkbox, col_translate, col_clear, col_example = st.columns([2, 3, 2, 2])
        with col_checkbox:
            line_by_line = st.checkbox(
                "Block/Passage mode",
                value=False,
                help="When enabled (recommended), translates the entire text as a passage for better context. When disabled, translates each line independently."
            )

        with col_translate:
            translate_clicked = st.button("🔄 Translate", type="primary", width='content', help="Translate text")

        with col_clear:
            if st.button("🗑️ Clear", help="Clear source text"):
                st.session_state.source_text_input = ""
                st.rerun()

        with col_example:
            if st.button("📋 Example", help="Load example text"):
                example_text = "The geometry of meaning reveals how concepts are structured in semantic space."
                st.session_state.source_text_input = example_text
                st.rerun()

    with col2:
        # st.subheader("🔄 Translation")

        # Handle translation
        if translate_clicked:
            if not source_text.strip():
                st.warning("Please enter text to translate.")
            else:
                try:
                    with st.spinner("Translating..."):
                        translated_text, detected_lang = translator.translate(
                            source_text,
                            target_lang,
                            source_lang,
                            line_by_line=not line_by_line  # Inverted: unchecked = line-by-line, checked = block
                        )

                        # DEBUG: Display API response
                        mode = "Block/Passage" if line_by_line else "Line-by-line"
                        st.info(f"🔍 DEBUG - API Response:\n- **Mode**: {mode}\n- Translated text: `{translated_text}`\n- Type: `{type(translated_text)}`\n- Length: `{len(translated_text) if translated_text else 0}`\n- Detected language: `{detected_lang}`")

                        # Store in session state
                        st.session_state.translated_text = translated_text
                        st.session_state.detected_language = detected_lang

                        # Show detection info if auto-detect was used
                        if source_lang == "auto":
                            detected_name = source_languages.get(detected_lang, detected_lang)
                            st.success(f"✅ Detected source language: **{detected_name}**")

                        # Force rerun to update display
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Translation error: {e}")
        
        # Display translation
        if 'translated_text' in st.session_state:
            translated_text = st.text_area(
                "Translation result (editable):",
                value=st.session_state.translated_text,
                height=500,
                key="translated_text_display"
            )
            
            # Action buttons for translation
            _, col_copy, col_save, col_use = st.columns(4)
            
            with col_copy:
                if st.button("📋 Copy", help="Copy to clipboard"):
                    st.info("💡 Use Ctrl+C to copy the text above")
            
            with col_save:
                if st.button("💾 Save", help="Save translation pair"):
                    # Create translations directory if it doesn't exist
                    translations_dir = DATA_PATH / "translations"
                    translations_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save translation pair
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    filename = translations_dir / f"translation_{timestamp}.txt"
                    
                    content = f"Source ({source_languages.get(st.session_state.get('detected_language', source_lang), source_lang)}):\n{source_text}\n\nTarget ({target_languages[target_lang]}):\n{translated_text}\n"
                    
                    try:
                        filename.write_text(content, encoding='utf-8')
                        st.success(f"✅ Saved: {filename.name}")
                    except Exception as e:
                        st.error(f"❌ Save error: {e}")
            
            with col_use:
                if st.button("📥 Use for Research", help="Add to semantic datasets"):
                    st.info("💡 Copy the translation and use it in the Semanscope!")
        else:
            st.text_area(
                "Translation will appear here:",
                value="",
                height=500,
                disabled=True,
                placeholder="Click 'Translate' to see results..."
            )
    
    # st.divider()
    
    # Usage tips
    with st.sidebar:
        with st.expander("💡 Usage Tips", expanded=False):
            st.markdown("""
            ### 🎯 For Semantic Research:
            - **Translate concept lists** to explore cross-lingual patterns
            - **Use consistent terminology** for better semantic alignment
            - **Compare translations** to understand cultural differences
            
            ### 🔧 DeepL Features:
            - **High-quality translations** for research purposes
            - **Automatic language detection** for unknown text
            - **Supports 30+ languages** including Chinese, Japanese, Korean
            
            ### 📚 Research Applications:
            - Translate semantic categories (colors, emotions, animals)
            - Create multilingual datasets for embedding analysis  
            - Verify cross-lingual concept alignment
            """)
    
    # Statistics
    if 'translated_text' in st.session_state and 'detected_language' in st.session_state:
        st.sidebar.subheader("📊 Translation Info")
        st.sidebar.info(f"""
        **Source**: {source_languages.get(st.session_state.detected_language, 'Unknown')}  
        **Target**: {target_languages[target_lang]}  
        **Characters**: {len(source_text)} → {len(st.session_state.translated_text)}
        """)

if __name__ == "__main__":
    main()