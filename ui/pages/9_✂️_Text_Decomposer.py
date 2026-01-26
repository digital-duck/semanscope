import streamlit as st
import re
from pathlib import Path
from typing import List

from semanscope.config import check_login, SRC_DIR

# Page config
st.set_page_config(
    page_title="Text Decomposer",
    page_icon="✂️",
    layout="wide"
)


class TextDecomposer:
    """Decomposes text into hierarchical semantic components"""

    @staticmethod
    def is_chinese(text: str) -> bool:
        """Check if text contains Chinese characters"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    @staticmethod
    def decompose_chinese_text(text_lines: List[str]) -> str:
        """Decompose Chinese text into hierarchical components"""
        components = []
        all_chars = []

        # Clean and validate lines
        text_lines = [line.strip() for line in text_lines if line.strip()]

        # Level 1: Individual characters
        for line in text_lines:
            for char in line:
                if TextDecomposer.is_chinese(char):
                    all_chars.append(char)

        components.extend(all_chars)
        components.append("")  # Separator

        # Level 2: Generate n-character phrases
        two_char_phrases = set()
        three_char_phrases = set()
        longer_phrases = set()

        for line in text_lines:
            chars = [c for c in line if TextDecomposer.is_chinese(c)]

            # 2-character phrases
            for i in range(len(chars) - 1):
                two_char_phrases.add(chars[i] + chars[i+1])

            # 3-character phrases
            for i in range(len(chars) - 2):
                three_char_phrases.add(chars[i] + chars[i+1] + chars[i+2])

            # Longer phrases (up to line length - 1)
            for length in range(4, len(chars)):
                for i in range(len(chars) - length + 1):
                    phrase = ''.join(chars[i:i+length])
                    longer_phrases.add(phrase)

        # Add phrases in order: 2-char, 3-char, longer
        components.extend(sorted(two_char_phrases))
        if three_char_phrases:
            components.append("")
            components.extend(sorted(three_char_phrases))
        if longer_phrases:
            components.append("")
            components.extend(sorted(longer_phrases))

        # Level 3: Complete lines
        components.append("")
        components.extend(text_lines)

        # Level 4: Full text (all lines concatenated)
        components.append("")
        full_text = ''.join(text_lines)
        components.append(full_text)

        return '\n'.join(components)

    @staticmethod
    def decompose_english_text(text_lines: List[str]) -> str:
        """Decompose English text into hierarchical components"""
        components = []
        all_words = []

        # Clean and validate lines
        text_lines = [line.strip() for line in text_lines if line.strip()]

        # Level 1: Individual words (remove punctuation)
        for line in text_lines:
            words = re.findall(r'\b[a-zA-Z]+\b', line.lower())
            all_words.extend(words)

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in all_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)

        components.extend(unique_words)
        components.append("")  # Separator

        # Level 2: Generate n-word phrases
        two_word_phrases = set()
        three_word_phrases = set()

        for line in text_lines:
            words = re.findall(r'\b[a-zA-Z]+\b', line.lower())

            # 2-word phrases
            for i in range(len(words) - 1):
                two_word_phrases.add(f"{words[i]} {words[i+1]}")

            # 3-word phrases
            for i in range(len(words) - 2):
                three_word_phrases.add(f"{words[i]} {words[i+1]} {words[i+2]}")

        components.extend(sorted(two_word_phrases))
        if three_word_phrases:
            components.append("")
            components.extend(sorted(three_word_phrases))

        # Level 3: Complete lines (preserve original case and punctuation)
        components.append("")
        components.extend(text_lines)

        # Level 4: Full text (all lines joined)
        components.append("")
        full_text = ' '.join(text_lines)
        components.append(full_text)

        return '\n'.join(components)

    @staticmethod
    def decompose_text(text: str, language: str = 'auto') -> str:
        """Main decomposition method"""
        lines = text.strip().split('\n')

        # Determine language
        if language == 'auto':
            use_chinese = TextDecomposer.is_chinese(text)
        elif language == 'chn':
            use_chinese = True
        else:
            use_chinese = False

        if use_chinese:
            return TextDecomposer.decompose_chinese_text(lines)
        else:
            return TextDecomposer.decompose_english_text(lines)


def main():
    # Check login status
    check_login()

    st.subheader("✂️ Text Decomposer - Hierarchical Analysis")
    st.caption("Decompose poems and passages into hierarchical components for semantic geometry exploration")

    # Help section in sidebar
    with st.sidebar:
        with st.expander("ℹ️ How to Use", expanded=False):
            st.markdown("""
            ### Workflow
            1. **Enter Text**: Paste your poem or passage in the left panel
            2. **Select Language**: Auto-detect or manually specify (Chinese/English)
            3. **Decompose**: Click the "✂️ Decompose" button to generate hierarchical components
            4. **Review/Edit**: Check the decomposed text in the right panel (editable)
            5. **Save**: Enter a filename and click "💾 Save"
            6. **Load**: Go to Semanscope, click 🔄 Refresh, then select your file

            ### Output Structure
            **Chinese Text**:
            - Individual characters (床, 前, 明, 月, 光...)
            - 2-character phrases (床前, 明月, 月光...)
            - 3+ character phrases (床前明, 明月光...)
            - Complete lines (床前明月光, 疑是地上霜...)
            - Full text (entire poem concatenated)

            **English Text**:
            - Individual words (shall, compare, thee, summer...)
            - 2-word phrases (shall i, compare thee...)
            - 3-word phrases (shall i compare, compare thee to...)
            - Complete lines (original formatting)
            - Full text (all lines joined)

            ### Tips
            - One line per sentence/verse for best results
            - Empty lines are ignored
            - You can edit the decomposed text before saving
            - Use descriptive filenames (e.g., `poem-author-title`)
            - Files are saved as `<filename>-chn.txt` or `<filename>-enu.txt`

            ### Research Applications
            - **Compositional Semantics**: Study how meaning builds hierarchically
            - **Poetic Structure**: Analyze geometric patterns in verse
            - **Cross-Cultural Comparison**: Compare Chinese vs English semantic geometry
            - **Author Signatures**: Identify unique semantic patterns
            """)

    # Two-column layout
    col_left, col_right = st.columns(2)

    # Initialize session state
    if 'decomposed_text' not in st.session_state:
        st.session_state.decomposed_text = ""
    if 'decomposition_status' not in st.session_state:
        st.session_state.decomposition_status = None

    with col_left:
        st.markdown("### 📝 Input Text")
        st.caption("Enter your poem or passage (one line per verse/sentence)")

        # Show status message if exists
        if st.session_state.decomposition_status:
            st.success(st.session_state.decomposition_status)
            st.session_state.decomposition_status = None

        # Language selection
        language = st.selectbox(
            "Language",
            options=['auto', 'chn', 'enu'],
            format_func=lambda x: {
                'auto': '🔍 Auto-detect',
                'chn': '🇨🇳 Chinese',
                'enu': '🇺🇸 English'
            }[x],
            help="Auto-detect or manually specify language"
        )

        # Input text area
        input_text = st.text_area(
            "Raw Text",
            height=400,
            placeholder="Enter text here...\n\nExample (Chinese):\n床前明月光\n疑是地上霜\n举头望明月\n低头思故乡\n\nExample (English):\nShall I compare thee to a summer's day?\nThou art more lovely and more temperate",
            key="input_text_area"
        )

        # Decompose button
        if st.button("✂️ Decompose", type="primary", width='stretch'):
            if not input_text.strip():
                st.warning("⚠️ Please enter some text to decompose")
            else:
                with st.spinner("Decomposing text..."):
                    try:
                        decomposed = TextDecomposer.decompose_text(input_text, language)
                        st.session_state.decomposed_text = decomposed

                        # Detect language for user feedback
                        detected_lang = 'Chinese' if TextDecomposer.is_chinese(input_text) else 'English'
                        st.session_state.decomposition_status = f"✅ Decomposed as {detected_lang} text"

                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error during decomposition: {str(e)}")

    with col_right:
        st.markdown("### 📊 Decomposed Components")
        st.caption("Hierarchical breakdown: characters/words → phrases → lines → full text")

        # Output text area (editable)
        decomposed_display = st.text_area(
            "Decomposed Text (editable)",
            value=st.session_state.decomposed_text,
            height=500,
            help="You can edit the decomposed text before saving",
            key="decomposed_text_display"
        )

        # Update session state if user edits the text
        if decomposed_display != st.session_state.decomposed_text:
            st.session_state.decomposed_text = decomposed_display

        col_filename, col_save = st.columns([3, 1])

        with col_filename:
            filename = st.text_input(
                "Filename",
                placeholder="poem-li-bai-moonlight",
                help="Enter filename without extension (e.g., poem-li-bai-moonlight)",
                key="save_filename"
            )

        with col_save:
            st.markdown("<br>", unsafe_allow_html=True)  # Alignment spacing
            save_button = st.button(
                "💾 Save",
                type="secondary",
                width='stretch',
                disabled=not bool(filename and st.session_state.decomposed_text)
            )

        if save_button:
            if not filename:
                st.error("❌ Please enter a filename")
            elif not st.session_state.decomposed_text:
                st.error("❌ No decomposed text to save")
            else:
                try:
                    # Detect language for file extension
                    is_chinese = TextDecomposer.is_chinese(st.session_state.decomposed_text)
                    lang_suffix = 'chn' if is_chinese else 'enu'

                    # Sanitize filename
                    safe_filename = re.sub(r'[^\w\-]', '-', filename)
                    full_filename = f"{safe_filename}-{lang_suffix}.txt"

                    # Save to data/input directory
                    input_dir = SRC_DIR / "data" / "input"
                    input_dir.mkdir(parents=True, exist_ok=True)
                    filepath = input_dir / full_filename

                    # Write file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(st.session_state.decomposed_text)

                    st.success(f"✅ Saved to: `{filepath.relative_to(SRC_DIR.parent)}`")
                    st.info("💡 Click the 🔄 Refresh button in Semanscope to load this file")

                except Exception as e:
                    st.error(f"❌ Error saving file: {str(e)}")


if __name__ == "__main__":
    main()
