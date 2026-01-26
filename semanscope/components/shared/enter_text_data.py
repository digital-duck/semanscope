"""
Reusable Enter Text Data Component

This component provides a consistent "Enter Text Data" sidebar control
for all pages (main, Dual View, ECharts-3D) to share.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from config import (
    LANGUAGE_CODE_MAP,
    DEFAULT_LANG_SET, DEFAULT_DATASET,
    sample_chn_input_data,
    sample_enu_input_data,
    SRC_DIR, DATA_PATH,
    get_sorted_language_names,
    get_language_codes_with_prefix,
    get_language_code_from_name,
    get_language_name_from_code
)
from utils.global_settings import get_global_default_dataset


class EnterTextDataWidget:
    """Reusable Enter Text Data sidebar widget supporting up to 6 languages"""

    def __init__(self, key_prefix: str = "", max_languages: int = 3):
        """
        Initialize the widget with a key prefix to avoid conflicts between pages.

        Args:
            key_prefix: Prefix for all Streamlit widget keys (e.g., "main_", "dual_", "echarts_")
            max_languages: Maximum number of languages to support (3 or 6)
        """
        self.key_prefix = key_prefix
        self.max_languages = max_languages
        self.input_dir = DATA_PATH / "input"

        # Default text samples for different languages
        self.default_texts = {
            "chn": sample_chn_input_data,
            "enu": sample_enu_input_data,
            "fra": "rouge\nbleu\nvert\njaune\norange",  # Sample French
            "spa": "rojo\nazul\nverde\namarillo\nnaranja",  # Sample Spanish
            "deu": "rot\nblau\ngrün\ngelb\norange",  # Sample German
            "ara": "أحمر\nأزرق\nأخضر\nأصفر\nبرتقالي",  # Sample Arabic colors
            "heb": "אדום\nכחול\nירוק\nצהוב\nכתום",  # Sample Hebrew colors
            "hin": "लाल\nनीला\nहरा\nपीला\nनारंगी",  # Sample Hindi colors
            "jpn": "赤\n青\n緑\n黄\nオレンジ",  # Sample Japanese colors
            "kor": "빨간색\n파란색\n초록색\n노란색\n주황색",  # Sample Korean colors
            "rus": "красный\nсиний\nзелёный\nжёлтый\nоранжевый",  # Sample Russian colors
            "tha": "แดง\nน้ำเงิน\nเขียว\nเหลือง\nส้ม",  # Sample Thai colors
            "vie": "đỏ\nxanh dương\nxanh lá\nvàng\ncam"  # Sample Vietnamese colors
        }

    def _get_available_inputs(self) -> List[str]:
        """Get list of available input datasets"""
        available_inputs = []
        if self.input_dir.exists():
            input_names = set()
            # Get all language codes with prefix using helper function
            lang_codes = get_language_codes_with_prefix("-")
            for file_path in self.input_dir.glob("*.txt"):
                name_part = file_path.stem
                # Strip all possible language suffixes (only the last one found)
                for lang_code in lang_codes:
                    if name_part.endswith(lang_code):
                        name_part = name_part[:-len(lang_code)]
                        break  # Only remove one suffix to avoid over-stripping
                input_names.add(name_part)
            available_inputs = sorted(list(input_names)) if input_names else ["sample_1"]
        else:
            available_inputs = ["sample_1"]
        return available_inputs

    def _get_language_options_and_defaults(self, input_name_selected: str, visualizer) -> Tuple[List[str], List[int]]:
        """Get language options and default indices based on selected dataset"""
        if input_name_selected and visualizer:
            # Get ALL available languages for this dataset (for dropdown options)
            all_available_lang_codes = visualizer.get_all_available_languages_for_dataset(input_name_selected)
            # Convert codes to display names for dropdowns
            available_lang_names = []
            for code in all_available_lang_codes:
                for name, mapped_code in LANGUAGE_CODE_MAP.items():
                    if mapped_code == code:
                        available_lang_names.append(name)
                        break

            if len(available_lang_names) >= 1:
                # Filter dropdown options to show only available languages
                unique_available = list(dict.fromkeys(available_lang_names))  # Remove duplicates while preserving order
                lang_options = sorted(unique_available)  # Sort alphabetically

                # Use DEFAULT_LANG_SET to determine default selections, prioritizing available languages
                default_indices = []
                for i in range(3):
                    if i < len(DEFAULT_LANG_SET):
                        # Try to use the corresponding language from DEFAULT_LANG_SET
                        preferred_lang_name = get_language_name_from_code(DEFAULT_LANG_SET[i])
                        if preferred_lang_name in lang_options:
                            default_indices.append(lang_options.index(preferred_lang_name))
                        else:
                            # If preferred language not available, cycle through available languages
                            cycle_index = i % len(lang_options)
                            default_indices.append(cycle_index)
                    else:
                        # For slots beyond DEFAULT_LANG_SET length, cycle through available languages
                        cycle_index = i % len(lang_options)
                        default_indices.append(cycle_index)
                return lang_options, default_indices

        # No dataset selected or no languages found, use defaults from DEFAULT_LANG_SET
        lang_options = get_sorted_language_names()
        default_langs = [get_language_name_from_code(code) for code in DEFAULT_LANG_SET[:3]]
        default_indices = []
        for default_lang in default_langs:
            if default_lang in lang_options:
                default_indices.append(lang_options.index(default_lang))
            else:
                default_indices.append(0)
        return lang_options, default_indices

    def render(self, visualizer=None) -> Dict:
        """
        Render the Enter Text Data widget and return the collected data.

        Args:
            visualizer: EmbeddingVisualizer instance for dataset operations

        Returns:
            Dict containing:
            - input_name_selected: Selected dataset name
            - languages: List of (lang_name, lang_code, text_content, is_selected) tuples
            - lang_codes: List of language codes
            - expanded_view: Whether showing 6 languages (for multilingual page)
        """
        with st.expander("✏️ Enter Text Data", expanded=True):
            # Input selection dropdown
            available_inputs = self._get_available_inputs()
            # Use global settings override for default dataset
            default_dataset = get_global_default_dataset()
            input_name_selected = st.selectbox(
                "Select Input Dataset",
                options=available_inputs,
                index=available_inputs.index(default_dataset) if default_dataset in available_inputs else 0,
                key=f"{self.key_prefix}cfg_input_text_selected"
            )

            # Language selection columns
            if self.max_languages == 6:
                # First row: Lang-1, Lang-2, Lang-3
                col_lang1, col_lang2, col_lang3 = st.columns(3)
                # Second row: Lang-4, Lang-5, Lang-6 (initially hidden)
                lang_cols_row2 = None
            else:
                # Standard 3-language layout
                col_lang1, col_lang2, col_lang3 = st.columns(3)

            # Get language options and default indices
            lang_options, default_indices = self._get_language_options_and_defaults(input_name_selected, visualizer)

            # First row language selectors
            with col_lang1:
                lang1 = st.selectbox(
                    "**Lang-1**",
                    options=lang_options,
                    index=default_indices[0],
                    help="Select first language",
                    key=f'{self.key_prefix}lang1'
                )

            with col_lang2:
                lang2 = st.selectbox(
                    "**Lang-2**",
                    options=lang_options,
                    index=default_indices[1],
                    help="Select second language",
                    key=f'{self.key_prefix}lang2'
                )

            with col_lang3:
                lang3 = st.selectbox(
                    "**Lang-3**",
                    options=lang_options,
                    index=default_indices[2],
                    help="Select third language",
                    key=f'{self.key_prefix}lang3'
                )

            # Extended language support for 6-language mode
            lang4, lang5, lang6 = None, None, None
            show_extended = False
            
            if self.max_languages == 6:
                # Check if Load Text was clicked to expand
                show_extended = st.session_state.get(f'{self.key_prefix}show_extended_langs', False)

            # Language code mapping
            lang1_code = get_language_code_from_name(lang1)
            lang2_code = get_language_code_from_name(lang2)
            lang3_code = get_language_code_from_name(lang3)

            # Load and Refresh buttons
            col_load_txt, col_refresh = st.columns([3, 1])
            with col_load_txt:
                btn_load_txt = st.button(
                    "Load Text",
                    type="primary",
                    help="Load input texts",
                    disabled=not input_name_selected,
                    width='stretch',
                    key=f"{self.key_prefix}btn_load_txt"
                )
            with col_refresh:
                btn_refresh = st.button(
                    "🔄",
                    help="Refresh available inputs from input folder",
                    key=f"{self.key_prefix}refresh_inputs",
                    width='stretch'
                )

            # Handle refresh button click
            if btn_refresh:
                st.rerun()

            # Handle load text button
            if btn_load_txt and visualizer and input_name_selected:
                self._handle_load_text(input_name_selected, [lang1_code, lang2_code, lang3_code], visualizer)

            # Text input areas with position-based keys
            col1, col2, col3 = st.columns(3)

            # Position-based configurations to avoid duplicate key errors
            lang_configs = [
                (col1, lang1, lang1_code, "pos1"),
                (col2, lang2, lang2_code, "pos2"),
                (col3, lang3, lang3_code, "pos3")
            ]

            languages_data = []
            for col, lang_name, lang_code, position in lang_configs:
                with col:
                    # Try both old and new key formats for compatibility
                    default_value = (
                        st.session_state.get(f'{self.key_prefix}{lang_code}_text_area_{position}', '') or
                        st.session_state.get(f'{self.key_prefix}{lang_code}_text_area', '') or
                        self.default_texts.get(lang_code, "")
                    )

                    text_content = st.text_area(
                        f"{lang_name} ({lang_code}):",
                        value=default_value,
                        height=200,
                        key=f'{self.key_prefix}{lang_code}_text_area_{position}'
                    )
                    text_content = (text_content or "").strip()

                    # Checkbox for inclusion
                    is_selected = st.checkbox(
                        "Include",
                        value=(len(text_content) > 0),
                        key=f'{self.key_prefix}{lang_code}_include_checkbox_{position}'
                    )

                    # Word count display (only shown after Load Text is clicked)
                    if st.session_state.get(f'{self.key_prefix}text_loaded', False) and text_content:
                        word_count = len([word.strip() for word in text_content.split('\n') if word.strip()])
                        st.caption(f"**Words:** {word_count}")

                    languages_data.append((lang_name, lang_code, text_content, is_selected))

            return {
                'input_name_selected': input_name_selected,
                'languages': languages_data,
                'lang_codes': [lang1_code, lang2_code, lang3_code]
            }

    def _handle_load_text(self, input_name_selected: str, lang_codes: List[str], visualizer):
        """Handle the Load Text button functionality"""
        missing_files = []
        positions = ["pos1", "pos2", "pos3"]

        for i, lang_code in enumerate(lang_codes):
            position = positions[i]
            words, word_color_map = visualizer.load_semantic_data_from_file(input_name_selected, lang_code)
            if words:
                # Filter out lines starting with '#'
                filtered_words = [word for word in words if not word.strip().startswith('#')]
                # Convert back to text format for text area display
                loaded_text = '\n'.join(filtered_words)
                # Store in both old and new key formats for compatibility
                st.session_state[f'{self.key_prefix}{lang_code}_text_area'] = loaded_text
                st.session_state[f'{self.key_prefix}{lang_code}_text_area_{position}'] = loaded_text
                # Store semantic colors as dictionary for later use
                session_key = f"{lang_code}_semantic_colors"
                st.session_state[session_key] = word_color_map
            else:
                missing_files.append(f"'{input_name_selected}-{lang_code}.txt'")

        if missing_files:
            st.warning("No text files found: " + " ; ".join(missing_files))
        else:
            # Set flag to show word counts
            st.session_state[f'{self.key_prefix}text_loaded'] = True
            # Force rerun to update text areas
            st.rerun()