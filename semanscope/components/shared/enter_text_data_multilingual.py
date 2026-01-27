"""
Enhanced Enter Text Data Component Supporting 9 Languages

This component extends the original EnterTextDataWidget to support up to 9 languages
for cross-writing-system analysis and multilingual semantic comparison.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from semanscope.config import (
    LANGUAGE_CODE_MAP,
    DEFAULT_LANG_SET,DEFAULT_DATASET,
    sample_chn_input_data,
    sample_enu_input_data,
    SRC_DIR, DATA_PATH,
    get_sorted_language_names,
    get_language_codes_with_prefix,
    get_language_code_from_name,
    get_language_name_from_code
)
from semanscope.utils.global_settings import get_global_default_dataset


class MultilingualEnterTextDataWidget:
    """Enhanced Enter Text Data widget supporting up to 9 languages for cross-script analysis"""

    def __init__(self, key_prefix: str = "multilingual_"):
        """
        Initialize the multilingual widget.

        Args:
            key_prefix: Prefix for all Streamlit widget keys to avoid conflicts
        """
        self.key_prefix = key_prefix
        self.input_dir = DATA_PATH / "input"

        # Default text samples for different languages (expanded for writing systems research)
        self.default_texts = {
            "chn": sample_chn_input_data,
            "enu": sample_enu_input_data,
            "fra": "rouge\nbleu\nvert\njaune\norange",  # French
            "spa": "rojo\nazul\nverde\namarillo\nnaranja",  # Spanish
            "deu": "rot\nblau\ngrün\ngelb\norange",  # German
            "ara": "أحمر\nأزرق\nأخضر\nأصفر\nبرتقالي",  # Arabic
            "heb": "אדום\nכחול\nירוק\nצהוב\nכתום",  # Hebrew
            "hin": "लाल\nनीला\nहरा\nपीला\nनारंगी",  # Hindi
            "jpn": "赤\n青\n緑\n黄\nオレンジ",  # Japanese
            "kor": "빨간색\n파란색\n초록색\n노란색\n주황색",  # Korean
            "rus": "красный\nсиний\nзелёный\nжёлтый\nоранжевый",  # Russian
            "tha": "แดง\nน้ำเงิน\nเขียว\nเหลือง\nส้ม",  # Thai
            "grk": "κόκκινο\nμπλε\nπράσινο\nκίτρινο\nπορτοκαλί",  # Greek
            "vie": "đỏ\nxanh dương\nxanh lá\nvàng\ncam"  # Vietnamese
        }

    def _get_available_inputs(self) -> List[str]:
        """Get list of available input datasets"""
        available_inputs = []
        if self.input_dir.exists():
            input_names = set()
            lang_codes = get_language_codes_with_prefix("-")
            for file_path in self.input_dir.glob("*.txt"):
                name_part = file_path.stem
                for lang_code in lang_codes:
                    if name_part.endswith(lang_code):
                        name_part = name_part[:-len(lang_code)]
                        break
                input_names.add(name_part)
            available_inputs = sorted(list(input_names)) if input_names else ["sample_1"]
        else:
            available_inputs = ["sample_1"]
        return available_inputs

    def _get_language_options_and_defaults(self, input_name_selected: str, visualizer) -> Tuple[List[str], List[int]]:
        """Get language options and default indices based on selected dataset (with smart duplication)"""
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
                for i in range(9):
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

        # No dataset selected or no languages found, use defaults
        lang_options = get_sorted_language_names()
        
        # Enhanced default language selection for cross-script research using DEFAULT_LANG_SET
        default_langs = [get_language_name_from_code(code) for code in DEFAULT_LANG_SET]
        
        # Ensure defaults exist in options, fallback if needed
        default_indices = []
        for default_lang in default_langs:
            if default_lang in lang_options:
                default_indices.append(lang_options.index(default_lang))
            else:
                default_indices.append(0)
        
        return lang_options, default_indices

    def render(self, visualizer=None) -> Dict:
        """
        Render the multilingual Enter Text Data widget.

        Args:
            visualizer: EmbeddingVisualizer instance for dataset operations

        Returns:
            Dict containing:
            - input_name_selected: Selected dataset name
            - languages: List of (lang_name, lang_code, text_content, is_selected) tuples
            - lang_codes: List of language codes
            - expanded_view: Whether showing 6 languages
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
                key=f"{self.key_prefix}cfg_input_text_selected",
                help="Choose a dataset for multilingual analysis"
            )

            # Create dataset-safe key suffix to reset language selections when dataset changes
            dataset_safe = input_name_selected.replace(" ", "_").replace("-", "_") if input_name_selected else "none"

            # Get language options and default indices (with smart duplication for datasets)
            lang_options, safe_defaults = self._get_language_options_and_defaults(input_name_selected, visualizer)

            # First row: Lang-1, Lang-2, Lang-3
            st.markdown("**Primary Languages:**")
            col_lang1, col_lang2, col_lang3 = st.columns(3)

            with col_lang1:
                lang1 = st.selectbox(
                    "**Lang-1**",
                    options=lang_options,
                    index=safe_defaults[0],
                    help="Select first language for comparison",
                    key=f'{self.key_prefix}lang1_{dataset_safe}'
                )

            with col_lang2:
                lang2 = st.selectbox(
                    "**Lang-2**",
                    options=lang_options,
                    index=safe_defaults[1],
                    help="Select second language for comparison",
                    key=f'{self.key_prefix}lang2_{dataset_safe}'
                )

            with col_lang3:
                lang3 = st.selectbox(
                    "**Lang-3**",
                    options=lang_options,
                    index=safe_defaults[2],
                    help="Select third language for comparison",
                    key=f'{self.key_prefix}lang3_{dataset_safe}'
                )

            # Language codes for primary languages
            lang1_code = get_language_code_from_name(lang1)
            lang2_code = get_language_code_from_name(lang2)
            lang3_code = get_language_code_from_name(lang3)

            # Extended languages (always shown for 9-language mode)
            lang4, lang5, lang6, lang7, lang8, lang9 = None, None, None, None, None, None
            lang4_code, lang5_code, lang6_code, lang7_code, lang8_code, lang9_code = None, None, None, None, None, None

            # Always show Lang-4 to Lang-6 dropdowns
            # st.markdown("---")
            col_lang4, col_lang5, col_lang6 = st.columns(3)

            with col_lang4:
                lang4 = st.selectbox(
                    "**Lang-4**",
                    options=lang_options,
                    index=safe_defaults[3],
                    help="Select fourth language",
                    key=f'{self.key_prefix}lang4_{dataset_safe}'
                )

            with col_lang5:
                lang5 = st.selectbox(
                    "**Lang-5**",
                    options=lang_options,
                    index=safe_defaults[4],
                    help="Select fifth language",
                    key=f'{self.key_prefix}lang5_{dataset_safe}'
                )

            with col_lang6:
                lang6 = st.selectbox(
                    "**Lang-6**",
                    options=lang_options,
                    index=safe_defaults[5],
                    help="Select sixth language",
                    key=f'{self.key_prefix}lang6_{dataset_safe}'
                )

            # Always show Lang-7 to Lang-9 dropdowns
            col_lang7, col_lang8, col_lang9 = st.columns(3)

            with col_lang7:
                lang7 = st.selectbox(
                    "**Lang-7**",
                    options=lang_options,
                    index=safe_defaults[6],
                    help="Select seventh language",
                    key=f'{self.key_prefix}lang7_{dataset_safe}'
                )

            with col_lang8:
                lang8 = st.selectbox(
                    "**Lang-8**",
                    options=lang_options,
                    index=safe_defaults[7],
                    help="Select eighth language",
                    key=f'{self.key_prefix}lang8_{dataset_safe}'
                )

            with col_lang9:
                lang9 = st.selectbox(
                    "**Lang-9**",
                    options=lang_options,
                    index=safe_defaults[8],
                    help="Select ninth language",
                    key=f'{self.key_prefix}lang9_{dataset_safe}'
                )

            # Extended language codes
            lang4_code = get_language_code_from_name(lang4)
            lang5_code = get_language_code_from_name(lang5)
            lang6_code = get_language_code_from_name(lang6)
            lang7_code = get_language_code_from_name(lang7)
            lang8_code = get_language_code_from_name(lang8)
            lang9_code = get_language_code_from_name(lang9)

            # Collect all language codes and names (always include all 9 languages)
            all_lang_codes = [lang1_code, lang2_code, lang3_code, lang4_code, lang5_code, lang6_code, lang7_code, lang8_code, lang9_code]
            all_lang_names = [lang1, lang2, lang3, lang4, lang5, lang6, lang7, lang8, lang9]

            # Load text button (placed after all language selectors)
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                btn_load_txt = st.button(
                    "Load Text",
                    help="Load text data and expand to 9-language view for cross-script analysis",
                    key=f"{self.key_prefix}btn_load_txt",
                    width='stretch',
                    type="primary"
                )
            with col_btn2:
                btn_refresh = st.button(
                    "🔄",
                    help="Refresh Input Dataset Dropdown",
                    key=f"{self.key_prefix}btn_refresh"
                )

            # Handle refresh button
            if btn_refresh:
                st.session_state[f'{self.key_prefix}show_extended_langs'] = False
                st.rerun()

            # Handle load text button
            if btn_load_txt and visualizer and input_name_selected:
                st.session_state[f'{self.key_prefix}show_extended_langs'] = True
                self._handle_load_text(input_name_selected, all_lang_codes, visualizer, dataset_safe)

            # Text input areas
            # st.markdown("---")
            languages_data = []
            
            # Primary language text areas (always shown)
            st.markdown("**Text Input Areas:**")
            col1, col2, col3 = st.columns(3)
            primary_configs = [
                (col1, lang1, lang1_code, "pos1"),
                (col2, lang2, lang2_code, "pos2"),
                (col3, lang3, lang3_code, "pos3")
            ]
            
            for col, lang_name, lang_code, position in primary_configs:
                with col:
                    default_value = (
                        st.session_state.get(f'{self.key_prefix}{lang_code}_text_area_{position}_{dataset_safe}', '') or
                        self.default_texts.get(lang_code, "")
                    )

                    text_content = st.text_area(
                        f"{lang_name} ({lang_code}):",
                        value=default_value,
                        height=200,
                        key=f'{self.key_prefix}{lang_code}_text_area_{position}_{dataset_safe}'
                    )
                    text_content = (text_content or "").strip()

                    is_selected = st.checkbox(
                        "Include",
                        value=(len(text_content) > 0),
                        key=f'{self.key_prefix}{lang_code}_include_checkbox_{position}_{dataset_safe}'
                    )

                    # Word count display (only shown after Load Text is clicked)
                    if st.session_state.get(f'{self.key_prefix}text_loaded_{dataset_safe}', False) and text_content:
                        word_count = len([word.strip() for word in text_content.split('\n') if word.strip()])
                        st.caption(f"**Words:** {word_count}")

                    languages_data.append((lang_name, lang_code, text_content, is_selected))

            # Extended language text areas (always shown in 9-language mode)
            col4, col5, col6 = st.columns(3)
            extended_configs_1 = [
                (col4, lang4, lang4_code, "pos4"),
                (col5, lang5, lang5_code, "pos5"),
                (col6, lang6, lang6_code, "pos6")
            ]

            for col, lang_name, lang_code, position in extended_configs_1:
                with col:
                    default_value = (
                        st.session_state.get(f'{self.key_prefix}{lang_code}_text_area_{position}_{dataset_safe}', '') or
                        self.default_texts.get(lang_code, "")
                    )

                    text_content = st.text_area(
                        f"{lang_name} ({lang_code}):",
                        value=default_value,
                        height=200,
                        key=f'{self.key_prefix}{lang_code}_text_area_{position}_{dataset_safe}'
                    )
                    text_content = (text_content or "").strip()

                    is_selected = st.checkbox(
                        "Include",
                        value=(len(text_content) > 0),
                        key=f'{self.key_prefix}{lang_code}_include_checkbox_{position}_{dataset_safe}'
                    )

                    # Word count display (only shown after Load Text is clicked)
                    if st.session_state.get(f'{self.key_prefix}text_loaded_{dataset_safe}', False) and text_content:
                        word_count = len([word.strip() for word in text_content.split('\n') if word.strip()])
                        st.caption(f"**Words:** {word_count}")

                    languages_data.append((lang_name, lang_code, text_content, is_selected))

            # Additional text areas for Lang-7, Lang-8, Lang-9
            col7, col8, col9 = st.columns(3)
            extended_configs_2 = [
                (col7, lang7, lang7_code, "pos7"),
                (col8, lang8, lang8_code, "pos8"),
                (col9, lang9, lang9_code, "pos9")
            ]

            for col, lang_name, lang_code, position in extended_configs_2:
                with col:
                    default_value = (
                        st.session_state.get(f'{self.key_prefix}{lang_code}_text_area_{position}_{dataset_safe}', '') or
                        self.default_texts.get(lang_code, "")
                    )

                    text_content = st.text_area(
                        f"{lang_name} ({lang_code}):",
                        value=default_value,
                        height=200,
                        key=f'{self.key_prefix}{lang_code}_text_area_{position}_{dataset_safe}'
                    )
                    text_content = (text_content or "").strip()

                    is_selected = st.checkbox(
                        "Include",
                        value=(len(text_content) > 0),
                        key=f'{self.key_prefix}{lang_code}_include_checkbox_{position}_{dataset_safe}'
                    )

                    # Word count display (only shown after Load Text is clicked)
                    if st.session_state.get(f'{self.key_prefix}text_loaded_{dataset_safe}', False) and text_content:
                        word_count = len([word.strip() for word in text_content.split('\n') if word.strip()])
                        st.caption(f"**Words:** {word_count}")

                    languages_data.append((lang_name, lang_code, text_content, is_selected))

            return {
                'input_name_selected': input_name_selected,
                'languages': languages_data,
                'lang_codes': all_lang_codes,
                'expanded_view': True  # Always in 9-language mode
            }

    def _handle_load_text(self, input_name_selected: str, lang_codes: List[str], visualizer, dataset_safe: str):
        """Handle the Load Text button functionality for all languages"""
        missing_files = []
        positions = ["pos1", "pos2", "pos3", "pos4", "pos5", "pos6", "pos7", "pos8", "pos9"]

        for i, lang_code in enumerate(lang_codes):
            if i < len(positions):
                position = positions[i]
                words, word_color_map = visualizer.load_semantic_data_from_file(input_name_selected, lang_code)
                if words:
                    # Filter out lines starting with '#'
                    filtered_words = [word for word in words if not word.strip().startswith('#')]
                    # Convert back to text format for text area display
                    loaded_text = '\n'.join(filtered_words)
                    # Store in session state with dataset-specific key
                    st.session_state[f'{self.key_prefix}{lang_code}_text_area_{position}_{dataset_safe}'] = loaded_text
                    # Store semantic colors as dictionary for later use
                    session_key = f"{lang_code}_semantic_colors"
                    st.session_state[session_key] = word_color_map
                else:
                    missing_files.append(f"'{input_name_selected}-{lang_code}.txt'")

        if missing_files:
            st.warning("No text files found: " + " ; ".join(missing_files))
        else:
            # Set flag to show word counts (dataset-specific)
            st.session_state[f'{self.key_prefix}text_loaded_{dataset_safe}'] = True
            # Force rerun to update text areas with extended view
            st.rerun()