import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import os
import re
from pathlib import Path
import unicodedata

from semanscope.config import (
    MODEL_INFO, METHOD_INFO, DEFAULT_MODEL, DEFAULT_METHOD, DEFAULT_DATASET,
    COLOR_MAP, LANGUAGE_CODE_MAP, DEFAULT_LANG_SET,
    sample_chn_input_data, sample_enu_input_data,
    SRC_DIR, DATA_PATH, get_domain_color, get_all_domain_colors,
    get_sorted_language_names, get_language_codes_with_prefix,
    get_language_code_from_name, get_language_name_from_code,
    ENABLE_CHINESE_TEXT_PREPROCESSING, SHOW_PREPROCESSING_WARNINGS
)
from semanscope.utils.global_settings import get_global_default_dataset
from semanscope.utils.text_preprocessing import preprocess_texts_for_embedding
from semanscope.models.model_manager import get_model, get_active_models, get_model_with_strategy
from semanscope.utils.error_handling import handle_errors
from semanscope.components.shared.publication_settings import PublicationSettingsWidget
from semanscope.utils.cache_manager import (
    get_cached_embeddings, save_embeddings_to_cache,
    get_cached_dimension_reduction, save_dimension_reduction_to_cache
)
from semanscope.utils.title_filename_helper import create_title_and_filename

from semanscope.components.plotting import PlotManager

def get_active_methods():
    """Get only active methods for UI display"""
    active_methods = {}
    for name, info in METHOD_INFO.items():
        if info.get("is_active", True):  # Default to True for backward compatibility
            active_methods[name] = info
    return active_methods

def rearrange_by_ollama(models):
    l1 = []
    l2 = []
    for i in models:
        if "(Ollama)" in i:
            l2.append(i)
        else:
            l1.append(i)
    return l1 + l2

class EmbeddingVisualizer:
    def __init__(self):
        # Use only active models and methods
        active_models = get_active_models()
        self.model_names = rearrange_by_ollama(sorted(list(active_models.keys())))

        self.active_methods = get_active_methods()
        self.method_names = sorted(list(self.active_methods.keys()))
        self.input_dir = DATA_PATH / "input"
        self.images_dir = DATA_PATH / "images"

        # Initialize session state for plot rotation
        if 'plot_rotation' not in st.session_state:
            st.session_state.plot_rotation = 0
        if 'current_figure' not in st.session_state:
            st.session_state.current_figure = None

    def render_sidebar(self) -> Tuple[str, str, str, bool, Optional[int], bool]:
        """Render sidebar controls and return settings"""
        with st.sidebar:
            st.subheader("⚙️ Settings")

            # V2O Optimization Strategy Selector (Global Setting)
            from components.optimization_strategy import render_optimization_strategy_selector
            render_optimization_strategy_selector(location="sidebar", expanded=False)

            with st.expander("🎨 Visualization Settings", expanded=False):

                # Model selection
                model_name = st.radio(
                    "Choose Embedding Model (Ollama is slower)",
                    options=self.model_names,
                    index=self.model_names.index(DEFAULT_MODEL),
                    help="Select a multilingual embedding model",
                    key="cfg_embed_model_name"
                )

                # Warning for E5-Base-v2 with Chinese text
                if model_name == "E5-Base-v2":
                    st.warning("⚠️ E5-Base-v2 may produce NaN errors with Chinese text. Consider using Sentence-BERT Multilingual or BGE-M3 for Chinese-English datasets.")
                if model_name == DEFAULT_MODEL:
                    info_msg = f"**{model_name}** (default): {MODEL_INFO[model_name]['help']}"
                else:
                    info_msg = f"**{model_name}**: {MODEL_INFO[model_name]['help']}"
                st.info(info_msg)

                # Method selection
                method_name = st.radio(
                    "Choose Dimensionality Reduction Method",
                    options=self.method_names,
                    index=self.method_names.index(DEFAULT_METHOD),
                    help="Select a dimensionality reduction method",
                    key="cfg_dim_reduc_method_name"
                )
                if method_name == DEFAULT_METHOD:
                    info_msg = f"**{method_name}** (default): {self.active_methods[method_name]['help']}"
                else:
                    info_msg = f"**{method_name}**: {self.active_methods[method_name]['help']}"
                st.info(info_msg)

                # Dimensions
                dimensions = st.radio(
                    "Choose Dimensions",
                    options=["2D", "3D"],
                    index=0,
                    help="Select 2D or 3D visualization",
                    key="cfg_vis_dimensions"
                )

                # Clustering
                do_clustering = st.checkbox(
                    "Enable Clustering?", 
                    value=False,
                    help="Toggle clustering of points in the visualization",
                    key="cfg_enable_clustering"
                )
                n_clusters = None
                if do_clustering:
                    n_clusters = st.slider("Number of Clusters", min_value=3, max_value=10, value=5)

                # Debug mode for character encoding
                debug_flag = st.checkbox(
                    "🔍 Debug Character Encoding",
                    value=False,
                    help="Show detailed debugging information for character encoding failures",
                    key="cfg_debug_character_encoding"
                )

            # Publication Settings (using shared component)
            publication_settings = PublicationSettingsWidget.render_publication_settings("main_viz")

            # Store settings in session state for PlotManager to access
            st.session_state.publication_settings = publication_settings

        return model_name, method_name, dimensions, do_clustering, n_clusters, debug_flag

    @handle_errors
    def process_text(self, text: str, dedup: bool = True) -> List[str]:
        """Process input text into list of words
        
        Args:
            text: Input text string
            dedup: Whether to remove duplicates
            
        Returns:
            List of processed words, ignoring comment lines starting with #
        """
        # Split into lines and filter out comment lines starting with #
        lines = text.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('#')]
        
        # Join back and process as before
        filtered_text = '\n'.join(filtered_lines)
        filtered_text = filtered_text.replace("\n", " ").replace(",", " ").replace(";", " ").replace("，", " ").replace("；", " ")
        results = [w.strip('"') for w in filtered_text.split() if w.strip('"')]
        return list(set(results)) if dedup else results

    def get_available_inputs(self) -> List[str]:
        """Get list of available input names from data/input directory"""
        if not self.input_dir.exists():
            return ["sample_1"]

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

        return sorted(list(input_names)) if input_names else ["sample_1"]

    def get_all_available_languages_for_dataset(self, dataset_name: str):
        """Get ALL available language codes for a specific dataset (for dropdown options)"""
        if not dataset_name:
            return []

        available_langs = []
        # Check which language files exist for this dataset
        for file_path in self.input_dir.glob(f"{dataset_name}-*.txt"):
            # Extract language code from filename
            lang_code = file_path.stem.replace(f"{dataset_name}-", "")
            if lang_code in LANGUAGE_CODE_MAP.values():  # Valid language code
                available_langs.append(lang_code)

        # Sort to ensure consistent ordering
        available_langs.sort()
        return available_langs

    def get_available_languages_for_dataset(self, dataset_name: str):
        """Get available language codes for a specific dataset with smart duplication for 3 slots"""
        if not dataset_name:
            # Return default fallback languages from DEFAULT_LANG_SET (first 3)
            return [get_language_name_from_code(code) for code in DEFAULT_LANG_SET[:3]]

        # Use the helper function to get all available languages
        available_langs = self.get_all_available_languages_for_dataset(dataset_name)

        if not available_langs:
            # No files found, return defaults from DEFAULT_LANG_SET (first 3)
            return [get_language_name_from_code(code) for code in DEFAULT_LANG_SET[:3]]

        # Smart duplication logic for exactly 3 slots
        if len(available_langs) == 1:
            # Only 1 language: duplicate it for all 3 slots
            return [available_langs[0], available_langs[0], available_langs[0]]
        elif len(available_langs) == 2:
            # 2 languages: use both, duplicate the second for Lang-3
            return [available_langs[0], available_langs[1], available_langs[1]]
        else:
            # 3 or more languages: take first 3
            return available_langs[:3]

    def load_text_from_file(self, input_name: str, language: str) -> str:
        """Load text content from file"""
        file_path = self.input_dir / f"{input_name}-{language}.txt"
        if file_path.exists():
            try:
                return file_path.read_text(encoding='utf-8').strip()
            except Exception as e:
                st.error(f"Error reading file {file_path}: {e}")
        return ""

    def load_semantic_data_from_file(self, input_name: str, language: str) -> tuple[list, dict]:
        """
        Load data from file and parse as either plain text or CSV with semantic domains
        Returns: (words_list, word_color_map_dict)
        """
        # Try .txt first, then .csv
        txt_path = self.input_dir / f"{input_name}-{language}.txt"
        csv_path = self.input_dir / f"{input_name}-{language}.csv"

        file_path = None
        if txt_path.exists():
            file_path = txt_path
        elif csv_path.exists():
            file_path = csv_path
        else:
            return [], {}

        try:
            content = file_path.read_text(encoding='utf-8').strip()
            if not content:
                return [], {}

            lines = content.split('\n')
            first_line = lines[0].strip()

            # Check if this is CSV format with domain classification
            if ',' in first_line and ('word' in first_line.lower()) and ('domain' in first_line.lower()):
                return self._parse_csv_format(content, language)
            else:
                # Plain text format (either single words per line or free text)
                return self._parse_plain_text_format(content)

        except Exception as e:
            st.error(f"Error loading semantic data from {file_path}: {e}")
            return [], {}

    def _parse_plain_text_format(self, content: str) -> tuple[list, dict]:
        """
        Parse plain text format - handles two cases:
        1. Single column of words (one word per line)
        2. Free text that needs to be split into words and deduplicated (if split_lines is enabled)
        """
        from utils.global_settings import get_global_default_split_lines
        words = []
        lines = content.split('\n')

        # Check if this looks like single-word-per-line format
        single_word_lines = 0
        multi_word_lines = 0

        for line in lines[:10]:  # Check first 10 lines to determine format
            clean_line = line.strip()
            if clean_line and not clean_line.startswith('#'):
                word_count = len(clean_line.split())
                if word_count == 1:
                    single_word_lines += 1
                elif word_count > 1:
                    multi_word_lines += 1

        # Get global setting for line splitting (with fallback)
        try:
            split_lines_enabled = get_global_default_split_lines()
        except Exception:
            # Fallback to True if global settings are not available
            split_lines_enabled = True

        if multi_word_lines > single_word_lines and split_lines_enabled:
            # Free text format - split by spaces, concatenate, and deduplicate
            all_words = []
            for line in lines:
                clean_line = line.strip()
                if clean_line and not clean_line.startswith('#'):  # Skip comments and empty lines
                    # Split by spaces and filter out empty strings
                    line_words = [w.strip() for w in clean_line.split() if w.strip()]
                    all_words.extend(line_words)

            # Deduplicate while preserving order
            seen = set()
            words = []
            for word in all_words:
                if word not in seen:
                    words.append(word)
                    seen.add(word)

            st.info(f"📝 Preprocessed text data: Found {len(all_words)} total words, {len(words)} unique words after deduplication")
        else:
            # Either single word per line format OR multi-word lines with splitting disabled
            if multi_word_lines > single_word_lines and not split_lines_enabled:
                st.info("📝 Multi-word lines detected but line splitting is disabled. Treating each line as a single unit.")

            # Single word/line format (traditional)
            for line in lines:
                word = line.strip()
                if word and not word.startswith('#'):  # Skip comments
                    words.append(word)

        # All words get blue color (default) - return as dictionary
        word_color_map = {word: '#4444FF' for word in words}
        return words, word_color_map

    def _parse_csv_format(self, content: str, language: str) -> tuple[list, dict]:
        """Parse CSV format with word,domain,type,notes columns using pandas"""
        try:
            from io import StringIO
            
            # Parse the dataset CSV using pandas, forcing 'word' column to be treated as string
            # to prevent automatic conversion of numbers like "1" to 1.0
            df = pd.read_csv(StringIO(content), dtype={'word': str})
            
            # Validate required columns
            if 'word' not in df.columns:
                st.warning(f"CSV file missing 'word' column. Using plain text format.")
                return self._parse_plain_text_format(content)
            
            # Extract words (excluding NaN values, already string type)
            words = df['word'].dropna().tolist()
            
            # Get dataset name for color mapping file from various possible session state keys
            input_name = (
                st.session_state.get('main_cfg_input_text_selected', '') or
                st.session_state.get('dual_cfg_input_text_selected', '') or
                st.session_state.get('multilingual_cfg_input_text_selected', '') or
                st.session_state.get('echarts_cfg_input_text_selected', '') or
                st.session_state.get('compare_lang_cfg_input_text_selected', '') or  # Compare page - by language
                st.session_state.get('compare_model_cfg_input_text_selected', '') or  # Compare page - by model
                st.session_state.get('compare_method_cfg_input_text_selected', '') or  # Compare page - by method
                st.session_state.get('cfg_input_text_selected', '') or
                st.session_state.get('input_name_selected', '')
            )
            
            # Build path to color-code CSV file  
            color_code_file = self.input_dir / f"{input_name}-{language}.color-code.csv"
            
            # Create word-to-color mapping
            word_color_map = {}
            
            if color_code_file.exists() and 'domain' in df.columns:
                try:
                    # Load color mapping CSV using pandas
                    color_df = pd.read_csv(color_code_file)
                    
                    # Validate color CSV has required columns
                    if 'domain' in color_df.columns and 'color_hex' in color_df.columns:
                        # Create domain-to-color mapping
                        domain_colors = {}
                        for _, color_row in color_df.iterrows():
                            if pd.notna(color_row['domain']) and pd.notna(color_row['color_hex']):
                                domain = str(color_row['domain']).strip().lower()
                                color = str(color_row['color_hex']).strip()
                                domain_colors[domain] = color
                        
                        # Map each word to its domain color
                        domain_stats = {}
                        for _, data_row in df.iterrows():
                            if pd.notna(data_row['word']):
                                word = str(data_row['word']).strip()
                                domain = str(data_row.get('domain', 'unknown')).strip().lower()
                                
                                # Handle missing/empty domains
                                if pd.isna(data_row.get('domain')) or domain == '' or domain == 'nan':
                                    domain = 'unknown'
                                
                                # Count domain frequency for stats
                                domain_stats[domain] = domain_stats.get(domain, 0) + 1
                                
                                # Get color for this domain
                                color = domain_colors.get(domain, '#CCCCCC')  # Light grey for unknown
                                word_color_map[word] = color
                        
                        # Show success message with stats
                        st.success(f"🎨 Applied domain colors: {len(domain_colors)} domains, {len(word_color_map)} words")
                        with st.expander("🔍 Domain Statistics", expanded=False):
                            for domain, count in sorted(domain_stats.items()):
                                color = domain_colors.get(domain, '#CCCCCC')
                                st.write(f"• **{domain}**: {count} words → `{color}`")
                        
                    else:
                        st.warning(f"Color CSV missing required columns. Expected: domain, color_hex")
                        # Fallback to default colors
                        for word in words:
                            word_color_map[word] = '#4444FF'
                            
                except Exception as color_error:
                    st.error(f"Error loading color mapping: {color_error}")
                    # Fallback to default colors
                    for word in words:
                        word_color_map[word] = '#4444FF'
            else:
                # No color file - check if we have domain column to use CUSTOM_SEMANTIC_DOMAINS
                if 'domain' in df.columns:
                    # Use CUSTOM_SEMANTIC_DOMAINS mapping as fallback
                    from semanscope.config import get_domain_color

                    domain_stats = {}
                    for _, data_row in df.iterrows():
                        if pd.notna(data_row['word']):
                            word = str(data_row['word']).strip()
                            domain = str(data_row.get('domain', 'unknown')).strip()

                            # Handle missing/empty domains
                            if pd.isna(data_row.get('domain')) or domain == '' or domain == 'nan':
                                domain = 'unknown'

                            # Count domain frequency for stats
                            domain_stats[domain] = domain_stats.get(domain, 0) + 1

                            # Get color from CUSTOM_SEMANTIC_DOMAINS via get_domain_color
                            color = get_domain_color(domain)
                            word_color_map[word] = color

                    # Show info about using CUSTOM_SEMANTIC_DOMAINS
                    st.info(f"ℹ️ No color mapping file found: `{color_code_file.name}`. Using CUSTOM_SEMANTIC_DOMAINS mapping.")
                    st.success(f"🎨 Applied {len(set(domain_stats.keys()))} domain colors from config: {len(word_color_map)} words")

                    # Show domain statistics
                    with st.expander("🔍 Domain Color Mapping", expanded=False):
                        for domain, count in sorted(domain_stats.items()):
                            color = get_domain_color(domain)
                            st.write(f"• **{domain}**: {count} words → `{color}`")
                else:
                    # No domain column at all - use default blue
                    for word in words:
                        word_color_map[word] = '#4444FF'
                    st.info(f"ℹ️ Dataset has no 'domain' column - using default colors")
            
            return words, word_color_map
            
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return self._parse_plain_text_format(content)
    
    def save_text_to_file(self, input_name: str, chinese_text: str, english_text: str, 
                          chinese_selected: bool, english_selected: bool):
        """Save text content to files"""
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename using the proper method
        safe_name = self.sanitize_filename(input_name)
        
        success_count = 0
        
        # Save Chinese text if provided
        if chinese_selected and chinese_text.strip():
            chn_file = self.input_dir / f"{safe_name}-chn.txt"
            try:
                chn_file.write_text(chinese_text.strip(), encoding='utf-8')
                success_count += 1
            except Exception as e:
                st.error(f"Error saving Chinese text: {e}")
        
        # Save English text if provided
        if english_selected and english_text.strip():
            enu_file = self.input_dir / f"{safe_name}-enu.txt"
            try:
                enu_file.write_text(english_text.strip(), encoding='utf-8')
                success_count += 1
            except Exception as e:
                st.error(f"Error saving English text: {e}")
        
        if success_count > 0:
            st.success(f"Saved {success_count} text file(s) as '{safe_name}'")
            st.rerun()  # Refresh to update the selectbox options
        else:
            st.warning("No text to save")

    def save_multilingual_text(self, input_name_raw: str, chinese_text: str, chinese_selected: bool, 
                              target_words_dict: dict, target_selected_dict: dict, lang_code_map: dict):
        """Save multilingual text content to files"""
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name = self.sanitize_filename(input_name_raw)
        success_count = 0
        
        # Save Chinese text if provided
        if chinese_selected and chinese_text.strip():
            chn_file = self.input_dir / f"{safe_name}-chn.txt"
            try:
                chn_file.write_text(chinese_text.strip(), encoding='utf-8')
                success_count += 1
            except Exception as e:
                st.error(f"Error saving Chinese text: {e}")
        
        # Save target language texts
        reverse_lang_map = {v: k for k, v in lang_code_map.items()}  # Map lang codes back to names
        for lang_code, selected in target_selected_dict.items():
            if selected:
                # Get the actual text from session state
                lang_text = st.session_state.get(f'{lang_code}_text_area', '')
                if lang_text.strip():
                    lang_file = self.input_dir / f"{safe_name}-{lang_code}.txt"
                    try:
                        lang_file.write_text(lang_text.strip(), encoding='utf-8')
                        success_count += 1
                    except Exception as e:
                        lang_name = reverse_lang_map.get(lang_code, lang_code)
                        st.error(f"Error saving {lang_name} text: {e}")
        
        if success_count > 0:
            st.success(f"Saved {success_count} text file(s) as '{safe_name}'")
            st.rerun()  # Refresh to update the selectbox options
        else:
            st.warning("No text to save")

    def save_dynamic_multilingual_text(self, input_name_raw: str, all_texts: dict, all_selected: dict):
        """Save dynamic multilingual text content to files"""
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self.sanitize_filename(input_name_raw)
        success_count = 0

        # Save texts for each selected language
        for lang_code, selected in all_selected.items():
            if selected and lang_code in all_texts:
                text_content = all_texts[lang_code]
                if text_content.strip():
                    lang_file = self.input_dir / f"{safe_name}-{lang_code}.txt"
                    try:
                        lang_file.write_text(text_content.strip(), encoding='utf-8')
                        success_count += 1
                    except Exception as e:
                        st.error(f"Error saving {lang_code} text: {e}")

        if success_count > 0:
            st.success(f"Saved {success_count} text file(s) as '{safe_name}'")
            st.rerun()  # Refresh to update the selectbox options
        else:
            st.warning("No text to save")

    def render_input_areas(self) -> Tuple[List[str], List[str], List[str]]:
        """Render text input areas and return processed words

        Returns:
            Tuple containing:
            - btn_actions: (btn_visualize, btn_rotate_90, btn_save_png)
            - chinese_words: List[str] (LEGACY - for backward compatibility)
            - target_words_dict: Dict[str, List[str]] (LEGACY - for backward compatibility)
            - all_colors: List[str] - Color codes for all words
            - chinese_selected: bool (LEGACY - for backward compatibility)
            - target_selected_dict: Dict[str, bool] (LEGACY - for backward compatibility)
            - selected_languages: Dict - MODERN trilingual structure with lang1/lang2/lang3
        """
        with st.sidebar:
            with st.expander("✏️ Enter Text Data", expanded=True):

                # Input selection dropdown - at the top
                available_inputs = self.get_available_inputs()
                # Use global settings override for default dataset
                default_dataset = get_global_default_dataset()
                input_name_selected = st.selectbox(
                    "Select Input Dataset",
                    options=available_inputs,
                    index=available_inputs.index(default_dataset) if default_dataset in available_inputs else 0,
                    key="input_name_selected"
                )

                # Language selection - three columns for trilingual support
                col_lang1, col_lang2, col_lang3 = st.columns(3)

                # Get language options based on selected dataset
                available_lang_names = []  # Initialize to empty list
                if input_name_selected:
                    # Get ALL available languages for this dataset (for dropdown options)
                    available_lang_codes = self.get_all_available_languages_for_dataset(input_name_selected)
                    # Convert codes to display names for dropdowns
                    for code in available_lang_codes:
                        for name, mapped_code in LANGUAGE_CODE_MAP.items():
                            if mapped_code == code:
                                available_lang_names.append(name)
                                break

                    if len(available_lang_names) >= 1:
                        # Filter dropdown options to show only available languages
                        unique_available = list(dict.fromkeys(available_lang_names))  # Remove duplicates while preserving order
                        lang_options = unique_available

                        # Get smart duplication for auto-selection (separate from dropdown options)
                        smart_selection_codes = self.get_available_languages_for_dataset(input_name_selected)
                        # Convert smart selection codes to display names
                        smart_selected_names = []
                        for code in smart_selection_codes:
                            for name, mapped_code in LANGUAGE_CODE_MAP.items():
                                if mapped_code == code:
                                    smart_selected_names.append(name)
                                    break

                        selected_langs = smart_selected_names

                        # Set default indices based on selected languages
                        default_indices = [
                            lang_options.index(selected_langs[0]),
                            lang_options.index(selected_langs[1]) if selected_langs[1] in lang_options else 0,
                            lang_options.index(selected_langs[2]) if selected_langs[2] in lang_options else 0
                        ]
                    else:
                        # Fallback to full list if no valid languages found
                        lang_options = get_sorted_language_names()
                        default_langs = [get_language_name_from_code(code) for code in DEFAULT_LANG_SET[:3]]
                        default_indices = []
                        for default_lang in default_langs:
                            if default_lang in lang_options:
                                default_indices.append(lang_options.index(default_lang))
                            else:
                                default_indices.append(0)
                        available_lang_names = []  # Clear for help text
                else:
                    # No dataset selected, use full language list with defaults
                    lang_options = get_sorted_language_names()
                    default_langs = [get_language_name_from_code(code) for code in DEFAULT_LANG_SET[:3]]
                    default_indices = []
                    for default_lang in default_langs:
                        if default_lang in lang_options:
                            default_indices.append(lang_options.index(default_lang))
                        else:
                            default_indices.append(0)

                with col_lang1:
                    lang1 = st.selectbox(
                        "**Lang-1**",
                        options=lang_options,
                        index=default_indices[0],
                        help="Select first language" + (f" (dataset-filtered)" if input_name_selected and len(available_lang_names) >= 1 else ""),
                        key='lang1'
                    )

                with col_lang2:
                    lang2 = st.selectbox(
                        "**Lang-2**",
                        options=lang_options,
                        index=default_indices[1],
                        help="Select second language" + (f" (dataset-filtered)" if input_name_selected and len(available_lang_names) >= 1 else ""),
                        key='lang2'
                    )

                with col_lang3:
                    lang3 = st.selectbox(
                        "**Lang-3**",
                        options=lang_options,
                        index=default_indices[2],
                        help="Select third language" + (f" (dataset-filtered)" if input_name_selected and len(available_lang_names) >= 1 else ""),
                        key='lang3'
                    )

                # Language code mapping using helper function
                lang1_code = get_language_code_from_name(lang1)
                lang2_code = get_language_code_from_name(lang2)
                lang3_code = get_language_code_from_name(lang3)

                # Store all languages for easy iteration
                all_languages = [
                    (lang1, lang1_code),
                    (lang2, lang2_code),
                    (lang3, lang3_code)
                ]

                # Load and Refresh buttons on same row
                col_load_txt, col_refresh = st.columns([3, 1])
                with col_load_txt:
                    btn_load_txt = st.button("Load Text", type="primary",
                                             help="Load input texts",
                                             disabled=not input_name_selected,
                                             width='stretch')
                with col_refresh:
                    btn_refresh = st.button("🔄",
                                           help="Refresh available inputs from data/input folder",
                                           key="refresh_inputs",
                                           width='stretch')


                # Handle refresh button click
                if btn_refresh:
                    # Force re-scan of input directory by triggering rerun
                    st.rerun()


                # Initialize text areas with default or loaded content
                default_texts = {
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
                    "tha": "แดง\nน้ำเงิน\nเขียว\nเหลือง\nส้ม"  # Sample Thai colors
                }
                
                # Load text if button is clicked
                if btn_load_txt:
                    missing_files = []
                    loaded_count = 0

                    # Load currently selected languages into their respective positions
                    language_positions = [
                        (lang1, lang1_code, "pos1"),
                        (lang2, lang2_code, "pos2"),
                        (lang3, lang3_code, "pos3")
                    ]

                    for lang_name, lang_code, position in language_positions:
                        words, word_color_map = self.load_semantic_data_from_file(input_name_selected, lang_code)
                        if words:
                            # Ignore/skip lines starting with # (comment lines) before displaying
                            filtered_words = [word for word in words if not word.strip().startswith('#')]
                            # Convert back to text format for text area display
                            loaded_text = '\n'.join(filtered_words)
                            # Update both the old key (for backward compatibility) and new position-based key
                            st.session_state[f"{lang_code}_text_area"] = loaded_text
                            st.session_state[f"{lang_code}_text_area_{position}"] = loaded_text
                            # Store semantic colors as dictionary for later use
                            session_key = f"{lang_code}_semantic_colors"
                            st.session_state[session_key] = word_color_map
                            loaded_count += 1
                        else:
                            # Clear text area and semantic colors for missing files
                            st.session_state[f"{lang_code}_text_area"] = ""
                            st.session_state[f"{lang_code}_text_area_{position}"] = ""
                            st.session_state[f"{lang_code}_semantic_colors"] = {}
                            missing_files.append(f"'{input_name_selected}-{lang_code}.txt'")

                    if missing_files:
                        st.warning("No text files found: " + " ; ".join(missing_files))
                    if loaded_count > 0:
                        st.success(f"Loaded {loaded_count} language file(s)")

                    # Force UI refresh to immediately show cleared/loaded text areas
                    st.rerun()


                # Handle trilingual text input - show three text areas in columns
                col1, col2, col3 = st.columns(3)

                # Store text contents and selections for all languages
                language_data = {}

                # Lang-1 (left column)
                with col1:
                    lang1_text = st.text_area(
                        f"{lang1} ({lang1_code}):",
                        value=st.session_state.get(f'{lang1_code}_text_area_pos1', st.session_state.get(f'{lang1_code}_text_area', default_texts.get(lang1_code, ""))),
                        height=200,
                        key=f'{lang1_code}_text_area_pos1'
                    )
                    lang1_text = (lang1_text or "").strip()
                    lang1_content = st.session_state.get(f'{lang1_code}_text_area_pos1', st.session_state.get(f'{lang1_code}_text_area', '')).strip()
                    lang1_selected = st.checkbox("Include", value=(len(lang1_content) > 0), key=f'{lang1_code}_include_checkbox_pos1')
                    lang1_words = self.process_text(lang1_content) if lang1_content else []

                    # Word count display
                    if lang1_words:
                        st.caption(f"📊 **{len(lang1_words)} words**")
                    else:
                        st.caption("📊 **0 words**")

                    language_data[lang1_code] = {
                        'name': lang1,
                        'text': lang1_text,
                        'selected': lang1_selected,
                        'words': lang1_words
                    }

                # Lang-2 (middle column)
                with col2:
                    lang2_text = st.text_area(
                        f"{lang2} ({lang2_code}):",
                        value=st.session_state.get(f'{lang2_code}_text_area_pos2', st.session_state.get(f'{lang2_code}_text_area', default_texts.get(lang2_code, ""))),
                        height=200,
                        key=f'{lang2_code}_text_area_pos2'
                    )
                    lang2_text = (lang2_text or "").strip()
                    lang2_content = st.session_state.get(f'{lang2_code}_text_area_pos2', st.session_state.get(f'{lang2_code}_text_area', '')).strip()
                    lang2_selected = st.checkbox("Include", value=(len(lang2_content) > 0), key=f'{lang2_code}_include_checkbox_pos2')
                    lang2_words = self.process_text(lang2_content) if lang2_content else []

                    # Word count display
                    if lang2_words:
                        st.caption(f"📊 **{len(lang2_words)} words**")
                    else:
                        st.caption("📊 **0 words**")

                    language_data[lang2_code] = {
                        'name': lang2,
                        'text': lang2_text,
                        'selected': lang2_selected,
                        'words': lang2_words
                    }

                # Lang-3 (right column)
                with col3:
                    lang3_text = st.text_area(
                        f"{lang3} ({lang3_code}):",
                        value=st.session_state.get(f'{lang3_code}_text_area_pos3', st.session_state.get(f'{lang3_code}_text_area', default_texts.get(lang3_code, ""))),
                        height=200,
                        key=f'{lang3_code}_text_area_pos3'
                    )
                    lang3_text = (lang3_text or "").strip()
                    lang3_content = st.session_state.get(f'{lang3_code}_text_area_pos3', st.session_state.get(f'{lang3_code}_text_area', '')).strip()
                    lang3_selected = st.checkbox("Include", value=(len(lang3_content) > 0), key=f'{lang3_code}_include_checkbox_pos3')
                    lang3_words = self.process_text(lang3_content) if lang3_content else []

                    # Word count display
                    if lang3_words:
                        st.caption(f"📊 **{len(lang3_words)} words**")
                    else:
                        st.caption("📊 **0 words**")

                    language_data[lang3_code] = {
                        'name': lang3,
                        'text': lang3_text,
                        'selected': lang3_selected,
                        'words': lang3_words
                    }

                # Check for duplicate language codes and show warning
                lang_codes = [lang1_code, lang2_code, lang3_code]
                unique_codes = set(lang_codes)
                if len(unique_codes) < len(lang_codes):
                    duplicate_codes = [code for code in unique_codes if lang_codes.count(code) > 1]
                    st.warning(f"⚠️ **Duplicate Language Detected**: {', '.join(duplicate_codes)} selected multiple times. This is allowed, but results may show overlapping data points.")

                # ========================================
                # MODERN PARADIGM: Trilingual Structure
                # ========================================
                # Build clean data structure treating all languages equally
                # No more source/target hierarchy - lang1/lang2/lang3 are peers
                # FIX: Use position-specific selections instead of language_data to avoid duplicate overwrite issue
                selected_languages = {
                    'lang1': {
                        'name': lang1,
                        'code': lang1_code,
                        'words': lang1_words,
                        'selected': lang1_selected
                    },
                    'lang2': {
                        'name': lang2,
                        'code': lang2_code,
                        'words': lang2_words,
                        'selected': lang2_selected
                    },
                    'lang3': {
                        'name': lang3,
                        'code': lang3_code,
                        'words': lang3_words,
                        'selected': lang3_selected
                    }
                }

                # ========================================
                # BACKWARD COMPATIBILITY LAYER
                # ========================================
                # TODO: Remove this section once all code migrated to selected_languages
                # Maps new structure to old source/target paradigm for gradual migration
                source_words = []
                chinese_words = []
                chinese_selected = False
                target_words_dict = {}
                target_selected_dict = {}

                # Find the first selected language to use as "source"
                source_found = False
                for lang_code, data in language_data.items():
                    if data['selected'] and data['words']:
                        if not source_found:
                            # This becomes the primary source
                            source_words = data['words']
                            source_found = True
                            # Map to Chinese for backward compatibility if it's Chinese
                            if lang_code == "chn":
                                chinese_words = data['words']
                                chinese_selected = True
                        else:
                            # All others go to target_words_dict
                            target_words_dict[lang_code] = data['words']
                            target_selected_dict[lang_code] = True
                            # Also map Chinese if found in targets
                            if lang_code == "chn":
                                chinese_words = data['words']
                                chinese_selected = True

                # If no Chinese was found in source or targets, but we have source words, map for compatibility
                if not chinese_selected and source_words:
                    chinese_words = source_words
                    chinese_selected = True

                # User can enter a name for the input and save the texts 
                col_input_enter, col_save_txt = st.columns([3, 1])
                with col_input_enter:
                    input_name_raw = st.text_input(
                        "Name Input",
                        value=st.session_state.get("input_name_selected", "untitled"),
                        key="cfg_input_text_entered",
                        help="Name will be automatically sanitized for filename compatibility"
                    )
                    # Show sanitized preview
                    sanitized_preview = self.sanitize_filename(input_name_raw)
                    if sanitized_preview != input_name_raw.lower():
                        st.caption(f"📝 Preview: `{sanitized_preview}`")
                        
                with col_save_txt:
                    btn_save_txt = st.button("Save Text", type="primary", 
                                             help="Save input texts", 
                                             disabled=(input_name_raw=="untitled"))
                    
                # Handle save text
                if btn_save_txt:
                    # Save all three languages
                    all_texts = {}
                    all_selected = {}

                    for lang_code, data in language_data.items():
                        all_texts[lang_code] = data['text'] if data['selected'] else ""
                        all_selected[lang_code] = data['selected']

                    self.save_dynamic_multilingual_text(input_name_raw, all_texts, all_selected)

            col_vis, col_rotate, col_save_png = st.columns([1, 1, 1])
            with col_vis:
                btn_visualize = st.button("Visualize", type="primary")
            with col_rotate:
                btn_rotate_90 = st.button("Rotate", help="Rotate 2D plot by 90°")
            with col_save_png:
                btn_save_png = st.button("Save Image", help="Save current plot as PNG image")
            btn_actions = (btn_visualize, btn_rotate_90, btn_save_png)

            # Combine all words and create corresponding colors with semantic support
            all_words = []
            all_colors = []

            # Map language codes to color map keys (fallback for non-semantic data)
            lang_color_map = {
                "chn": "chinese",
                "enu": "english",
                "fra": "french",
                "spa": "spanish",
                "deu": "german",
                "ara": "arabic"
            }

            # Process all selected languages
            for lang_code, data in language_data.items():
                if data['selected'] and data['words']:
                    all_words.extend(data['words'])

                    # Check if we have semantic color mapping stored for this language
                    semantic_color_map = st.session_state.get(f"{lang_code}_semantic_colors", {})

                    # If no session-stored mapping, try to get domain colors from currently selected dataset
                    if not semantic_color_map:
                        input_name = st.session_state.get('input_name_selected', '')
                        if input_name:
                            # Try to load fresh domain mapping for current dataset
                            dataset_words, dataset_color_map = self.load_semantic_data_from_file(input_name, lang_code)
                            if dataset_color_map:
                                semantic_color_map = dataset_color_map

                    if semantic_color_map:
                        # Map each word to its semantic color
                        for word in data['words']:
                            if word in semantic_color_map:
                                color = semantic_color_map[word]
                                all_colors.append(color)
                            else:
                                # Word not found in mapping, use fallback
                                color_key = lang_color_map.get(lang_code, "english")
                                fallback_color = COLOR_MAP[color_key]
                                all_colors.append(fallback_color)
                    else:
                        # Fallback to language-based coloring
                        color_key = lang_color_map.get(lang_code, "english")
                        all_colors.extend([COLOR_MAP[color_key]] * len(data['words']))

            # ========================================
            # RETURN VALUE
            # ========================================
            # Returns both NEW and LEGACY structures during migration period
            #
            # MODERN (recommended): Use selected_languages (last element)
            # LEGACY (deprecated): Use chinese_words, target_words_dict, etc.
            #
            # Example usage:
            #   _, _, _, colors, _, _, selected_languages = visualizer.render_input_areas()
            #   # Use selected_languages for modern code paths
            return btn_actions, chinese_words, target_words_dict, all_colors, chinese_selected, target_selected_dict, selected_languages
    
    def get_embeddings(self, words: List[str], model_name: str, lang: str, debug_flag: bool = False) -> np.ndarray:
        """Get embeddings for words using specified model with cross-page caching"""

        # Preprocess texts to handle problematic characters (especially Chinese radicals)
        if ENABLE_CHINESE_TEXT_PREPROCESSING and any('\u4e00' <= char <= '\u9fff' for word in words for char in word):
            # Contains Chinese characters, apply preprocessing
            preprocessed_words = preprocess_texts_for_embedding(words, show_warnings=SHOW_PREPROCESSING_WARNINGS)
        else:
            # No Chinese characters or preprocessing disabled, use original words
            preprocessed_words = words

        # Try to get cached embeddings first
        cached_embeddings = get_cached_embeddings(preprocessed_words, lang, model_name)
        if cached_embeddings is not None:
            return cached_embeddings

        # No cache hit, generate embeddings
        # Use get_model_with_strategy to respect global optimization setting
        model = get_model_with_strategy(model_name)
        embeddings = model.get_embeddings(preprocessed_words, lang, debug_flag=debug_flag)

        # Save to cache for future use
        if embeddings is not None:
            save_embeddings_to_cache(embeddings, preprocessed_words, lang, model_name)

        return embeddings

    
    def sanitize_filename(self, text: str) -> str:
        """Sanitize text for use in filename"""


        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove or replace characters that are problematic in filenames
        # Keep alphanumeric, Chinese/CJK characters, hyphens, underscores, and spaces
        sanitized = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf-]', '', text)

        # Replace multiple whitespace with single underscore
        sanitized = re.sub(r'\s+', '_', sanitized.strip())

        # Remove duplicate underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')

        return sanitized if sanitized else "untitled"

    
    def save_plot_image(self, input_name: str, model_name: str, method_name: str, chinese_selected: bool, english_selected: bool, dimensions: str = "2D"):
        """Save the current plot as image with language tags and dimension suffix, supporting PNG/PDF formats"""
        if st.session_state.current_figure is None:
            st.warning("No plot to save. Please generate a visualization first.")
            return ""

        # Get publication settings to determine export format and directory
        pub_settings = st.session_state.get('global_settings', {}).get('publication', {})
        publication_mode = pub_settings.get('publication_mode', False)
        export_format = pub_settings.get('export_format', 'PNG').upper()
        export_dpi = pub_settings.get('export_dpi', 300)
        plot_width = pub_settings.get('plot_width', 1400)
        plot_height = pub_settings.get('plot_height', 1100)

        # Determine output directory based on format
        if export_format == 'PDF' and publication_mode:
            output_dir = self.images_dir / "PDF"
        else:
            output_dir = self.images_dir

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect active languages
        lang_tags = []
        lang_code_map = LANGUAGE_CODE_MAP

        # Check for language selections from multiple possible key prefixes
        # Try main page keys first, then dual page keys, then default keys
        key_prefixes = ['main_', 'dual_', '']

        for prefix in key_prefixes:
            # Get language selections for this prefix
            lang1 = st.session_state.get(f'{prefix}lang1', 'Chinese')
            lang2 = st.session_state.get(f'{prefix}lang2', 'English')
            lang3 = st.session_state.get(f'{prefix}lang3', 'German')

            lang1_code = lang_code_map.get(lang1, "chn")
            lang2_code = lang_code_map.get(lang2, "enu")
            lang3_code = lang_code_map.get(lang3, "deu")

            # Check which languages are actually selected via checkboxes
            if st.session_state.get(f'{lang1_code}_include_checkbox', False):
                lang_tags.append(lang1_code.upper())
            if st.session_state.get(f'{lang2_code}_include_checkbox', False) and lang2_code.upper() not in lang_tags:
                lang_tags.append(lang2_code.upper())
            if st.session_state.get(f'{lang3_code}_include_checkbox', False) and lang3_code.upper() not in lang_tags:
                lang_tags.append(lang3_code.upper())

            # If we found languages with this prefix, break
            if lang_tags:
                break

        # Fallback to boolean parameters if session state detection fails (backward compatibility)
        if not lang_tags:
            if chinese_selected:
                lang_tags.append("CHN")
            if english_selected:
                lang_tags.append("ENU")

        # Use centralized helper to generate consistent filename
        file_extension = export_format.lower() if publication_mode else "png"
        _, filename = create_title_and_filename(
            [method_name],
            [model_name],
            input_name,
            lang_tags,
            file_extension
        )

        # Add 3D suffix if it's a 3D visualization (insert before file extension)
        if dimensions == "3D":
            name_part, ext_part = filename.rsplit('.', 1)
            filename = f"{name_part}-3d.{ext_part}"
        file_path = output_dir / filename

        try:
            # Save the figure in the specified format
            if publication_mode and export_format == 'PDF':
                st.session_state.current_figure.write_image(
                    str(file_path),
                    format="pdf",
                    width=plot_width,
                    height=plot_height
                )
            elif publication_mode and export_format == 'SVG':
                st.session_state.current_figure.write_image(
                    str(file_path),
                    format="svg",
                    width=plot_width,
                    height=plot_height
                )
            else:
                # Default PNG format
                scale_factor = export_dpi / 96 if publication_mode else 2
                st.session_state.current_figure.write_image(
                    str(file_path),
                    format="png",
                    width=plot_width if publication_mode else 1200,
                    height=plot_height if publication_mode else 800,
                    scale=scale_factor
                )

            # st.success(f"Image saved as: {filename}")
            return filename
        except Exception as e:
            st.error(f"Error saving image: {e}")
            return ""

    def save_detail_view_image(self, detail_figure, input_name: str, model_name: str, method_name: str, chinese_selected: bool, english_selected: bool):
        """Save the detail view plot as PNG image with zoom ID"""
        if detail_figure is None:
            st.warning("No detail view to save. Please generate a visualization first.")
            return ""
            
        # Ensure images directory exists
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sanitized filename
        safe_input = self.sanitize_filename(input_name)
        safe_model = self.sanitize_filename(model_name)
        safe_method = self.sanitize_filename(method_name)
        
        # Add language tags
        lang_tags = []
        if chinese_selected:
            lang_tags.append("chn")
        if english_selected:
            lang_tags.append("enu")
        
        lang_suffix = "-".join(lang_tags) if lang_tags else "none"
        
        # Initialize zoom counter if not exists
        if 'zoom_save_counter' not in st.session_state:
            st.session_state.zoom_save_counter = 1
        
        zoom_id = st.session_state.zoom_save_counter
        filename = f"{safe_input}-{safe_model}-{safe_method}-{lang_suffix}-zoom-{zoom_id}.png"
        file_path = self.images_dir / filename
        
        try:
            # Save the detail figure as PNG with higher resolution for paper figures
            detail_figure.write_image(str(file_path), width=1600, height=1200, scale=2)
            
            # Increment counter for next save
            st.session_state.zoom_save_counter += 1
            
            return filename
        except Exception as e:
            st.error(f"Error saving detail view image: {e}")
            return ""

    def create_plot(self, embeddings, labels, colors, model_name, method_name,
                    dimensions, do_clustering, n_clusters, dataset_name="User Input", lang_codes=None, word_search_config=None):
        """Create and display the plot"""
        plot_title = f"[Model] {model_name}, [Method] {method_name}"
        plot_mgr = PlotManager()
        
        # Apply rotation if needed (only for 2D plots)
        if dimensions == "2D" and st.session_state.plot_rotation > 0:
            # Apply rotation transformation
            angle = np.radians(st.session_state.plot_rotation)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            embeddings = embeddings @ rotation_matrix.T

        if dimensions == "2D":
            fig = plot_mgr.plot_2d(
                embeddings=embeddings,
                labels=labels,
                colors=colors,
                title=plot_title,
                clustering=do_clustering,
                n_clusters=n_clusters if do_clustering else None,
                method_name=method_name,
                model_name=model_name,
                dataset_name=dataset_name,
                lang_codes=lang_codes,
                word_search_config=word_search_config
            )
        else:
            fig = plot_mgr.plot_3d(
                embeddings=embeddings,
                labels=labels,
                colors=colors,
                title=plot_title,
                clustering=do_clustering,
                n_clusters=n_clusters if do_clustering else None,
                method_name=method_name,
                model_name=model_name,
                dataset_name=dataset_name,
                lang_codes=lang_codes,
                word_search_config=word_search_config
            )
        
        # Store the figure in session state for saving
        st.session_state.current_figure = fig
        
    def display_saved_images(self):
        """Display all saved images in the images directory"""
        if not self.images_dir.exists():
            st.info("No images saved yet. Generate a visualization and click 'Save Image'.")
            return
            
        image_files = list(self.images_dir.glob("*.png"))
        
        if not image_files:
            st.info("No images found in the images directory.")
            return
            
        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        st.write(f"Found {len(image_files)} saved images:")
        
        # Display images in a grid
        cols = st.columns(2)  # 2 columns for images
        
        for idx, image_file in enumerate(image_files):
            col = cols[idx % 2]
            
            with col:
                # Display filename
                st.write(f"**{image_file.name}**")
                
                # Display image
                try:
                    st.image(str(image_file), caption=image_file.stem, width='stretch')
                    
                    # Add download button
                    with open(image_file, "rb") as file:
                        st.download_button(
                            label=f"Download {image_file.name}",
                            data=file.read(),
                            file_name=image_file.name,
                            mime="image/png",
                            key=f"download_{image_file.stem}_{idx}"
                        )
                        
                    # Add delete button
                    if st.button(f"Delete", key=f"delete_{image_file.stem}_{idx}", help=f"Delete {image_file.name}"):
                        try:
                            image_file.unlink()
                            st.success(f"Deleted {image_file.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting {image_file.name}: {e}")
                    
                    st.divider()
                    
                except Exception as e:
                    st.error(f"Error displaying {image_file.name}: {e}")