import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from semanscope.config import check_login, DATA_PATH
from semanscope.utils.unitranslator import UniTranslator

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Deep-Translator",
    page_icon="🌐",
    layout="wide"
)

# Get provider information from UniTranslator
PROVIDER_INFO = UniTranslator.get_provider_info()


class MultiProviderTranslator:
    """Wrapper for managing multiple translation providers using UniTranslator"""

    def __init__(self):
        self.available_providers = {}
        self.unavailable_providers = {}
        self.traditional_providers = {}
        self.llm_providers = {}
        self._check_providers()

    def _check_providers(self):
        """Check which providers are available based on API keys"""
        for name, info in PROVIDER_INFO.items():
            provider_type = info.get("type", "traditional")

            if info["requires_key"]:
                api_key = os.getenv(info["env_var"])
                if api_key:
                    self.available_providers[name] = info
                    if provider_type == "llm":
                        self.llm_providers[name] = info
                    else:
                        self.traditional_providers[name] = info
                else:
                    self.unavailable_providers[name] = info
            else:
                self.available_providers[name] = info
                if provider_type == "llm":
                    self.llm_providers[name] = info
                else:
                    self.traditional_providers[name] = info

    def get_supported_languages(self, provider_name: str) -> Dict[str, str]:
        """Get supported languages with display names"""
        # Common language codes supported by most providers
        return {
            "auto": "Auto-detect",
            "en": "English",
            "zh": "Chinese (Simplified)",
            "zh-CN": "Chinese (Simplified)",
            "zh-TW": "Chinese (Traditional)",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish"
        }

    def translate(self, text: str, source_lang: str, target_lang: str,
                  provider_name: str, line_by_line: bool = True) -> Tuple[str, Optional[str]]:
        """
        Translate text using specified provider

        Returns:
            Tuple of (translated_text, error_message)
        """
        try:
            # Handle auto-detect for source language
            if source_lang == "auto":
                source_lang = "auto"

            # Create UniTranslator instance
            translator = UniTranslator(
                provider=provider_name,
                source=source_lang,
                target=target_lang
            )

            # Translate
            if line_by_line and '\n' in text:
                lines = text.split('\n')
                translated_lines = []
                for line in lines:
                    if line.strip():
                        translated = translator.translate(line.strip())
                        translated_lines.append(translated)
                    else:
                        translated_lines.append('')
                result = '\n'.join(translated_lines)
            else:
                result = translator.translate(text)

            return result, None

        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            return "", error_msg


def main():
    # Check login status
    check_login()

    st.markdown("### 🌐 Multi-Provider Translation Comparison with LLM Support")
    st.caption("Compare traditional translation services with AI language models")

    # Initialize translator
    translator = MultiProviderTranslator()

    # Sidebar - Provider selection and information
    with st.sidebar:
        st.subheader("🔧 Translation Providers")

        # Show provider statistics
        total_available = len(translator.available_providers)
        total_traditional = len(translator.traditional_providers)
        total_llm = len(translator.llm_providers)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", total_available)
        with col2:
            st.metric("Traditional", total_traditional)
        with col3:
            st.metric("LLM", total_llm)

        st.divider()

        # Available providers with categorization
        if translator.available_providers:
            # Traditional providers
            traditional_available = [k for k in translator.traditional_providers.keys()]
            llm_available = [k for k in translator.llm_providers.keys()]

            selected_providers = []

            if traditional_available:
                st.markdown("**📡 Traditional Translation APIs**")
                selected_traditional = st.multiselect(
                    "Select traditional providers:",
                    options=traditional_available,
                    default=[traditional_available[0]] if traditional_available else [],
                    help="Traditional translation services",
                    key="traditional_select"
                )
                selected_providers.extend(selected_traditional)

            if llm_available:
                st.markdown("**🤖 LLM-based Translation**")
                selected_llm = st.multiselect(
                    "Select LLM providers:",
                    options=llm_available,
                    default=[],
                    help="AI language models via OpenRouter",
                    key="llm_select"
                )
                selected_providers.extend(selected_llm)

            if not selected_providers and translator.available_providers:
                # Fallback to first available
                selected_providers = [list(translator.available_providers.keys())[0]]

        else:
            st.error("No providers available!")
            selected_providers = []

        # Unavailable providers with setup instructions
        if translator.unavailable_providers:
            st.divider()
            st.warning(f"⚠️ {len(translator.unavailable_providers)} providers need setup")

            with st.expander("🔑 Setup Additional Providers", expanded=False):
                # Group by traditional and LLM
                unavailable_traditional = {k: v for k, v in translator.unavailable_providers.items()
                                          if v.get("type", "traditional") == "traditional"}
                unavailable_llm = {k: v for k, v in translator.unavailable_providers.items()
                                  if v.get("type", "traditional") == "llm"}

                if unavailable_traditional:
                    st.markdown("**Traditional Providers:**")
                    for name, info in unavailable_traditional.items():
                        st.markdown(f"**{name}**")
                        st.markdown(f"*{info['description']}*")
                        st.info(f"""
**Setup:**
1. Visit: {info['website']}
2. Get API key
3. Add to `.env`:
```bash
{info['env_var']}=your_api_key
```
                        """)
                        st.markdown(f"💡 {info.get('cost', 'See website for pricing')}")
                        st.divider()

                if unavailable_llm:
                    st.markdown("**LLM Providers (via OpenRouter):**")
                    st.info("""
**One API Key for All LLMs:**
Get your `OPENROUTER_API_KEY` at https://openrouter.ai/

Add to your `.env` file:
```bash
OPENROUTER_API_KEY=your_api_key
```

This unlocks all LLM providers below:
                    """)
                    for name, info in unavailable_llm.items():
                        st.markdown(f"- **{name}**: {info.get('cost', 'See OpenRouter pricing')}")

        # Provider detailed information
        st.divider()
        st.subheader("📚 Provider Information")

        # Show info for selected providers
        for name in selected_providers:
            if name in PROVIDER_INFO:
                info = PROVIDER_INFO[name]
                provider_type = "🤖 LLM" if info.get("type") == "llm" else "📡 Traditional"

                with st.expander(f"{provider_type} {name}", expanded=False):
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown(f"**Website:** {info['website']}")
                    st.markdown(f"**✅ Strengths:** {info['strengths']}")
                    st.markdown(f"**⚠️ Limitations:** {info['limitations']}")
                    st.markdown(f"**💰 Cost:** {info.get('cost', 'N/A')}")

        # Usage tips
        st.divider()
        with st.expander("💡 Usage Tips", expanded=False):
            st.markdown("""
### 🎯 Compare Traditional vs LLM:
- **Google Translate vs Gemini**: Same vendor, different approaches
- **Traditional**: Fast, deterministic, cost-effective
- **LLM**: Context-aware, handles nuance, better for creative text

### 🔧 Best Use Cases:
**Traditional Services:**
- High-volume translations
- Consistent terminology
- Cost-sensitive projects
- Real-time applications

**LLM Translation:**
- Literary/creative content
- Cultural nuance preservation
- Idioms and wordplay
- Technical context understanding

### 📊 Research Applications:
- Cross-validate translations
- Study provider biases
- Analyze LLM vs rule-based differences
- Build consensus datasets
            """)

    # Main interface
    if not selected_providers:
        st.warning("⚠️ Please select at least one provider from the sidebar")
        return

    # For single provider, use full height
    text_height = 300

    # Language selection
    col_source, col_target = st.columns(2)

    with col_source:
        lang_dict = translator.get_supported_languages(selected_providers[0])
        lang_codes = list(lang_dict.keys())
        lang_names = list(lang_dict.values())

        # Default to zh-CN for source
        default_source = "zh-CN" if "zh-CN" in lang_codes else lang_codes[0]
        default_source_idx = lang_codes.index(default_source)

        source_idx = st.selectbox(
            "Source language:",
            options=range(len(lang_codes)),
            index=default_source_idx,
            format_func=lambda i: f"{lang_names[i]} ({lang_codes[i]})",
            key="source_lang_select"
        )
        source_lang = lang_codes[source_idx]

    with col_target:
        # Remove "auto" from target languages
        target_codes = [code for code in lang_codes if code != "auto"]
        target_names = [lang_dict[code] for code in target_codes]

        # Default to English for target
        default_target = "en" if "en" in target_codes else target_codes[0]
        default_target_idx = target_codes.index(default_target)

        target_idx = st.selectbox(
            "Target language:",
            options=range(len(target_codes)),
            index=default_target_idx,
            format_func=lambda i: f"{target_names[i]} ({target_codes[i]})",
            key="target_lang_select"
        )
        target_lang = target_codes[target_idx]

    # Debug: Show selected languages
    st.caption(f"🔍 Translation direction: {source_lang} → {target_lang}")

    # Clear cached translations if language selection changed
    if 'last_source_lang' not in st.session_state:
        st.session_state.last_source_lang = source_lang
    if 'last_target_lang' not in st.session_state:
        st.session_state.last_target_lang = target_lang

    if (st.session_state.last_source_lang != source_lang or
        st.session_state.last_target_lang != target_lang):
        # Language changed - clear old translations
        if 'translations' in st.session_state:
            del st.session_state['translations']
        st.session_state.last_source_lang = source_lang
        st.session_state.last_target_lang = target_lang

    # Translation mode
    col_mode, col_save = st.columns([3, 1])
    with col_mode:
        line_by_line = st.checkbox(
            "Block/Passage mode",
            value=False,
            help="When enabled (recommended), translates the entire text as a passage for better context. When disabled, translates each line independently."
        )
    with col_save:
        if st.button("💾 Save All", width='stretch') and 'translations' in st.session_state:
            # Save all translations
            translations_dir = DATA_PATH / "translations"
            translations_dir.mkdir(parents=True, exist_ok=True)

            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = translations_dir / f"multi_translation_{timestamp}.txt"

            content = f"Source ({source_lang}):\n{st.session_state.get('source_text_input', '')}\n\n"
            content += "=" * 80 + "\n\n"

            for provider, translation in st.session_state.translations.items():
                provider_type = PROVIDER_INFO[provider].get("type", "traditional")
                content += f"{provider} ({provider_type}) Translation ({target_lang}):\n{translation}\n\n"
                content += "-" * 80 + "\n\n"

            try:
                filename.write_text(content, encoding='utf-8')
                st.success(f"✅ Saved: {filename.name}")
            except Exception as e:
                st.error(f"❌ Save error: {e}")

    st.divider()

    # Side-by-side layout: Source (left) and Translations (right)
    col_source, col_target = st.columns([1, 1])

    # LEFT COLUMN: Source text
    with col_source:
        st.subheader("📝 Source Text")

        # Initialize with default text if not already set
        if 'source_text_input' not in st.session_state:
            st.session_state.source_text_input = "丈夫\n妻子"

        source_text = st.text_area(
            "Enter text to translate:",
            height=text_height,
            placeholder="Type or paste your text here...",
            key="source_text_input",
            label_visibility="collapsed"
        )

        # Action buttons for source
        col_translate, col_clear, col_example = st.columns([2, 1, 1])

        with col_translate:
            translate_clicked = st.button("🔄 Translate All", type="primary", width='stretch')

        with col_clear:
            if st.button("🗑️ Clear", width='stretch'):
                st.session_state.source_text_input = ""
                if 'translations' in st.session_state:
                    del st.session_state['translations']
                st.rerun()

        with col_example:
            if st.button("📋 Example", width='stretch'):
                example_text = "机器学习正在改变我们理解语言的方式。\n深度学习模型能够捕捉语义的几何结构。\n这为跨语言研究开辟了新的可能性。"
                st.session_state.source_text_input = example_text
                st.rerun()

        # Show source text stats
        if source_text:
            st.caption(f"📊 {len(source_text)} characters, {len(source_text.split())} words")

    # RIGHT COLUMN: Translations
    with col_target:
        st.subheader("🔄 Translations")

        # Perform translation
        if translate_clicked:
            if not source_text.strip():
                st.warning("Please enter text to translate.")
            else:
                st.session_state.translations = {}

                with st.spinner("Translating with all providers..."):
                    progress_bar = st.progress(0)
                    total_providers = len(selected_providers)

                    for idx, provider in enumerate(selected_providers):
                        # Update progress
                        progress_bar.progress((idx + 1) / total_providers)

                        # Debug: Show what we're passing
                        print(f"\n[PAGE DEBUG] Calling translate for {provider}:")
                        print(f"  - Source lang: {source_lang}")
                        print(f"  - Target lang: {target_lang}")
                        print(f"  - Text: {source_text[:50]}...")

                        translated_text, error = translator.translate(
                            source_text, source_lang, target_lang, provider,
                            not line_by_line  # Inverted: unchecked = line-by-line, checked = block/passage
                        )

                        print(f"  - Result: {translated_text[:50] if translated_text else 'ERROR'}...")

                        if error:
                            st.session_state.translations[provider] = f"❌ Error: {error}"
                        else:
                            st.session_state.translations[provider] = translated_text

                        # Store metadata for debugging
                        if 'translation_metadata' not in st.session_state:
                            st.session_state.translation_metadata = {}
                        st.session_state.translation_metadata[provider] = {
                            'source': source_lang,
                            'target': target_lang
                        }

                    progress_bar.empty()

                st.rerun()

        # Display translations
        if 'translations' in st.session_state and st.session_state.translations:
            # Check if showing cached results
            cached_source = st.session_state.get('translation_metadata', {})
            if cached_source:
                first_provider = list(cached_source.keys())[0]
                cached_langs = cached_source[first_provider]
                if (cached_langs.get('source') != source_lang or
                    cached_langs.get('target') != target_lang):
                    st.warning(f"⚠️ Showing cached results for {cached_langs.get('source')} → {cached_langs.get('target')}. Click 'Translate All' to update.")

            # Separate traditional and LLM translations
            traditional_results = {}
            llm_results = {}

            for provider, translation in st.session_state.translations.items():
                if provider in PROVIDER_INFO:
                    provider_type = PROVIDER_INFO[provider].get("type", "traditional")
                    if provider_type == "llm":
                        llm_results[provider] = translation
                    else:
                        traditional_results[provider] = translation



            # Display traditional translations
            if traditional_results:
                if len(traditional_results) > 1:
                    st.markdown("**📡 Traditional Services**")

                for provider, translation in traditional_results.items():
                    # Show debug info about what languages were used
                    metadata = st.session_state.get('translation_metadata', {}).get(provider, {})
                    src_lang = metadata.get('source', '?')
                    tgt_lang = metadata.get('target', '?')

                    if translation.startswith("❌"):
                        st.error(translation)
                    else:
                        st.text_area(
                            f"{provider} result:",
                            value=translation,
                            height=text_height,
                            key=f"translation_trad_{provider}",
                            label_visibility="collapsed"
                        )
                        st.caption(f"📊 {len(translation)} characters")

                    st.markdown(f"**{provider}** `({src_lang} → {tgt_lang})`")

            # Count successful translations
            successful = sum(1 for t in st.session_state.translations.values() if not t.startswith("❌"))
            st.caption(f"✅ {successful}/{len(st.session_state.translations)} successful")


            # Display LLM translations
            if llm_results:
                if traditional_results:
                    st.divider()

                if len(llm_results) > 1:
                    st.markdown("**🤖 LLM Translation**")

                for provider, translation in llm_results.items():
                    provider_cost = PROVIDER_INFO[provider].get("cost", "")
                    # Show debug info about what languages were used
                    metadata = st.session_state.get('translation_metadata', {}).get(provider, {})
                    src_lang = metadata.get('source', '?')
                    tgt_lang = metadata.get('target', '?')
                    st.markdown(f"**{provider}** ({provider_cost}) `({src_lang} → {tgt_lang})`")

                    if translation.startswith("❌"):
                        st.error(translation)
                    else:
                        st.text_area(
                            f"{provider} result:",
                            value=translation,
                            height=text_height,
                            key=f"translation_llm_{provider}",
                            label_visibility="collapsed"
                        )
                        st.caption(f"📊 {len(translation)} characters")

            # Comparison metrics (compact for side-by-side)
            valid_translations = {k: v for k, v in st.session_state.translations.items()
                                 if not v.startswith("❌")}

            if len(valid_translations) > 1:
                st.divider()
                st.markdown("**📊 Comparison**")

                lengths = [len(t) for t in valid_translations.values()]
                traditional_count = sum(1 for k in valid_translations.keys()
                                      if PROVIDER_INFO[k].get("type", "traditional") == "traditional")
                llm_count = sum(1 for k in valid_translations.keys()
                               if PROVIDER_INFO[k].get("type") == "llm")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Length", f"{sum(lengths) / len(lengths):.0f}")
                    st.metric("Traditional", traditional_count)
                with col2:
                    st.metric("Range", f"{min(lengths)}-{max(lengths)}")
                    st.metric("LLM", llm_count)

        else:
            # Placeholder
            st.info("👈 Enter text and click 'Translate All' to see results here")


if __name__ == "__main__":
    main()
