"""
UniTranslator - Unified Translation Interface
Supports both traditional translation APIs and LLM-based translation via OpenRouter
"""

import os
import requests
from typing import Optional, Dict, List, Tuple
from deep_translator import (
    GoogleTranslator,
    MyMemoryTranslator,
    LibreTranslator,
    DeeplTranslator,
    YandexTranslator,
    MicrosoftTranslator,
    PapagoTranslator,
    ChatGptTranslator,
)


class UniTranslator:
    """
    Unified translator supporting both traditional translation services
    and LLM-based translation via OpenRouter.ai

    Mimics deep-translator API design for consistency.
    """

    # Provider type classification
    TRADITIONAL_PROVIDERS = {
        "Google": GoogleTranslator,
        "MyMemory": MyMemoryTranslator,
        "Libre": LibreTranslator,
        "DeepL": DeeplTranslator,
        "Yandex": YandexTranslator,
        "Microsoft": MicrosoftTranslator,
        "Papago": PapagoTranslator,
        "ChatGPT": ChatGptTranslator,
    }

    # LLM providers via OpenRouter
    LLM_PROVIDERS = {
        "Gemini-Flash": "google/gemini-flash-1.5",
        "Gemini-Pro": "google/gemini-pro-1.5",
        "GPT-4o": "openai/gpt-4o",
        "GPT-4o-Mini": "openai/gpt-4o-mini",
        "Claude-Sonnet": "anthropic/claude-3.5-sonnet",
        "Claude-Haiku": "anthropic/claude-3.5-haiku",
        "Llama-3.3-70B": "meta-llama/llama-3.3-70b-instruct",
        "DeepSeek-V3": "deepseek/deepseek-chat",
        "Qwen-2.5-72B": "qwen/qwen-2.5-72b-instruct",
        "Mistral-Large": "mistralai/mistral-large",
    }

    def __init__(self, provider: str, source: str = "auto", target: str = "en"):
        """
        Initialize UniTranslator

        Args:
            provider: Provider name (from TRADITIONAL_PROVIDERS or LLM_PROVIDERS)
            source: Source language code
            target: Target language code
        """
        self.provider = provider
        self.source = source
        self.target = target
        self.translator = None
        self.is_llm = provider in self.LLM_PROVIDERS

        # Initialize the appropriate translator
        if self.is_llm:
            self._init_llm_translator()
        else:
            self._init_traditional_translator()

    def _normalize_language_code(self, lang_code: str, provider: str) -> str:
        """Normalize language codes for different providers"""
        # Handle auto-detect
        if lang_code == "auto":
            return "auto"

        # For Google Translate, keep exact case (zh-CN not zh-cn)
        if provider == "Google":
            # GoogleTranslator expects zh-CN, zh-TW (uppercase)
            # Most other codes are lowercase
            lang_map = {
                "zh": "zh-CN",  # Default simplified Chinese
            }
            return lang_map.get(lang_code, lang_code)

        # For other providers, return as-is
        return lang_code

    def _init_traditional_translator(self):
        """Initialize traditional translation service"""
        if self.provider not in self.TRADITIONAL_PROVIDERS:
            raise ValueError(f"Unknown traditional provider: {self.provider}")

        provider_class = self.TRADITIONAL_PROVIDERS[self.provider]

        # Normalize language codes
        source = self._normalize_language_code(self.source, self.provider)
        target = self._normalize_language_code(self.target, self.provider)

        print(f"[DEBUG] Initializing {self.provider}: source={source}, target={target}")

        # Handle different initialization patterns
        if self.provider == "Google":
            self.translator = provider_class(source=source, target=target)

        elif self.provider == "MyMemory":
            self.translator = provider_class(source=source, target=target)

        elif self.provider == "Libre":
            self.translator = provider_class(
                source=source,
                target=target,
                base_url="https://libretranslate.com"
            )

        elif self.provider == "DeepL":
            api_key = os.getenv("DEEPL_API_KEY")
            if not api_key:
                raise ValueError("DEEPL_API_KEY not found in environment")
            self.translator = provider_class(
                api_key=api_key,
                source=source,
                target=target
            )

        elif self.provider == "Yandex":
            api_key = os.getenv("YANDEX_API_KEY")
            if not api_key:
                raise ValueError("YANDEX_API_KEY not found in environment")
            self.translator = provider_class(
                api_key=api_key,
                source=source,
                target=target
            )

        elif self.provider == "Microsoft":
            api_key = os.getenv("MICROSOFT_TRANSLATOR_KEY")
            if not api_key:
                raise ValueError("MICROSOFT_TRANSLATOR_KEY not found in environment")
            self.translator = provider_class(
                api_key=api_key,
                target=target
            )

        elif self.provider == "Papago":
            client_id = os.getenv("PAPAGO_CLIENT_ID")
            client_secret = os.getenv("PAPAGO_CLIENT_SECRET")
            if not client_id or not client_secret:
                raise ValueError("PAPAGO_CLIENT_ID and PAPAGO_CLIENT_SECRET required")
            self.translator = provider_class(
                client_id=client_id,
                secret_key=client_secret,
                source=source,
                target=target
            )

        elif self.provider == "ChatGPT":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.translator = provider_class(
                api_key=api_key,
                target=target
            )

    def _init_llm_translator(self):
        """Initialize LLM-based translator via OpenRouter"""
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        if self.provider not in self.LLM_PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        self.model_id = self.LLM_PROVIDERS[self.provider]

    def _translate_with_llm(self, text: str) -> str:
        """Translate using LLM via OpenRouter API"""

        # Build translation prompt
        prompt = self._build_translation_prompt(text, self.source, self.target)

        # Call OpenRouter API
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,  # Lower temperature for more consistent translation
            "max_tokens": 2000,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            translated_text = result["choices"][0]["message"]["content"].strip()

            # Clean up LLM response (remove quotes, extra formatting)
            translated_text = self._clean_llm_output(translated_text)

            return translated_text

        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API error: {str(e)}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Invalid OpenRouter API response: {str(e)}")

    def _build_translation_prompt(self, text: str, source: str, target: str) -> str:
        """Build translation prompt for LLM"""

        # Language code to name mapping
        lang_names = {
            "en": "English",
            "zh": "Chinese",
            "zh-CN": "Simplified Chinese",
            "zh-TW": "Traditional Chinese",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "ar": "Arabic",
            "pt": "Portuguese",
            "it": "Italian",
            "auto": "automatically detected language"
        }

        source_name = lang_names.get(source, source)
        target_name = lang_names.get(target, target)

        prompt = f"""Translate the following text from {source_name} to {target_name}.

IMPORTANT INSTRUCTIONS:
- Provide ONLY the translation, no explanations or additional text
- Preserve the structure and formatting (line breaks, etc.)
- Maintain the meaning and tone of the original
- For technical or domain-specific terms, use appropriate terminology
- Do not add quotes or any formatting around the translation

Text to translate:
{text}

Translation:"""

        return prompt

    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM output to match traditional translator format"""
        # Remove common LLM artifacts
        text = text.strip()

        # Remove surrounding quotes if present
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]

        # Remove "Translation:" prefix if LLM added it
        if text.lower().startswith("translation:"):
            text = text[12:].strip()

        return text

    def translate(self, text: str, **kwargs) -> str:
        """
        Translate text using the configured provider

        Args:
            text: Text to translate
            **kwargs: Additional provider-specific arguments

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""

        try:
            print(f"[DEBUG] Translating with {self.provider}: '{text[:50]}...' ({self.source} -> {self.target})")

            if self.is_llm:
                result = self._translate_with_llm(text)
            else:
                result = self.translator.translate(text)

            print(f"[DEBUG] Translation result: '{result[:50]}...'")
            return result

        except Exception as e:
            print(f"[DEBUG] Translation error: {str(e)}")
            raise Exception(f"Translation failed with {self.provider}: {str(e)}")

    @classmethod
    def get_available_providers(cls) -> Dict[str, List[str]]:
        """Get all available providers categorized by type"""
        return {
            "traditional": list(cls.TRADITIONAL_PROVIDERS.keys()),
            "llm": list(cls.LLM_PROVIDERS.keys())
        }

    @classmethod
    def get_provider_info(cls) -> Dict[str, Dict]:
        """Get detailed information about all providers"""

        info = {
            # Traditional providers
            "Google": {
                "type": "traditional",
                "requires_key": False,
                "env_var": None,
                "description": "Google Translate - Fast, reliable, 100+ languages",
                "website": "https://translate.google.com/",
                "strengths": "Excellent general-purpose translation, vast language support, free",
                "limitations": "May have rate limits for heavy usage",
                "cost": "Free",
            },
            "MyMemory": {
                "type": "traditional",
                "requires_key": False,
                "env_var": None,
                "description": "MyMemory - World's largest translation memory",
                "website": "https://mymemory.translated.net/",
                "strengths": "Translation memory for consistency, handles technical content well",
                "limitations": "1000 words/day limit for free tier",
                "cost": "Free (limited)",
            },
            "Libre": {
                "type": "traditional",
                "requires_key": False,
                "env_var": None,
                "description": "LibreTranslate - Open-source, privacy-focused",
                "website": "https://libretranslate.com/",
                "strengths": "Privacy-focused, open-source, self-hostable",
                "limitations": "Smaller language support, quality varies",
                "cost": "Free",
            },
            "DeepL": {
                "type": "traditional",
                "requires_key": True,
                "env_var": "DEEPL_API_KEY",
                "description": "DeepL - Premium neural translation",
                "website": "https://www.deepl.com/pro-api",
                "strengths": "Highest quality for European languages, excellent nuance handling",
                "limitations": "Requires API key, limited free tier",
                "cost": "Free tier: 500K chars/month",
            },
            "Yandex": {
                "type": "traditional",
                "requires_key": True,
                "env_var": "YANDEX_API_KEY",
                "description": "Yandex - Strong for Slavic languages",
                "website": "https://cloud.yandex.com/en/services/translate",
                "strengths": "Excellent for Russian/Slavic languages, good pricing",
                "limitations": "Requires API key",
                "cost": "Paid service",
            },
            "Microsoft": {
                "type": "traditional",
                "requires_key": True,
                "env_var": "MICROSOFT_TRANSLATOR_KEY",
                "description": "Microsoft Translator - Enterprise-grade",
                "website": "https://azure.microsoft.com/services/cognitive-services/translator/",
                "strengths": "Enterprise-grade reliability, document translation support",
                "limitations": "Requires Azure account",
                "cost": "Free tier: 2M chars/month",
            },
            "Papago": {
                "type": "traditional",
                "requires_key": True,
                "env_var": "PAPAGO_CLIENT_ID",
                "description": "Papago - Best for Asian languages",
                "website": "https://developers.naver.com/products/papago/",
                "strengths": "Best for Korean, excellent for Japanese/Chinese",
                "limitations": "Limited language pairs, requires Naver account",
                "cost": "Free tier available",
            },

            # LLM providers
            "Gemini-Flash": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "Google Gemini Flash 1.5 - Fast LLM translation",
                "website": "https://openrouter.ai/",
                "strengths": "Context-aware, fast, good for technical content, understands nuance",
                "limitations": "Costs per token, may be verbose",
                "cost": "$0.10 / 1M tokens (via OpenRouter)",
            },
            "Gemini-Pro": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "Google Gemini Pro 1.5 - High-quality LLM translation",
                "website": "https://openrouter.ai/",
                "strengths": "Excellent context understanding, handles idioms well, cultural awareness",
                "limitations": "Higher cost, may be slower",
                "cost": "$1.25 / 1M tokens (via OpenRouter)",
            },
            "GPT-4o": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "OpenAI GPT-4o - Advanced reasoning for translation",
                "website": "https://openrouter.ai/",
                "strengths": "Best-in-class context understanding, excellent for creative/literary text",
                "limitations": "Higher cost, can be verbose",
                "cost": "$2.50 / 1M tokens (via OpenRouter)",
            },
            "GPT-4o-Mini": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "OpenAI GPT-4o Mini - Cost-effective LLM translation",
                "website": "https://openrouter.ai/",
                "strengths": "Good balance of quality and cost, fast response",
                "limitations": "Slightly lower quality than GPT-4o",
                "cost": "$0.15 / 1M tokens (via OpenRouter)",
            },
            "Claude-Sonnet": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "Anthropic Claude 3.5 Sonnet - Balanced LLM translation",
                "website": "https://openrouter.ai/",
                "strengths": "Excellent balance of quality and speed, good for technical content",
                "limitations": "Costs per token",
                "cost": "$3.00 / 1M tokens (via OpenRouter)",
            },
            "Claude-Haiku": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "Anthropic Claude 3.5 Haiku - Fast, economical LLM",
                "website": "https://openrouter.ai/",
                "strengths": "Very fast, cost-effective, good for simple translations",
                "limitations": "May lack nuance for complex texts",
                "cost": "$0.80 / 1M tokens (via OpenRouter)",
            },
            "Llama-3.3-70B": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "Meta Llama 3.3 70B - Open-source LLM translation",
                "website": "https://openrouter.ai/",
                "strengths": "Strong open-source alternative, good multilingual support",
                "limitations": "May be less nuanced than commercial models",
                "cost": "$0.80 / 1M tokens (via OpenRouter)",
            },
            "DeepSeek-V3": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "DeepSeek V3 - Advanced Chinese LLM",
                "website": "https://openrouter.ai/",
                "strengths": "Excellent for Chinese translations, strong reasoning",
                "limitations": "Newer model, less battle-tested",
                "cost": "$0.27 / 1M tokens (via OpenRouter)",
            },
            "Qwen-2.5-72B": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "Alibaba Qwen 2.5 72B - Chinese-focused LLM",
                "website": "https://openrouter.ai/",
                "strengths": "Excellent for Chinese-English translation, culturally aware",
                "limitations": "May favor Chinese linguistic patterns",
                "cost": "$0.35 / 1M tokens (via OpenRouter)",
            },
            "Mistral-Large": {
                "type": "llm",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "description": "Mistral Large - European-focused LLM",
                "website": "https://openrouter.ai/",
                "strengths": "Strong for European languages, good reasoning",
                "limitations": "Higher cost",
                "cost": "$3.00 / 1M tokens (via OpenRouter)",
            },
        }

        return info
