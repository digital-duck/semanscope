# config.py
import streamlit as st 
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# App Constants
ST_APP_NAME = "Semanscope"

ST_ICON = "🧭"

SRC_DIR = Path(__file__).parent
# print(f"[DBG] src dir: {SRC_DIR}")

# Data directory path - centralized for easy refactoring
DATA_PATH = SRC_DIR.parent / "data"

# Cache directory path - centralized embedding cache (moved outside repo)
CACHE_PATH = Path.home() / 'projects' / 'embedding_cache' / 'semanscope'

# Semantic Affinity embedding cache - stored outside repo for persistence
# This cache is shared across all Semanscope pages (Semantic Affinity, Semantics Explorer, etc.)
# Format: {model_name: {lang_code: {word: embedding_vector}}}
# Using ~ will be expanded to home directory at runtime
SEMANTIC_CACHE_PATH = CACHE_PATH / "master_embeddings.pkl"


# Default Settings
DEFAULT_GEOMETRIC_ANALYSIS = False
DEFAULT_N_CLUSTERS = 5
DEFAULT_MIN_CLUSTERS = 2
DEFAULT_MAX_CLUSTERS = 10
DEFAULT_MAX_WORDS = 15

DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 1800  # Maximized height for optimal vertical space utilization


DEFAULT_DATASET = "NeurIPS-06-animal-gender-RA"  # "ACL-2-word-v2" # "ACL-6-Emoji" #   "ACL-1-Alphabets"
DEFAULT_METHOD = "PHATE"
DEFAULT_MODEL = "Qwen3-Embedding-0.6B"  # MTEB #5, proven baseline for geometric studies
DEFAULT_DIMENSION = "2D"
DEFAULT_CLUSTERING = True  # Enable geometric analysis and clustering by default
DEFAULT_SPLIT_LINES = False  # Enable line splitting/preprocessing by default

# V2O Optimization Settings
DEFAULT_OPTIMIZATION_STRATEGY = "baseline"  # "baseline" or "v2o"
# After v2o verification, can change to "v2o" as default

OPTIMIZATION_STRATEGIES = {
    "baseline": {
        "name": "Baseline (CPU, no caching)",
        "description": "Standard model loading with CPU inference. No query-level caching.",
        "help": "Use this for reference benchmarks or if you don't have a GPU."
    },
    "v2o": {
        "name": "V2O Optimized (GPU + fp16 + cache)",
        "description": "Optimized with GPU acceleration, fp16 precision, and query result caching. 2-50× faster!",
        "help": "Recommended for production use. Requires GPU for maximum speedup (auto-falls back to CPU)."
    }
}


# Alternative options for research:
# "Llama-Embed-Nemotron-8B"  # MTEB #1 (high memory requirements)
# "EmbeddingGemma-300M"       # Geometric collapse subject

DEFAULT_INSTRUCTION_MSG_1 = "👈 Configure Input Dataset and Languages in the sidebar and click **Visualize** to start Manifold analysis"

# Cache Settings
DEFAULT_TTL = 8  # Default cache TTL in hours

# Enhanced Error Handling Settings
ENABLE_CHINESE_TEXT_PREPROCESSING = True  # Enable preprocessing for Chinese radicals
ENABLE_ROBUST_DIMENSION_REDUCTION = True  # Enable enhanced NaN handling in dimension reduction
SHOW_PREPROCESSING_WARNINGS = True  # Show detailed preprocessing warnings to users
RESEARCH_MODE = True  # Preserve method failures for research insights (no automatic fallbacks)


# Language Code Mapping
# Maps language display names to their corresponding file suffixes
LANGUAGE_CODE_MAP = {
    "Chinese": "chn",
    "English": "enu",
    "French": "fra",
    "Spanish": "spa",
    "German": "deu",
    "Arabic": "ara",
    "Hebrew": "heb",
    "Hindi": "hin",
    "Japanese": "jpn",
    "Korean": "kor",
    "Russian": "rus",
    "Thai": "tha",
    "Greek": "grk",
    "Persian": "fas",
    "Turkish": "tur",
    "Georgian": "kat",
    "Armenian": "hye",
    "Vietnamese": "vie"
}

# Language code to model language mapping for embedding models
# Maps our internal language codes to the language codes expected by multilingual models
LANG_TO_MODEL_CODE_MAP = {
    "chn": "zh",  # Chinese -> Simplified Chinese
    "enu": "en",  # English -> English
    "fra": "fr",  # French -> French
    "spa": "es",  # Spanish -> Spanish
    "deu": "de",  # German -> German
    "ara": "ar",  # Arabic -> Arabic
    "heb": "he",  # Hebrew -> Hebrew
    "hin": "hi",  # Hindi -> Hindi
    "jpn": "ja",  # Japanese -> Japanese
    "kor": "ko",  # Korean -> Korean
    "rus": "ru",  # Russian -> Russian
    "tha": "th",  # Thai -> Thai
    "grk": "el",  # Greek -> Greek
    "fas": "fa",  # Persian -> Persian
    "tur": "tr",  # Turkish -> Turkish
    "kat": "ka",  # Georgian -> Georgian
    "hye": "hy",  # Armenian -> Armenian
    "vie": "vi"   # Vietnamese -> Vietnamese
}

# Reverse mapping for efficient lookups
LANGUAGE_CODE_TO_NAME_MAP = {v: k for k, v in LANGUAGE_CODE_MAP.items()}


# Default language preferences - simplified configuration
DEFAULT_LANG_SET = ["chn", "kor", "jpn", "vie", "enu", "ara"]
DEFAULT_LANG = "enu"  # Default fallback for model language

# Helper functions for language code operations
def get_language_name_from_code(code: str) -> str:
    """Get language display name from language code"""
    return LANGUAGE_CODE_TO_NAME_MAP.get(code, "Unknown")

def get_language_code_from_name(name: str) -> str:
    """Get language code from language display name"""
    return LANGUAGE_CODE_MAP.get(name, DEFAULT_LANG)

def get_all_language_codes() -> list:
    """Get list of all supported language codes"""
    return list(LANGUAGE_CODE_MAP.values())

def get_all_language_names() -> list:
    """Get list of all supported language names"""
    return list(LANGUAGE_CODE_MAP.keys())

def get_sorted_language_names() -> list:
    """Get alphabetically sorted list of language names"""
    return sorted(list(LANGUAGE_CODE_MAP.keys()))

def get_language_codes_with_prefix(prefix: str = "-") -> list:
    """Get language codes with specified prefix (useful for filename filtering)"""
    return [f"{prefix}{code}" for code in LANGUAGE_CODE_MAP.values()]

def get_model_language_code(lang_code: str) -> str:
    """Get model language code from internal language code"""
    return LANG_TO_MODEL_CODE_MAP.get(lang_code, "en")  # Default to English

def get_user_default_languages():
    """Get user's configured default languages from session state, fall back to config default"""
    if hasattr(st, 'session_state') and 'global_settings' in st.session_state:
        if 'languages' in st.session_state.global_settings:
            return st.session_state.global_settings['languages']['default_languages']
    return DEFAULT_LANG_SET


# Semantic Domain Color Coding Configuration (Case-Insensitive)
# This configuration supports ACL paper Figure 2 requirements for morphological network signatures
SEMANTIC_DOMAIN_COLORS = {
    # ========================================
    # MAJOR DOMAIN CATEGORIES
    # ========================================
    # These provide fallback colors when no dataset-specific colors are available

    # Language identification (for multilingual datasets)
    'chinese': '#FF0000',        # Red
    'english': '#0000FF',        # Blue
    'french': '#008000',         # Green
    'spanish': '#FFA500',        # Orange
    'german': '#800080',         # Purple
    'arabic': '#A52A2A',         # Brown

    # Core semantic categories
    'people': '#EB1532',         # Crimson - human agents and social roles
    'objects': '#2C2720',        # Dark brown - physical objects and tools
    'places': '#20B2AA',         # Light sea green - locations and environments
    'nature': '#228B22',         # Forest green - natural world and phenomena
    'abstract': '#DA70D6',       # Orchid - concepts and abstract ideas
    'activity': '#FF69B4',       # Hot pink - actions and processes
    'time': '#8A2BE2',           # Blue violet - temporal concepts
    'body': '#DC143C',           # Deep red - body parts and anatomy
    'food': '#CD853F',           # Peru - sustenance and nutrition

    # Linguistic structure
    'morphology': '#9400D3',     # Dark violet - word formation and roots
    'grammar': '#FF6347',        # Tomato - grammatical elements
    'numbers': '#44FF44',        # Bright green - numerical concepts

    # Mathematical Domains
    'algebra': '#4169E1',           # Royal blue - variables, equations, functions
    'geometry': '#0000FF',          # Blue - spatial concepts, shapes, angles
    'calculus': '#1E90FF',          # Dodger blue - derivatives, integrals, limits
    'statistics': '#87CEEB',        # Sky blue - probability, distributions
    'logic': '#6495ED',             # Cornflower blue - logical concepts

    # Physics - Mechanics
    'mechanics': '#228B22',         # Forest green - force, mass, energy
    'thermodynamics': '#32CD32',    # Lime green - heat, temperature, entropy
    'waves': '#00FF00',             # Bright green - wave, frequency, amplitude

    # Physics - Fields and Particles
    'electromagnetism': '#8A2BE2',  # Blue violet - charge, field, current
    'quantum': '#9400D3',           # Dark violet - quantum, particle, uncertainty
    'relativity': '#9932CC',        # Dark orchid - spacetime, relativistic
    'nuclear': '#BA55D3',           # Medium orchid - fusion, fission, decay

    # Scientific Method and Analysis
    'scientific_method': '#FF69B4', # Hot pink - theory, hypothesis, experiment
    'measurement': '#FFB6C1',       # Light pink - data, observation, units

    # Legacy scientific domains (for backward compatibility)
    'mathematics': '#FF1493',       # Deep pink - general mathematics
    'physics': "#10DF10",           # Dark slate gray - general physics
    'biology': "#1047DF",           # Dark slate gray - general physics

    # ========================================
    # ALPHABETS & WRITING SYSTEMS (ATOMIC-LEVEL ANALYSIS)
    # ========================================
    # Optimized color scheme for "Hunting Linguistic Quarks" project
    # High contrast colors for cross-script geometric pattern analysis

    # === CONFIRMED SUB-ATOMIC PARTICLE SYSTEMS ===
    # Chinese Radicals (CONFIRMED collapse patterns)
    'radicals_pure': '#8B0000',        # Dark red - Pure structural radicals (sub-atomic particles)
    'radicals_1d': '#FF4500',          # Orange red - 1D compositional radicals
    'elemental_chars': '#FF0000',      # Red - Meaningful elemental characters (2D)

    # === HIGH-PRIORITY SUB-ATOMIC CANDIDATES ===
    # Arabic Script (trilateral root patterns)
    'arabic_letters': '#FF1493',       # Deep pink - Arabic letters ا-ي
    'arabic_trilateral_roots': '#8B008B', # Dark magenta - Trilateral root candidates (ك-ت-ب)

    # Korean Script (jamo components)
    'korean_letters': '#4B0082',       # Indigo - Korean Hangul jamo (ㄱ,ㄴ,ㄷ,ㅏ,ㅓ,ㅗ)
    'korean_consonants': '#301934',    # Very dark purple - Consonant jamo candidates
    'korean_vowels': '#6A5ACD',        # Slate blue - Vowel jamo candidates

    # Hebrew Script (consonantal roots)
    'hebrew_letters': '#8B4513',       # Saddle brown - Hebrew letters א-ת
    'hebrew_roots': '#A0522D',         # Sienna - Consonantal root candidates

    # Sanskrit/Hindi (dhatu roots)
    'hindi_letters': "#14120F",        # Dark orange - Hindi Devanagari letters
    'sanskrit_dhatus': "#101296",      # Peru - Dhatu root candidates

    # === ATOMIC-LEVEL SYSTEMS (NO SUB-ATOMIC EXPECTED) ===
    # Latin Script Variants
    'latin_uppercase': '#FF0000',      # Bright red - A-Z uppercase (English baseline)
    'latin_lowercase': '#FF6347',      # Tomato red - a-z lowercase

    # French Script
    'french_uppercase': '#006633',     # Dark green - A-Z uppercase French
    'french_lowercase': '#00AA55',     # Medium green - a-z lowercase French
    'french_accented': '#004422',      # Dark forest green - À,é,ç etc.

    # Spanish Script
    'spanish_uppercase': "#5D43D3",    # Orange red - A-Z uppercase Spanish
    'spanish_lowercase': "#85690F",    # Tomato - a-z lowercase Spanish
    'spanish_special': "#384BA0",      # Dark red - Ñ,ñ,á,é etc.

    # German Script
    'german_uppercase': "#0A0733",     # Indigo - A-Z uppercase German
    'german_lowercase': "#2610B9",     # Slate blue - a-z lowercase German
    'german_umlauts': '#2E0054',       # Dark purple - Ä,ö,ü,ß

    # Russian Script (Cyrillic)
    'russian_letters': '#8A2BE2',      # Blue violet - Russian Cyrillic letters

    # === SYLLABIC SYSTEMS ===
    # Japanese Scripts
    'hiragana_letters': '#228B22',     # Forest green - Japanese Hiragana (あ,い,う)
    'katakana_letters': '#32CD32',     # Lime green - Japanese Katakana (ア,イ,ウ)
    'kanji_radicals': '#008000',       # Green - Japanese Kanji radicals

    # Thai Script
    'thai_letters': "#2B0BB9",         # Crimson - Thai letters

    # === ADDITIONAL WRITING SYSTEMS ===
    # Greek Alphabet (classical reference)
    'greek_uppercase': '#0000FF',      # Bright blue - Α-Ω uppercase letters
    'greek_lowercase': '#4169E1',      # Royal blue - α-ω lowercase letters

    # Persian Script (Arabic script variant)
    'persian_letters': '#00FF00',      # Bright Green - Creates brown when overlapping with red Arabic

    # Turkish Script (Latin variant with special letters)
    'turkish_uppercase': "#25C04C",    # Dark orange - Turkish Latin uppercase
    'turkish_lowercase': "#0C4117",    # Orange - Turkish Latin lowercase
    'turkish_special': "#BC59D4",      # Orange red - Turkish-specific letters (ç,ğ,ı,ş,ü,ö)

    # Georgian Script (unique Kartvelian system)
    'georgian_letters': '#2E8B57',     # Sea green - Georgian letters (ა,ბ,გ,დ,ე)

    # Armenian Script (unique Indo-European system)
    'armenian_uppercase': '#4682B4',   # Steel blue - Armenian uppercase (Ա,Բ,Գ,Դ,Ե)
    'armenian_lowercase': '#5F9EA0',   # Cadet blue - Armenian lowercase (ա,բ,գ,դ,ե)

    # Vietnamese Script (Latin with extensive diacritics)
    'vietnamese_letters': '#00CED1',   # Dark turquoise - Vietnamese letters with tone markers (distinct from all other scripts)

    # Semantic Words (3D+ Vocabulary)
    'semantic_words': '#FFD700',    # Gold - Full semantic vocabulary across languages
    'composite_chars': '#ADFF2F',   # Green yellow - Individual characters from compound words

    # Emoji Domains (Modern Pictographic System)
    'emoji_nature': '#228B22',      # Forest green - Natural elements (sun, moon, water, fire)
    'emoji_body': '#CD853F',        # Peru - Body parts and anatomy
    'emoji_people': '#4169E1',      # Royal blue - Human figures and roles
    'emoji_animals': '#D2691E',     # Chocolate - Animal representations
    'emoji_food': '#FF6347',        # Tomato - Food and sustenance
    'emoji_objects': '#708090',     # Slate gray - Tools and artifacts
    'emoji_transport': '#4682B4',   # Steel blue - Vehicles and movement
    'emoji_clothing': '#9932CC',    # Dark orchid - Garments and accessories
    'emoji_symbols': '#DAA520',     # Goldenrod - Abstract symbols and concepts
    'emoji_emotions': '#FF69B4',    # Hot pink - Facial expressions and feelings
    'emoji_punctuation': '#2F4F4F', # Dark slate gray - Punctuation marks
    'emoji_arrows': '#696969',      # Dim gray - Directional indicators
    'emoji_numbers': '#FF4500',     # Orange red - Numeric representations
    'emoji_letters': '#6495ED',     # Cornflower blue - Letter symbols
    'emoji_time': '#8A2BE2',        # Blue violet - Temporal indicators
    'emoji_colors': '#DC143C',      # Crimson - Color representations
    'emoji_elements': '#B22222',    # Fire brick - Elemental forces
    'emoji_growth': '#32CD32',      # Lime green - Growth and life
    'emoji_social': '#20B2AA',      # Light sea green - Social interactions
    'emoji_celebration': '#FFD700', # Gold - Festive and joyful
    'emoji_achievement': '#FF8C00', # Dark orange - Success and recognition
    'emoji_gesture': '#BA55D3',     # Medium orchid - Hand gestures
    'emoji_mystical': '#483D8B',    # Dark slate blue - Spiritual symbols
    'emoji_death': '#2F2F2F',       # Very dark gray - Mortality themes

    # Fallback colors
    'unknown': '#CCCCCC',
    'default': '#666666'
}

# Custom domains for specialized analysis (can be edited directly in this file)
CUSTOM_SEMANTIC_DOMAINS = {
    # NSM Prime Groups Color Mapping - Natural Semantic Metalanguage Categories
    # Core Substantives - Blue family (foundational concepts)
    'substantives': '#1E3A8A',                    # Deep blue - core entities (I, you, someone, something)
    'relational_substantives': '#3B82F6',        # Bright blue - relational concepts (kind, part)

    # Mental & Cognitive - Purple family (mind and thought)
    'mental_predicates': '#7C3AED',              # Violet - mental processes (think, know, want, feel)
    'speech': '#A855F7',                         # Purple - communication (say, words, true)

    # Physical Actions & Events - Red/Orange family (dynamic concepts)
    'actions_events_movement': '#DC2626',        # Red - actions and movement (do, happen, move, touch)
    'life_death': '#EA580C',                     # Orange-red - life processes (live, die)

    # Space & Time - Green family (dimensional concepts)
    'space': '#059669',                          # Green - spatial concepts (where, here, above, below)
    'time': '#10B981',                           # Light green - temporal concepts (when, now, before, after)

    # Properties & Evaluation - Yellow/Amber family (qualitative concepts)
    'descriptors': '#F59E0B',                    # Amber - size/property (big, small)
    'evaluators': '#EAB308',                     # Yellow - good/bad evaluation
    'intensifier_augmentor': '#FCD34D',          # Light yellow - degree (very, more)

    # Logical & Abstract - Gray/Blue family (structural concepts)
    'logical_concepts': '#6B7280',               # Gray - logic (not, maybe, can, because, if)
    'determiners': '#4B5563',                    # Dark gray - determiners (this, the same, other)
    'quantifiers': '#9CA3AF',                    # Light gray - quantities (one, two, some, all, much, many)
    'similarity': '#64748B',                     # Blue-gray - like concepts (like, as, way)

    # Existence & Location - Cyan family (being and location)
    'location_existence': '#0891B2',             # Cyan - existence (be, there is, exist)

    # Control & Meta - Pink family (meta-linguistic)
    'control_word': '#DB2777',                   # Pink - meta-linguistic control words

    # Physics-Inspired Groups - Alternative Classification Framework
    # Object Description Groups (Blue spectrum)
    'entity': "#202EE6",                         # Deep Blue - fundamental entities and subjects/objects
    'identity': "#1E2AD8",                       # Indigo - identifiers and classification concepts

    # Property Description Groups (Warm spectrum)
    'quantifier': "#048C91",                     # Orange - numbers and measurable quantities
    'qualifier': "#1A8855",                      # Amber - properties and descriptive attributes

    # State Transition Groups (Energy spectrum)
    'action': "#E20808",                         # Red - state transitions and dynamic processes
    'relation': "#550101",                       # Purple - interactions and connections between entities

    # World-View Coordinates (unchanged - use existing space/time)
    'spatial': '#16A34A',                        # Green - spatial location (using existing)
    'temporal': "#03420C",                         # Teal - temporal concepts (using existing)

    # Causality Framework
    'causality': "#D20DEC",                # Gray - logical operators and causal relationships

    'word':"#E20808", 
    'phrase':"#202EE6",  
    'sentence':"#1A8855",


    # Miscellaneous
    'misc': "#252933",                           # Pink - other control words and uncategorized concepts

}

# Plot Settings with unified color mapping
PLOT_CONFIG = {
    "width": DEFAULT_WIDTH,
    "height": DEFAULT_HEIGHT,
}

PLOT_WIDTH, PLOT_HEIGHT = PLOT_CONFIG["width"], PLOT_CONFIG["height"]

# Unified Color Mapping Functions (Case-Insensitive)
def get_all_domain_colors(dataset_name=None):
    """Get all domain colors combining predefined, custom, and dataset-specific domains (case-insensitive keys)"""
    all_colors = {}

    # Add predefined colors with lowercase keys
    for domain, color in SEMANTIC_DOMAIN_COLORS.items():
        all_colors[domain.lower()] = color

    # Add custom colors with lowercase keys (these override predefined if there are conflicts)
    for domain, color in CUSTOM_SEMANTIC_DOMAINS.items():
        all_colors[domain.lower()] = color

    # Add dataset-specific colors (highest priority - these override everything else)
    if dataset_name:
        dataset_colors = load_dataset_color_mapping(dataset_name)
        for domain, color in dataset_colors.items():
            all_colors[domain.lower()] = color

    return all_colors

def load_dataset_color_mapping(dataset_name):
    """Load dataset-specific color mapping from .color-code.csv file

    Supports both formats:
    - Legacy: domain,color,description
    - Enhanced: domain,color,color_name,description
    """
    color_file = DATA_PATH / "input" / f"{dataset_name}.color-code.csv"

    if not color_file.exists():
        return {}

    try:
        import pandas as pd
        df = pd.read_csv(color_file)

        # Create domain -> color mapping
        color_mapping = {}
        for _, row in df.iterrows():
            if 'domain' in row and ('color_hex' in row or 'color' in row):
                domain = str(row['domain']).strip().lower()
                # Try color_hex first, then fallback to color
                color = str(row.get('color_hex', row.get('color', ''))).strip()
                if domain and color:
                    color_mapping[domain] = color

        return color_mapping
    except Exception as e:
        # Silently return empty dict if file can't be loaded
        return {}

def get_domain_color(domain_name, dataset_name=None):
    """Get color for a specific semantic domain (case-insensitive)

    Args:
        domain_name: The semantic domain name
        dataset_name: Optional dataset name to load dataset-specific colors
    """
    if not domain_name:
        return SEMANTIC_DOMAIN_COLORS.get('default', '#666666')

    domain_lower = domain_name.lower()

    # First, check dataset-specific colors if dataset is provided
    if dataset_name:
        dataset_colors = load_dataset_color_mapping(dataset_name)
        if domain_lower in dataset_colors:
            return dataset_colors[domain_lower]

    # Check custom domains (they override predefined)
    for custom_domain, color in CUSTOM_SEMANTIC_DOMAINS.items():
        if custom_domain.lower() == domain_lower:
            return color

    # Then check predefined domains
    for predefined_domain, color in SEMANTIC_DOMAIN_COLORS.items():
        if predefined_domain.lower() == domain_lower:
            return color

    # Return default if not found
    return SEMANTIC_DOMAIN_COLORS.get('default', '#666666')

# Backward compatibility: Legacy COLOR_MAP for existing code
COLOR_MAP = {
    "chinese": get_domain_color("chinese"),
    "english": get_domain_color("english"),
    "french": get_domain_color("french"),
    "spanish": get_domain_color("spanish"),
    "german": get_domain_color("german"),
    "arabic": get_domain_color("arabic"),
    "vietnamese": get_domain_color("vietnamese_letters"),
    "japanese": get_domain_color("hiragana_letters"),
    "korean": get_domain_color("korean_letters"),
    "chn": get_domain_color("chinese"),
    "enu": get_domain_color("english"),
    "vie": get_domain_color("vietnamese_letters"),
    "jpn": get_domain_color("hiragana_letters"),
    "kor": get_domain_color("korean_letters")
}

# Publication Settings Configuration
# Centralized defaults for publication-quality visualizations
PUBLICATION_SETTINGS = {
    # Publication mode defaults (high-quality settings)
    'publication': {
        'textfont_size': 12,
        'point_size': 8,
        'plot_width': 1400,
        'plot_height': 1100,
        'export_format': 'PDF',
        'export_dpi': 300
    },
    # Standard mode defaults (interactive use)
    'standard': {
        'textfont_size': 12,
        'point_size': 8,
        'plot_width': 1000,
        'plot_height': 1100,
        'export_format': 'PNG',
        'export_dpi': 150
    },
    # Fallback defaults (when settings not available)
    'fallback': {
        'publication_mode': False,
        'textfont_size': 16,
        'point_size': 12,
        'plot_width': 1000,
        'plot_height': 1100,
        'export_format': 'PNG',
        'export_dpi': 150
    },
    # UI constraints and options
    'constraints': {
        'textfont_size': {'min': 6, 'max': 16, 'step': 1},
        'point_size': {'min': 4, 'max': 12, 'step': 1},
        'plot_width': {'min': 800, 'max': 1600, 'step': 50},
        'plot_height': {'min': 600, 'max': 1600, 'step': 50},
        'export_dpi': {'min': 150, 'max': 600, 'step': 50}
    },
    # Export format options
    'export_formats': ['PNG', 'SVG', 'PDF']
}

# Helper functions for publication settings
def get_publication_settings(mode: str = 'standard') -> dict:
    """Get publication settings for specified mode (publication, standard, or fallback)"""
    return PUBLICATION_SETTINGS.get(mode, PUBLICATION_SETTINGS['fallback'])

def get_publication_constraints(setting_name: str) -> dict:
    """Get UI constraints for a specific publication setting"""
    return PUBLICATION_SETTINGS['constraints'].get(setting_name, {})

def get_export_formats() -> list:
    """Get list of supported export formats"""
    return PUBLICATION_SETTINGS['export_formats']

# Sample Data
SAMPLE_DATA = {
    "chinese": "",
    "english": "",

}


sample_chn_input_data = SAMPLE_DATA["chinese"]
sample_enu_input_data = SAMPLE_DATA["english"]


# Cache Settings
CACHE_CONFIG = {
    "ttl": 3600,  # Time to live for cached data (in seconds)
    "max_entries": 100
}

# Logging Configuration
# Centralized logging settings for consistent debugging and monitoring
LOGGING_CONFIG = {
    # Log levels (using standard Python logging levels)
    'levels': {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    },
    # Default settings
    'default_level': 'INFO',
    'default_format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',

    # Component-specific log levels (can be customized per component)
    'component_levels': {
        'embedding_viz': 'INFO',
        'dimension_reduction': 'INFO',
        'geometric_analysis': 'INFO',
        'plotting_echarts': 'INFO',
        'global_settings': 'WARNING',
        'config': 'WARNING'
    },

    # Feature flags for debugging
    'enable_performance_logging': False,
    'enable_cache_logging': False,
    'enable_model_loading_logs': True,
    'enable_streamlit_logs': False,

    # Log output settings
    'log_to_console': True,
    'log_to_file': False,
    'log_file_path': 'logs/semanscope.log',
    'max_log_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Helper functions for logging configuration
def get_log_level(component_name: str = None) -> str:
    """Get log level for a specific component or default"""
    if component_name and component_name in LOGGING_CONFIG['component_levels']:
        return LOGGING_CONFIG['component_levels'][component_name]
    return LOGGING_CONFIG['default_level']

def get_log_format() -> str:
    """Get standard log format string"""
    return LOGGING_CONFIG['default_format']

def get_date_format() -> str:
    """Get standard date format for logs"""
    return LOGGING_CONFIG['date_format']

def is_logging_enabled(feature: str) -> bool:
    """Check if specific logging feature is enabled"""
    feature_key = f'enable_{feature}_logging'
    return LOGGING_CONFIG.get(feature_key, False)

def get_logging_config() -> dict:
    """Get complete logging configuration"""
    return LOGGING_CONFIG

# Add Ollama model information to MODEL_INFO
OLLAMA_MODELS = {
    "BGE-M3 (Ollama)": {
        "path": "bge-m3",
        "help": "BGE-M3 is a new model from BAAI distinguished for its versatility in Multi-Functionality, Multi-Linguality, and Multi-Granularity.",
        "is_active": True
    },
    "Paraphrase-Multilingual (Ollama)": {
        "path": "paraphrase-multilingual",
        "help": "Sentence-transformers model (multilingual) that can be used for tasks like clustering or semantic search.",
        "is_active": False
    },
    "Snowflake-Arctic-Embed2 (Ollama)": {
        "path": "snowflake-arctic-embed2",
        "help": "Snowflake Arctic model through Ollama offering efficient embedding generation with strong multilingual capabilities, especially for Chinese-English pairs.",
        "is_active": True
    },
    "EmbeddingGemma (Ollama)": {
        "path": "embeddinggemma",
        "help": "Google's EmbeddingGemma - 300M parameter state-of-the-art embedding model built from Gemma 3 with T5Gemma initialization. Trained on 100+ languages using Gemini model technology. Optimized for search, retrieval, classification, clustering, and semantic similarity tasks.",
        "is_active": False
    },
    "Nomic-Embed-Text (Ollama)": {
        "path": "nomic-embed-text",
        "help": "Nomic's v1.5 embedding model through Ollama. First truly open embedding model with multimodal capabilities and strong accuracy (86.2%). Optimized for semantic text embeddings.",
        "is_active": True
    },
    "Qwen3-Embedding-0.6B (Ollama)": {
        "path": "qwen3-embedding:0.6b",
        "help": "Qwen3 0.6B embedding model through Ollama - shows excellent metrics in EmbeddingGemma benchmarks. Part of MTEB #1 multilingual series. Updated 27 minutes ago! Perfect for geosemantic comparison with EmbeddingGemma.",
        "is_active": False
    },
    "Qwen3-Embedding-4B (Ollama)": {
        "path": "qwen3-embedding:4b",
        "help": "Qwen3 4B embedding model through Ollama - larger variant with enhanced capabilities. Part of MTEB #1 multilingual series (8B variant scores 70.58). Excellent for studying parameter scaling effects on semantic geometry.",
        "is_active": True
    },
    "Mistral (Ollama)": {
        "path": "mistral",
        "help": "Mistral model through Ollama offering efficient embedding generation with good multilingual capabilities.",
        "is_active": False
    },
    "Neural-Chat (Ollama)": {
        "path": "neural-chat",
        "help": "Neural Chat model through Ollama, optimized for conversational and semantic understanding tasks.",
        "is_active": False
    },
}

# Model information (name, Hugging Face path, and help text)
# Updated with 2025 MTEB leaderboard top performers for geosemantic analysis
#
# RESEARCH CONTEXT: Geometric Collapse Investigation
# This configuration supports the ACL paper "When Benchmarks Deceive: Geometric
# Collapse in State-of-the-Art Multilingual Embeddings" comparing MTEB top performers
# with focus on EmbeddingGemma-300M vs Qwen3/Nemotron models for semantic structure preservation
DEFAULT_TOP3_MODELS = [
    "LaBSE",
    "Sentence-BERT Multilingual",
    "Multilingual-E5-Large-Instruct-v2"
]
MODEL_INFO = {
    # === 2025 MTEB LEADERBOARD TOP 5 (99% ZERO-SHOT PERFORMANCE) ===
    # Added for geometric collapse investigation vs EmbeddingGemma-300M research
    "Llama-Embed-Nemotron-8B": {
        "path": "nvidia/llama-3.2-nemo-embed-8b",
        "help": "NVIDIA's Llama-3.2-based embedding model. 8B parameters, current MTEB leaderboard #1 with 99% zero-shot performance. Part of Nemotron family, optimized for high-quality embeddings across domains. Critical comparison model for EmbeddingGemma geometric collapse study.",
        "is_active": False,
        "alias": "Llama-8B",
        "mteb_rank": 1,
        "zero_shot_score": "99%",
        "memory_usage": "28629 MB",
        "parameters": "7B+",
        "embedding_dim": 4096
    },
    "Qwen3-Embedding-8B": {
        "path": "Qwen/Qwen3-Embedding-8B",
        "help": "Qwen3's largest embedding model. 8B parameters, MTEB leaderboard #2 with 99% zero-shot performance. Flagship model of the Qwen3 embedding series with enhanced cross-lingual capabilities and semantic structure preservation. Essential for parameter scaling analysis in geometric collapse research.",
        "is_active": False,
        "alias": "Qwen3-8B",
        "mteb_rank": 2,
        "zero_shot_score": "99%",
        "memory_usage": "28866 MB",
        "parameters": "7B+",
        "embedding_dim": 4096
    },
    "Gemini-Embedding-001": {
        "path": "google/gemini-embedding-001",  # API-only model
        "help": "Google's Gemini Embedding model. MTEB leaderboard #3 with 99% zero-shot performance. Key model for investigating whether Google's newer embedding architectures exhibit similar geometric collapse as EmbeddingGemma-300M. Critical for comparative analysis of Google's embedding evolution.",
        "is_active": False,  # Set to False - requires API access, not available via Hugging Face
        "alias": "Gemini-001",
        "mteb_rank": 3,
        "zero_shot_score": "99%",
        "memory_usage": "API-based",
        "parameters": "Unknown",
        "embedding_dim": 3072,
        "note": "Requires Google AI Studio API access - not available via Hugging Face download",
        "error": "API-only model, not available for direct download"
    },

    # === 2025 MTEB LEADERS: TIER 1 PERFORMANCE ===
    "Stella-400M": {
        "path": "dunzhang/stella_en_400M_v5",
        "help": "Current MTEB retrieval leaderboard leader for commercial use. 400M parameters, state-of-the-art architecture optimization by Dun Zhang. Top choice for geosemantic geometric analysis.",
        "is_active": False,
        "note": "too big and slow"
    },
    "Stella-1.5B": {
        "path": "dunzhang/stella_en_1.5B_v5",
        "help": "Larger variant of Stella model (1.5B parameters). Minimal accuracy gain over 400M version but useful for studying parameter scaling effects on semantic geometry.",
        "is_active": False,
        "error": """Error in get_embeddings: The repository dunzhang/stella_en_400M_v5 contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co/dunzhang/stella_en_400M_v5 . You can inspect the repository content at https://hf.co/dunzhang/stella_en_400M_v5. Please pass the argument trust_remote_code=True to allow custom code to be run.
        """,
    },
    "Jina-Embeddings-v3": {
        "path": "jinaai/jina-embeddings-v3",
        "help": "Best multilingual model (89 languages), 2nd on MTEB English leaderboard. 570M params, XLM-RoBERTa + task-specific LoRA adapters. Perfect for cross-lingual geosemantic studies.",
        "is_active": False,
        "alias": "Jina-v3",
        "error": """Error in get_embeddings: jinaai/xlm-roberta-flash-implementation You can inspect the repository content at https://hf.co/jinaai/jina-embeddings-v3. Please pass the argument trust_remote_code=True to allow custom code to be run.
        """
    },
    "BGE-M3": {
        "path": "BAAI/bge-m3",
        "help": "Highest retrieval accuracy (72%) in comparative studies. Multi-functionality, multi-linguality, multi-granularity. Excellent baseline for geometric structure analysis.",
        "is_active": False,
        "error": """Error in get_embeddings: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors. See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
"""
    },
    # === 2025 MTEB LEADERS: TIER 2 ARCHITECTURAL INNOVATION ===
    "Nomic-Embed-Text-v2": {
        "path": "nomic-ai/nomic-embed-text-v2",
        "help": "First MoE (Mixture-of-Experts) architecture for embeddings. Strong accuracy (86.2%) with novel approach. Critical for studying MoE vs standard transformer geometry.",
        "is_active": False,
        "alias": "Nomic-v2",
        "error": """Error in get_embeddings: nomic-ai/nomic-embed-text-v2 is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models' If this is a private repository, make sure to pass a token having permission to this repo either by logging in with hf auth login or by passing token=<your_token>
        """,
    },
    "GTE-Multilingual-Base": {
        "path": "Alibaba-NLP/gte-multilingual-base",
        "help": "MTEB 2025 leader. 305M params, 70+ languages, 10x faster inference. Encoder-only transformer optimized for efficiency while preserving semantic structure.",
        "is_active": False,
        "alias": "GTE-Base",
        "error": """Error in get_embeddings: Alibaba-NLP/new-impl You can inspect the repository content at https://hf.co/Alibaba-NLP/gte-multilingual-base. Please pass the argument trust_remote_code=True to allow custom code to be run.
        """,
    },
    "E5-Base-v2": {
        "path": "intfloat/e5-base-v2",
        "help": "Balanced accuracy-speed trade-off (83-85% accuracy, 79-82ms latency). Strong performer without prefix prompts, ideal for studying geometric consistency.",
        "is_active": False,
        "note": "very good",
        "warning": "Known to produce NaN values with Chinese text. Use Sentence-BERT Multilingual or BGE-M3 for Chinese-English datasets."
    },
    "Qwen3-Embedding-0.6B": {
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "help": "Qwen3 embedding model showing excellent metrics in EmbeddingGemma benchmark comparisons. 600M parameters, part of MTEB #1 multilingual series (8B variant scores 70.58). Supports 100+ languages with strong cross-lingual capabilities.",
        "alias": "Qwen3-06B",
        "is_active": True
    },
    "Qwen3-Embedding-4B": {
        "path": "Qwen/Qwen3-Embedding-4B",
        "help": "Qwen3 4B embedding model - larger variant with enhanced performance. Part of the MTEB #1 multilingual series with state-of-the-art cross-lingual capabilities. Ideal for studying parameter scaling effects on semantic geometry preservation.",
        "alias": "Qwen3-4B",
        "is_active": False
    },

    # === GOOGLE GEMMA EMBEDDING MODELS ===
    "EmbeddingGemma-300M": {
        "path": "google/embeddinggemma-300m",
        "help": "Google's state-of-the-art EmbeddingGemma model. 308M parameters, best-in-class for its size on MTEB benchmark. Built from Gemma 3 with 100+ languages support, optimized for on-device AI with 768-dimensional embeddings (MRL support for 512/256/128). RESEARCH FOCUS: Subject of geometric collapse investigation in Chinese poetry compositional semantics.",
        "is_active": True,
        "alias": "Gemma-300M",
        "research_status": "geometric_collapse_subject",
        "paper_reference": "ACL submission: 'When Benchmarks Deceive: Geometric Collapse in State-of-the-Art Multilingual Embeddings'"
    },

    # === LEGACY TOP PERFORMERS (ESTABLISHED BASELINES) ===
    "BGE-Multilingual-Gemma2": {
        "path": "BAAI/bge-multilingual-gemma2",
        "help": "SOTA on C-MTEB Chinese benchmark. Based on Gemma-2-9B, specialized for Chinese-Japanese-Korean-English. Excellent cross-lingual capabilities.",
        "is_active": False,
        "alias": "BGE-Gemma2",
        "error": """too big""",
    },
    "Jina-Embeddings-v2-ZH": {
        "path": "jinaai/jina-embeddings-v2-base-zh",
        "help": "Chinese-English bilingual specialist. 570M params, 8192 tokens, JinaBERT architecture. No Chinese-English bias, perfect for mixed input.",
        "is_active": False,
        "alias": "Jina-v2-ZH",
        "error": """Error in get_embeddings: You are using sdpa as attention type. However, non-absolute positional embeddings can not work with them. Please load the model with attn_implementation="eager".
        """,
    },
    "BGE-Base-ZH-v1.5": {
        "path": "BAAI/bge-base-zh-v1.5",
        "help": "Chinese-optimized BGE model with high C-MTEB performance. Specialized for character-level Chinese semantics and traditional character analysis.",
        "alias": "BGE-ZH-v1.5",
        "is_active": False,
        "error": """Error in get_embeddings: 
Due to a serious vulnerability issue in torch.load, even with weights_only=True, 
we now require users to upgrade torch to at least v2.6 in order to use the function. 
This version restriction does not apply when loading files with safetensors. 
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
"""
    },
    
    # === FOUNDATIONAL MODELS (ESTABLISHED RESEARCH BASELINES) ===
    "XLM-RoBERTa-v2": {
        "path": "xlm-roberta-base",
        "help": "A robust multilingual model trained on 100+ languages. Great for cross-lingual tasks like text classification and NER. NOW WITH L2 NORMALIZATION.",
        "alias": "XLM-R-v2",
        "is_active": True,
        "note": "Fixed: Now includes L2 normalization to prevent embedding collapse"
    },
    "mBERT": {
        "path": "bert-base-multilingual-cased",
        "help": "Multilingual BERT trained on 104 languages. Widely used for cross-lingual transfer learning.",
        "is_active": True
    },
    "LaBSE": {
        "path": "sentence-transformers/LaBSE",
        "help": "Language-agnostic BERT sentence embeddings for 109 languages. Excellent for sentence similarity and paraphrase detection.",
        "is_active": True
    },
    "DistilBERT Multilingual": {
        "path": "distilbert-base-multilingual-cased",
        "help": "A lightweight version of mBERT. Faster and more efficient, suitable for real-time applications.",
        "is_active": True,
        "alias": "DistilBERT",
    },
    "XLM": {
        "path": "xlm-mlm-100-1280",
        "help": "Cross-lingual language model trained using masked and translation language modeling. Good for translation tasks.",
        "is_active": False,
        "error": """Error in get_embeddings: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors. See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
        """
    },
    "InfoXLM": {
        "path": "microsoft/infoxlm-base",
        "help": "An extension of XLM-R with improved cross-lingual transferability. Great for low-resource languages.",
        "is_active": False,
        "error": """Error in get_embeddings: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors. See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
        """,
    },
    "Sentence-BERT Multilingual": {
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "help": "Proven baseline from cross-lingual research. MPNet architecture, 50+ languages. Excellent for semantic similarity tasks.",
        "alias": "Sentence-BERT",
        "is_active": True
    },
    "Multilingual-E5-Large-Instruct-v2": {
        "path": "intfloat/multilingual-e5-large-instruct",
        "help": "Instruction-tuned multilingual embedding model with 560M parameters. Supports 100+ languages with 1024-dimensional embeddings. Optimized for retrieval and semantic similarity with instruction-following capabilities. NOW WITH CORRECT 'query:' PREFIX.",
        "alias": "E5-Instruct-v2",
        "is_active": True,
        "parameters": "560M",
        "embedding_dim": 1024,
        "languages": 100,
        "max_tokens": 512,
        "note": "Fixed: Now properly adds 'query:' prefix for instruction-following"
    },
    # === OPENAI EMBEDDING MODELS (REFERENCE ONLY - REQUIRES API KEY) ===
    # Note: These models require OpenAI API access and are not directly usable in this app
    # Added for completeness and reference for other researchers
    "OpenAI text-embedding-ada-002": {
        "path": "text-embedding-ada-002",  # API model path
        "help": "OpenAI's 2nd generation embedding model (Dec 2022). 1536 dimensions. Industry standard that replaced 5 separate models with 99.8% cost reduction. Requires OpenAI API key.",
        "alias": "OpenAI-ada-002",
        "is_active": False
    },
    "OpenAI text-embedding-3-small": {
        "path": "text-embedding-3-small",  # API model path
        "help": "OpenAI's 3rd generation small model (Jan 2024). 1536 dimensions, 5x cheaper than ada-002. 44.0% vs 31.4% on MIRACL benchmark. Requires OpenAI API key.",
        "alias": "OpenAI-3-samll",
        "is_active": False
    },
    "OpenAI text-embedding-3-large": {
        "path": "text-embedding-3-large",  # API model path
        "help": "OpenAI's best performing 3rd generation model (Jan 2024). 3072 dimensions. SOTA performance: 54.9% on MIRACL, 64.6% on MTEB. Requires OpenAI API key.",
        "alias": "OpenAI-3-large",
        "is_active": False
    },

    # === ADDITIONAL CHINESE-FOCUSED MODELS ===
    "Universal-Sentence-Encoder-Multilingual": {
        "path": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "help": "Google's multilingual sentence encoder. Strong performance across languages with efficient inference.",
        "is_active": True,
        "alias": "U-Encoder",
        "note": "not good, highly degenerate"
    },
    
    # === EXPERIMENTAL/RESEARCH MODELS ===
    # "ModernBERT": {
    #     "path": "answerdotai/ModernBERT-base",
    #     "help": "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference."
    # },
    # "Erlangshen": {
    #     "path": "IDEA-CCNL/Erlangshen-Roberta-110M", 
    #     "help": "A Chinese-focused multilingual model optimized for Chinese-English tasks like translation and sentiment analysis."
    # }
}


# Update MODEL_INFO dictionary
MODEL_INFO.update(OLLAMA_MODELS)

# Add OpenRouter models
try:
    from semanscope.models.openrouter_model import OPENROUTER_MODELS
    MODEL_INFO.update(OPENROUTER_MODELS)
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    st.warning("OpenRouter integration not available. Install requests package to use OpenRouter API models.")

# Add Google Cloud Gemini models
try:
    from semanscope.models.gemini_model import GEMINI_MODELS
    MODEL_INFO.update(GEMINI_MODELS)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Gemini integration not available. Install google-generativeai package to use Google Cloud models.")

# Add Voyage AI models
try:
    from semanscope.models.voyage_model import VOYAGE_MODELS
    MODEL_INFO.update(VOYAGE_MODELS)
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False
    st.warning("Voyage AI integration not available. Install voyageai package to use Voyage AI models.")

# Helper function to get only active models
def get_active_models():
    """Get only active models from MODEL_INFO with grouped organization"""
    # Grouped model organization for better UX
    grouped_models = {
        "🤗 HuggingFace Models": [
            "Sentence-BERT Multilingual",                      # Proven baseline
            "Multilingual-E5-Large-Instruct",                 # Instruction-tuned multilingual
            "Qwen3-Embedding-0.6B",                            # Local baseline
            "Qwen3-Embedding-4B",                              # Mid-scale local
            "Qwen3-Embedding-8B",                              # Large scale local
            "Llama-Embed-Nemotron-8B",                         # MTEB #1
            "EmbeddingGemma-300M"                              # Collapse subject
        ],
        "🌐 OpenRouter Models": [
            # Qwen Series (Large to Small)
            "Qwen3-Embedding-8B (OpenRouter)",                 # 🚀 Large scale via API
            "Qwen3-Embedding-4B (OpenRouter)",                 # ⚡ Mid-scale via API
            "Qwen3-Embedding-0.6B (OpenRouter)",               # 📊 Baseline via API
            # Google Models
            "Gemini-Embedding-001 (OpenRouter)",               # 🔥 Google SOTA
            # OpenAI Models
            "OpenAI Text-Embedding-3-Large (OpenRouter)",      # 🚀 OpenAI SOTA
            "OpenAI Text-Embedding-3-Small (OpenRouter)",      # ⚡ OpenAI efficient
            "OpenAI Text-Embedding-Ada-002 (OpenRouter)",      # 🤖 OpenAI proven
            # Other Providers
            # "Mistral-Embed-2312 (OpenRouter)",                 # 🌟 European AI
            # "Voyage-Large-2-Instruct (OpenRouter)"             # 🌊 Instruction-tuned
        ],
        "🌊 Voyage AI Models": [
            # Voyage 3 Series (Latest)
            "Voyage-3 (Voyage AI)",                            # 🌊 Flagship model
            "Voyage-3-Lite (Voyage AI)",                       # ⚡ Efficient variant
            # Voyage Large 2 Series
            "Voyage-Large-2-Instruct (Voyage AI)",             # 🎯 Instruction-tuned
            "Voyage-Large-2 (Voyage AI)",                      # 📊 Baseline
            # Voyage Multilingual
            "Voyage-Multilingual-2 (Voyage AI)",               # 🌍 Multilingual specialist
        ],
        "🦙 Ollama Models": [
            # Local Ollama models (if any are active)
        ]
    }

    # Get all active models
    active_models = {name: info for name, info in MODEL_INFO.items()
                    if info.get('is_active', True)}  # Default to True for backward compatibility

    # Add Ollama models to their group
    for model_name in active_models:
        if model_name in OLLAMA_MODELS:
            grouped_models["🦙 Ollama Models"].append(model_name)

    # Create final ordered dict with grouped organization
    final_models = {}

    # Add models in group order
    for group_name, model_list in grouped_models.items():
        # Skip empty groups
        if not model_list:
            continue

        # Add models from this group (if active)
        for model_name in model_list:
            if model_name in active_models:
                final_models[model_name] = active_models[model_name]

    # Add any remaining active models not in groups
    for model_name, model_info in active_models.items():
        if model_name not in final_models:
            final_models[model_name] = model_info

    return final_models

def get_active_models_by_architecture():
    """Get active models organized by architecture (BERT, Hybrid, LLM) with alphabetical sorting within each group"""
    # Architecture-based organization
    architecture_groups = {
        "🧬 BERT-based Models": [
            # Classic BERT family models
            "Sentence-BERT Multilingual",                      # MPNet (BERT family)
            "XLM-RoBERTa-v2",                                  # RoBERTa (BERT variant)
            "mBERT",                                           # Original multilingual BERT
            "LaBSE",                                           # BERT-based sentence embeddings
            "DistilBERT Multilingual",                         # Distilled BERT
            "Universal-Sentence-Encoder-Multilingual",         # Google USE (encoder-based)
        ],
        "🔀 Hybrid Models": [
            # Instruction-tuned and enhanced architectures
            "Multilingual-E5-Large-Instruct-v2",              # Instruction-tuned E5
            "Nomic-Embed-Text (Ollama)",                       # MoE architecture
            "BGE-M3 (Ollama)",                                 # Multi-functionality hybrid
            "Snowflake-Arctic-Embed2 (Ollama)",               # Arctic architecture
        ],
        "🤖 LLM-based Models": [
            # Large Language Model embeddings
            "Llama-Embed-Nemotron-8B",                         # Llama-3.2 based (MTEB #1)
            "Qwen3-Embedding-8B",                              # Qwen3 8B
            "Qwen3-Embedding-4B",                              # Qwen3 4B
            "Qwen3-Embedding-0.6B",                            # Qwen3 0.6B
            "Qwen3-Embedding-4B (Ollama)",                     # Qwen3 via Ollama
            "EmbeddingGemma-300M",                             # Gemma-based
            "Qwen3-Embedding-8B (OpenRouter)",                 # Qwen3 via API
            "Qwen3-Embedding-4B (OpenRouter)",                 # Qwen3 via API
            "Qwen3-Embedding-0.6B (OpenRouter)",               # Qwen3 via API
            "Gemini-Embedding-001 (OpenRouter)",               # Gemini API
            "OpenAI Text-Embedding-3-Large (OpenRouter)",      # OpenAI GPT-based
            "OpenAI Text-Embedding-3-Small (OpenRouter)",      # OpenAI GPT-based
            "OpenAI Text-Embedding-Ada-002 (OpenRouter)",      # OpenAI GPT-based
            "Voyage-3 (Voyage AI)",                            # Voyage AI flagship
            "Voyage-3-Lite (Voyage AI)",                       # Voyage AI lite
            "Voyage-Large-2-Instruct (Voyage AI)",             # Voyage AI instruct
            "Voyage-Large-2 (Voyage AI)",                      # Voyage AI baseline
            "Voyage-Multilingual-2 (Voyage AI)",               # Voyage AI multilingual
        ]
    }

    # Get all active models
    active_models = {name: info for name, info in MODEL_INFO.items()
                    if info.get('is_active', True)}

    # Create final ordered dict with architecture-based organization and alphabetical sorting
    final_models = {}

    # Add models in architecture group order, sorted alphabetically within each group
    for group_name, model_list in architecture_groups.items():
        # Collect active models from this group
        group_active_models = [name for name in model_list if name in active_models]

        # Sort alphabetically within the group
        group_active_models.sort()

        # Add to final dict
        for model_name in group_active_models:
            final_models[model_name] = active_models[model_name]

    # Add any remaining active models not in architecture groups (sorted alphabetically)
    remaining_models = [name for name in active_models.keys() if name not in final_models]
    remaining_models.sort()

    for model_name in remaining_models:
        final_models[model_name] = active_models[model_name]

    return final_models

def get_active_models_with_headers():
    """Get active models with visual group headers for dropdown display"""
    # Architecture-based organization (same as above)
    architecture_groups = {
        "🧬 BERT-based Models": [
            "Sentence-BERT Multilingual",
            "XLM-RoBERTa-v2",
            "mBERT",
            "LaBSE",
            "DistilBERT Multilingual",
            "Universal-Sentence-Encoder-Multilingual",
        ],
        "🔀 Hybrid Models": [
            "Multilingual-E5-Large-Instruct-v2",
            "Nomic-Embed-Text (Ollama)",
            "BGE-M3 (Ollama)",
            "Snowflake-Arctic-Embed2 (Ollama)",
        ],
        "🤖 LLM-based Models": [
            "Llama-Embed-Nemotron-8B",
            "Qwen3-Embedding-8B",
            "Qwen3-Embedding-4B",
            "Qwen3-Embedding-0.6B",
            "Qwen3-Embedding-4B (Ollama)",
            "EmbeddingGemma-300M",
            "Qwen3-Embedding-8B (OpenRouter)",
            "Qwen3-Embedding-4B (OpenRouter)",
            "Qwen3-Embedding-0.6B (OpenRouter)",
            "Gemini-Embedding-001 (OpenRouter)",
            "OpenAI Text-Embedding-3-Large (OpenRouter)",
            "OpenAI Text-Embedding-3-Small (OpenRouter)",
            "OpenAI Text-Embedding-Ada-002 (OpenRouter)",
            "Voyage-3 (Voyage AI)",
            "Voyage-3-Lite (Voyage AI)",
            "Voyage-Large-2-Instruct (Voyage AI)",
            "Voyage-Large-2 (Voyage AI)",
            "Voyage-Multilingual-2 (Voyage AI)",
        ]
    }

    # Get all active models
    active_models = {name: info for name, info in MODEL_INFO.items()
                    if info.get('is_active', True)}

    # Create list with group headers
    model_list_with_headers = []
    model_info_dict = {}

    # Add models in architecture group order with headers
    for group_name, model_list in architecture_groups.items():
        # Collect active models from this group
        group_active_models = [name for name in model_list if name in active_models]

        # Sort alphabetically within the group
        group_active_models.sort()

        # Add group header
        if group_active_models:
            header = f"━━━━━━ {group_name} ━━━━━━"
            model_list_with_headers.append(header)

            # Add models from this group
            for model_name in group_active_models:
                model_list_with_headers.append(model_name)
                model_info_dict[model_name] = active_models[model_name]

    # Add any remaining active models not in architecture groups
    remaining_models = [name for name in active_models.keys() if name not in model_info_dict]
    if remaining_models:
        remaining_models.sort()
        model_list_with_headers.append("━━━━━━ Other Models ━━━━━━")
        for model_name in remaining_models:
            model_list_with_headers.append(model_name)
            model_info_dict[model_name] = active_models[model_name]

    return model_list_with_headers, model_info_dict

# Helper function to get only active methods
def get_active_methods_config():
    """Get only active methods from METHOD_INFO"""
    return {name: info for name, info in METHOD_INFO.items()
            if info.get('is_active', True)}  # Default to True for backward compatibility


# Dimensionality reduction method with help text
METHOD_INFO = {
    "t-SNE": {
        "help": "t-Distributed Stochastic Neighbor Embedding. Preserves local structure and is great for visualizing clusters."
    },
    "Isomap": {
        "help": "Isometric Mapping. Preserves geodesic distances and is ideal for capturing manifold structures."
    },
    "UMAP": {
        "help": "Uniform Manifold Approximation and Projection. Fast and preserves both local and global structure."
    },
    "LLE": {
        "help": "Locally Linear Embedding. Preserves local relationships by representing points as linear combinations of neighbors."
    },
    "MDS": {
        "help": "Multidimensional Scaling. Preserves pairwise distances between points, suitable for global structure visualization."
    },
    "PCA": {
        "help": "Principal Component Analysis. A linear method that projects data onto directions of maximum variance."
    },
    "Kernel PCA": {
        "help": "Kernel PCA. A nonlinear extension of PCA that uses kernel functions to capture complex structures.",
        "alias": "K-PCA",
    },
    "Spectral Embedding": {
        "help": "Spectral Embedding. Based on graph Laplacian, effective for capturing underlying data structure.",
        "alias": "Spectral",
    },
    f"PHATE": {
        "help": "Potential of Heat-diffusion for Affinity-based Transition Embedding. Great for visualizing complex, high-dimensional data."
    },
    "TriMap": {
        "help": "TriMap - Superior balance of local and global structure preservation. Perfect for morphological family clustering and cross-lingual comparisons.",
        "is_active": True
    },
    "PaCMAP": {
        "help": "PaCMAP - Exceptional global structure preservation for cross-lingual analysis. Ideal for comparing morphological signatures across languages.",
        "is_active": True
    },
    "ForceAtlas2": {
        "help": "ForceAtlas2 - Designed specifically for network visualization. Perfect for morphological networks (子-network, Haus/Arbeit families).",
        "is_active": True
    },
    # "Autoencoders": {
    #     "help": "Neural network-based approach for learning compressed representations of data."
    # },
    # "LDA": {
    #     "help": "Linear Discriminant Analysis. A supervised method that maximizes class separability."
    # }
}



DATASET_INFO = {
    'ACL-0-Radicals-Kangxi': {'alias': 'Radicals-Kangxi',
        'help': 'ACL-level-0-Radicals-Kangxi',
        'is_active': True,
        'note': ''},
    'ACL-0-Zinets': {'alias': 'Zinets',
        'help': 'ACL-0-Zinets',
        'is_active': True,
        'note': ''},
    'ACL-0-Zinets-All': {'alias': 'Zinets-All',
        'help': 'ACL-0-Zinets-All',
        'is_active': True,
        'note': ''},
    'ACL-0-Zinets-Pure': {'alias': 'Zinets-Pure',
        'help': 'ACL-0-Zinets-Pure',
        'is_active': True,
        'note': ''},
    'ACL-1-Alphabets': {'alias': 'Alphabets',
        'help': 'ACL-1-Alphabets',
        'is_active': True,
        'note': ''},
    'ACL-1-Alphabets-words': {'alias': 'Alphabets-words',
        'help': 'ACL-1-Alphabets-words',
        'is_active': True,
        'note': ''},
    'ACL-2-PeterG-ALL': {'alias': 'PeterG-ALL',
        'help': 'ACL-2-PeterG-ALL',
        'is_active': True,
        'note': ''},
    'ACL-2-PeterG-Adj': {'alias': 'PeterG-Adj',
        'help': 'ACL-2-PeterG-Adj',
        'is_active': True,
        'note': ''},
    'ACL-2-PeterG-Noun': {'alias': 'PeterG-Noun',
        'help': 'ACL-2-PeterG-Noun',
        'is_active': True,
        'note': ''},
    'ACL-2-PeterG-Verb': {'alias': 'PeterG-Verb',
        'help': 'ACL-2-PeterG-Verb',
        'is_active': True,
        'note': ''},
    'ACL-2-word-v2': {'alias': 'word-v2',
        'help': 'ACL-2-word-v2',
        'is_active': True,
        'note': ''},
    'ACL-3-network-haus-arbeit-v2': {'alias': 'network-haus-arbeit-v2',
        'help': 'ACL-3-network-haus-arbeit-v2',
        'is_active': True,
        'note': ''},
    'ACL-3-network-work-light-v2': {'alias': 'network-work-light-v2',
        'help': 'ACL-3-network-work-light-v2',
        'is_active': True,
        'note': ''},
    'ACL-3-network-子-v2': {'alias': 'network-子-v2',
        'help': 'ACL-3-network-子-v2',
        'is_active': True,
        'note': ''},
    'ACL-4-numbers': {'alias': 'numbers',
        'help': 'ACL-4-numbers',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-du-fu-climb': {'alias': 'du-fu-climb',
        'help': 'ACL-5-poem-du-fu-climb',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-du-fu-night': {'alias': 'du-fu-night',
        'help': 'ACL-5-poem-du-fu-night',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-du-fu-quatrain': {'alias': 'du-fu-quatrain',
        'help': 'ACL-5-poem-du-fu-quatrain',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-du-fu-rain': {'alias': 'du-fu-rain',
        'help': 'ACL-5-poem-du-fu-rain',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-frost-road': {'alias': 'frost-road',
        'help': 'ACL-5-poem-frost-road',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-frost-woods': {'alias': 'frost-woods',
        'help': 'ACL-5-poem-frost-woods',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-li-bai-moonlight': {'alias': 'li-bai-moonlight',
        'help': 'ACL-5-poem-li-bai-moonlight',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-li-bai-waterfall': {'alias': 'li-bai-waterfall',
        'help': 'ACL-5-poem-li-bai-waterfall',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-shelley-moon': {'alias': 'shelley-moon',
        'help': 'ACL-5-poem-shelley-moon',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-wang-wei-lodge': {'alias': 'wang-wei-lodge',
        'help': 'ACL-5-poem-wang-wei-lodge',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-wang-wei-park': {'alias': 'wang-wei-park',
        'help': 'ACL-5-poem-wang-wei-park',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-wang-wei-sickness': {'alias': 'wang-wei-sickness',
        'help': 'ACL-5-poem-wang-wei-sickness',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-wang-wei-valley': {'alias': 'wang-wei-valley',
        'help': 'ACL-5-poem-wang-wei-valley',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-wordsworth-daffodils': {'alias': 'wordsworth-daffodils',
        'help': 'ACL-5-poem-wordsworth-daffodils',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-wordsworth-robin': {'alias': 'wordsworth-robin',
        'help': 'ACL-5-poem-wordsworth-robin',
        'is_active': True,
        'note': ''},
    'ACL-5-poem-wordsworth-strange': {'alias': 'wordsworth-strange',
        'help': 'ACL-5-poem-wordsworth-strange',
        'is_active': True,
        'note': ''},
    'ACL-6-Emoji': {'alias': 'Emoji',
        'help': 'ACL-6-Emoji',
        'is_active': True,
        'note': ''},
    'ACL-6-Pictograph': {'alias': 'Pictograph',
        'help': 'ACL-6-Pictograph',
        'is_active': True,
        'note': ''}
}

# Simulate login handling (for demonstration purposes)
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Check if API key is already set in environment variables
    if os.getenv("HF_API_KEY"):
        st.session_state.logged_in = True
        return

    if not st.session_state.logged_in:
        st.sidebar.title("Login")
        api_key = st.sidebar.text_input("Enter Hugging Face API Key", type="password")
        if st.sidebar.button("Login"):
            if api_key:  # Simulate API key validation
                os.environ["HF_API_KEY"] = api_key
                st.session_state.logged_in = True
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Please enter a valid API key.")
        st.stop()  # Stop execution if not logged in
