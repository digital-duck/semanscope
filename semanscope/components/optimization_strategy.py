"""
Optimization Strategy Widget for Semanscope

Provides a global optimization strategy selector that controls whether
baseline or v2o optimized models are used.

Usage:
    from components.optimization_strategy import render_optimization_strategy_selector

    # In sidebar
    with st.sidebar:
        render_optimization_strategy_selector()

    # Later in code
    from models.model_manager import get_model_with_strategy
    model = get_model_with_strategy(model_name)  # Automatically uses strategy from session state
"""

import streamlit as st
from semanscope.config import DEFAULT_OPTIMIZATION_STRATEGY, OPTIMIZATION_STRATEGIES


def render_optimization_strategy_selector(
    location: str = "sidebar",
    expanded: bool = False,
    key_suffix: str = ""
):
    """
    Render optimization strategy selector

    Args:
        location: "sidebar" or "main" - where to render
        expanded: Whether expander starts expanded
        key_suffix: Optional suffix for widget keys (for multiple instances)

    Returns:
        Selected strategy ("baseline" or "v2o")
    """
    # Initialize session state if not present
    if 'optimization_strategy' not in st.session_state:
        st.session_state.optimization_strategy = DEFAULT_OPTIMIZATION_STRATEGY

    # Render in expander
    with st.expander("⚡ Optimization Strategy", expanded=expanded):
        st.markdown("""
        Control embedding model performance optimization.
        """)

        # Radio button for strategy selection
        strategy = st.radio(
            "Select Strategy",
            options=["baseline", "v2o"],
            format_func=lambda x: OPTIMIZATION_STRATEGIES[x]["name"],
            index=0 if st.session_state.optimization_strategy == "baseline" else 1,
            help="Choose between baseline (CPU, reference) or v2o (GPU + caching, 2-50× faster)",
            key=f"optimization_strategy_radio{key_suffix}"
        )

        # Update session state
        st.session_state.optimization_strategy = strategy

        # Show description and help
        strategy_info = OPTIMIZATION_STRATEGIES[strategy]

        if strategy == "baseline":
            st.info(f"**{strategy_info['name']}**\n\n{strategy_info['description']}")
            st.caption(strategy_info['help'])
        else:
            st.success(f"**{strategy_info['name']}**\n\n{strategy_info['description']}")
            st.caption(strategy_info['help'])

            # GPU status check (if v2o selected)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    st.success(f"✅ GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
                else:
                    st.warning("⚠️ No GPU detected. V2O will run on CPU (caching benefits only)")
            except ImportError:
                st.info("ℹ️ PyTorch not available for GPU detection")

        # Show performance comparison
        if strategy == "v2o":
            with st.expander("📊 Expected Performance Gains", expanded=False):
                st.markdown("""
                **Cold Start (First Query):**
                - HuggingFace (GPU): 2-5× faster
                - Ollama: 10-25× faster

                **Warm Cache (Repeated Query):**
                - All models: 10-100× faster

                **Overall:**
                - Typical 50-word dataset: 200-500ms → 40-100ms (GPU)
                - Ollama: 2.5-5s → 100-300ms
                """)

    return strategy


def get_current_optimization_strategy() -> str:
    """
    Get current optimization strategy from session state

    Returns:
        "baseline" or "v2o"
    """
    return st.session_state.get('optimization_strategy', DEFAULT_OPTIMIZATION_STRATEGY)


def set_optimization_strategy(strategy: str):
    """
    Set optimization strategy programmatically

    Args:
        strategy: "baseline" or "v2o"
    """
    if strategy not in ["baseline", "v2o"]:
        raise ValueError(f"Invalid strategy: {strategy}. Must be 'baseline' or 'v2o'")

    st.session_state.optimization_strategy = strategy
