#!/usr/bin/env python3
"""
Clear session state and cache to force reload of semantic data
"""

def clear_streamlit_caches():
    """Clear Streamlit session state entries related to semantic colors"""
    try:
        import streamlit as st

        # Keys to clear related to semantic color mapping
        keys_to_clear = []

        # Find all semantic color keys
        for key in list(st.session_state.keys()):
            if 'semantic_colors' in key:
                keys_to_clear.append(key)
            elif '_text_area' in key:
                keys_to_clear.append(key)
            elif 'include_checkbox' in key:
                keys_to_clear.append(key)

        print(f"🧹 Clearing {len(keys_to_clear)} session state keys:")
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                print(f"   ✅ Cleared: {key}")

        # Clear any cached data
        st.cache_data.clear()
        print("🗑️ Cleared cache_data")

        print("✅ Session state and cache cleared!")

    except ImportError:
        print("❌ This script needs to be run within Streamlit context")
        print("💡 Instead, you can manually refresh the page or use the 🔄 refresh button")

if __name__ == "__main__":
    clear_streamlit_caches()