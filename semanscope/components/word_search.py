"""
Shared Word Search component for Semanscope pages
Provides consistent word search functionality across different visualization types
"""

import streamlit as st

class WordSearchManager:
    """Manages word search functionality for semantic visualizations"""

    def __init__(self):
        self.default_color_palette = [
            "#FFD700",  # Gold
            "#FF6B6B",  # Red
            "#4ECDC4",  # Teal
            "#45B7D1",  # Blue
            "#96CEB4",  # Green
            "#FECA57",  # Yellow
            "#FF9FF3",  # Pink
            "#54A0FF",  # Light Blue
        ]

    def get_word_search_settings(self, key_prefix="main"):
        """Get word search settings from UI with unique keys"""
        with st.sidebar.expander("🔍 Word Search", expanded=False):
            st.markdown("### Search and highlight specific words in the visualization")

            # Enable/disable highlighting
            enable_highlight = st.checkbox(
                "Enable Word Search",
                value=False,
                key=f"{key_prefix}_enable_word_search",
                help="Search and highlight specific words with custom colors and sizes"
            )

            if not enable_highlight:
                return None

            # Word search input
            keywords_input = st.text_area(
                "Search Words (one per line)",
                placeholder="1\n-1\n0.01\n2.7182\n3.141592",
                height=150,
                key=f"{key_prefix}_search_keywords",
                help="Enter words to search and highlight, one per line"
            )

            if not keywords_input.strip():
                st.warning("Please enter at least one word to search")
                return None

            keywords = [kw.strip() for kw in keywords_input.strip().split('\n') if kw.strip()]

            # Color and size settings
            col1, col2 = st.columns(2)
            with col1:
                highlight_color = st.color_picker(
                    "Search Result Color",
                    "#FFD700",
                    key=f"{key_prefix}_search_color",
                    help="Color for highlighted search results"
                )
            with col2:
                highlight_size = st.slider(
                    "Search Result Size",
                    10, 50, 20,
                    key=f"{key_prefix}_search_size",
                    help="Point size for highlighted search results"
                )

            # Font size for highlighted labels (if applicable to visualization type)
            highlight_font_size = st.slider(
                "Label Font Size",
                10, 30, 16,
                key=f"{key_prefix}_search_font_size",
                help="Font size for highlighted search result labels"
            )

            # Multiple keyword groups option
            use_multiple_colors = st.checkbox(
                "Use different colors for each search word",
                value=False,
                key=f"{key_prefix}_use_multiple_colors"
            )

            if use_multiple_colors:
                st.markdown("**Predefined color palette for search words:**")
                st.caption(f"First {len(self.default_color_palette)} keywords will use different colors automatically")

            return {
                'keywords': keywords,
                'color': highlight_color,
                'size': highlight_size,
                'font_size': highlight_font_size,
                'use_multiple_colors': use_multiple_colors,
                'color_palette': self.default_color_palette if use_multiple_colors else None
            }

    def find_matching_indices(self, labels, search_config):
        """Find indices of labels that match search keywords"""
        if not search_config:
            return []

        keywords = search_config['keywords']
        matching_indices = []

        for i, label in enumerate(labels):
            label_str = str(label).strip()
            for keyword in keywords:
                # Exact match (case-insensitive)
                if label_str.lower() == keyword.lower():
                    matching_indices.append(i)
                    break
                # Partial match for longer labels
                elif keyword.lower() in label_str.lower():
                    matching_indices.append(i)
                    break

        return matching_indices

    def get_search_colors(self, labels, search_config):
        """Get colors for search results based on configuration"""
        if not search_config:
            return []

        keywords = search_config['keywords']
        use_multiple_colors = search_config['use_multiple_colors']
        base_color = search_config['color']
        color_palette = search_config['color_palette']

        search_colors = []

        for label in labels:
            label_str = str(label).strip()
            found_keyword_index = None

            # Find which keyword matches this label
            for keyword_idx, keyword in enumerate(keywords):
                if label_str.lower() == keyword.lower() or keyword.lower() in label_str.lower():
                    found_keyword_index = keyword_idx
                    break

            if found_keyword_index is not None:
                if use_multiple_colors and color_palette:
                    # Use different color for each keyword
                    color_idx = found_keyword_index % len(color_palette)
                    search_colors.append(color_palette[color_idx])
                else:
                    # Use single color for all matches
                    search_colors.append(base_color)
            else:
                # No match found
                search_colors.append(None)

        return search_colors

    def apply_search_highlighting_plotly(self, fig, labels, search_config):
        """Apply search highlighting to a Plotly figure"""
        if not search_config:
            return fig

        matching_indices = self.find_matching_indices(labels, search_config)
        if not matching_indices:
            return fig

        # Update marker properties for matching points
        search_colors = self.get_search_colors(labels, search_config)
        search_size = search_config['size']

        # Create arrays for colors and sizes
        marker_colors = []
        marker_sizes = []

        for i, _ in enumerate(labels):
            if i in matching_indices:
                color_idx = matching_indices.index(i) if search_config['use_multiple_colors'] else 0
                if search_config['use_multiple_colors'] and search_config['color_palette']:
                    color = search_config['color_palette'][color_idx % len(search_config['color_palette'])]
                else:
                    color = search_config['color']
                marker_colors.append(color)
                marker_sizes.append(search_size)
            else:
                # Keep original color/size - we'll need to pass these from the calling function
                marker_colors.append(None)  # Will be filled by calling function
                marker_sizes.append(None)   # Will be filled by calling function

        return fig, marker_colors, marker_sizes, matching_indices