#!/usr/bin/env python3
"""
NSM Prime Words Browser
Displays ICML NSM Prime Words dataset using streamlit-aggrid with search and pagination
Now powered by SQLite database for improved performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from semanscope.utils.database import get_database, initialize_nsm_data
from semanscope.config import DATA_PATH
# Try to import AgGrid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
    HAS_AGGRID = True
except ImportError:
    HAS_AGGRID = False

# Page configuration
st.set_page_config(page_title="NSM Prime Words", page_icon="📝", layout="wide")

@st.cache_data
def load_nsm_data_from_db():
    """Load NSM Prime Words dataset from SQLite database with caching"""
    try:
        db = get_database()
        df = db.query_nsm_words()

        if df.empty:
            st.warning("Database is empty. Attempting to initialize from CSV...")
            initialize_nsm_data()
            df = db.query_nsm_words()

        if df.empty:
            st.error("❌ No data available in database")
            return None

        # Add derived columns for better analysis (keep for compatibility)
        df['word_length'] = df['word'].str.len()
        df['is_multiword'] = df['word'].str.contains(' ', na=False)

        return df

    except Exception as e:
        st.error(f"Error loading dataset from database: {str(e)}")
        return None

@st.cache_data
def get_filter_options_cached():
    """Get filter options from database with caching"""
    try:
        db = get_database()
        return db.get_filter_options()
    except Exception as e:
        st.error(f"Error loading filter options: {str(e)}")
        return {
            'domains': ['All'],
            'types': ['All'],
            'languages': ['All'],
            'tiers': ['All'],
            'nsm_groups': ['All']
        }

@st.cache_data
def get_dataset_stats_cached():
    """Get dataset statistics from database with caching"""
    try:
        db = get_database()
        return db.get_dataset_stats()
    except Exception as e:
        st.error(f"Error loading dataset stats: {str(e)}")
        return {
            'total_words': 0,
            'unique_domains': 0,
            'languages': 0,
            'nsm_groups': 0
        }

def apply_filters_and_query(filters):
    """Apply filters and query database directly"""
    try:
        db = get_database()
        return db.query_nsm_words(filters)
    except Exception as e:
        st.error(f"Error querying with filters: {str(e)}")
        return pd.DataFrame()

def create_nsm_table_aggrid(df):
    """Create AgGrid table display with advanced features"""
    if not HAS_AGGRID:
        st.warning("⚠️ Install streamlit-aggrid for enhanced table features: `pip install streamlit-aggrid`")
        return create_nsm_table_basic(df)

    # Prepare display columns (reorder with ID at the end, remove Length and Multi-word)
    display_cols = ['word', 'domain', 'type', 'tier', 'language', 'nsm_prime_group', 'id']
    display_df = df[display_cols].copy()

    # Rename columns for better display
    column_mapping = {
        'word': 'Word',
        'domain': 'Domain',
        'type': 'Type',
        'tier': 'Tier',
        'language': 'Language',
        'nsm_prime_group': 'NSM Group',
        'id': 'ID'
    }
    display_df = display_df.rename(columns=column_mapping)

    # Create GridOptionsBuilder
    gb = GridOptionsBuilder.from_dataframe(display_df)

    # Configure pagination and features (fixed at 10 per page)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
    gb.configure_side_bar()
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_default_column(
        groupable=True,
        value=True,
        enableRowGroup=True,
        editable=False,
        filter=True,
        sortable=True,
        resizable=True
    )

    # Configure specific columns (no checkboxSelection on Word column)
    gb.configure_column('Word', width=180, checkboxSelection=True, headerCheckboxSelection=True)
    gb.configure_column('Domain', width=120)
    gb.configure_column('Type', width=120)
    gb.configure_column('Tier', width=80, type=["numericColumn"])
    gb.configure_column('Language', width=100)
    gb.configure_column('NSM Group', width=180)
    gb.configure_column('ID', width=80, type=["numericColumn"])

    # Add custom CSS for better styling
    custom_css = {
        ".ag-header-cell-text": {"font-weight": "bold !important"},
        ".ag-theme-streamlit .ag-header": {"background-color": "#f0f2f6 !important"},
        ".ag-theme-streamlit .ag-row-even": {"background-color": "#fafafa !important"},
        ".ag-cell-wrap-text": {"white-space": "normal !important"},
    }

    gridOptions = gb.build()

    # Display the grid (adjusted height to avoid white space)
    grid_response = AgGrid(
        display_df,
        gridOptions=gridOptions,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=400,
        theme='streamlit',
        custom_css=custom_css,
        allow_unsafe_jscode=True,
        key="nsm_words_grid"
    )

    # Show selection info
    selected_rows = grid_response['selected_rows']
    if selected_rows is not None and len(selected_rows) > 0:
        st.info(f"✅ Selected {len(selected_rows)} word(s)")

        # Show selected words in expandable section
        with st.expander("📝 View Selected Words"):
            selected_df = pd.DataFrame(selected_rows)
            st.dataframe(selected_df, width='stretch')

            # Export options for selected rows
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("📥 Export Selected Words"):
                    selected_df = pd.DataFrame(selected_rows)
                    csv = selected_df.to_csv(index=False)
                    st.download_button(
                        label="Download Selected Words CSV",
                        data=csv,
                        file_name=f"nsm_selected_words_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="download_selected_words"
                    )

    return grid_response

def create_nsm_table_basic(df):
    """Basic fallback table display when AgGrid is not available"""
    total_words = len(df)
    page_size = 20
    total_pages = max(1, (total_words - 1) // page_size + 1)

    # Page selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_page = st.selectbox(
            f"Page (showing {page_size} words per page)",
            range(1, total_pages + 1),
            index=0,
            key="page_selector"
        )

    # Calculate slice indices
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_words)

    # Display current page info
    st.caption(f"Showing words {start_idx + 1}-{end_idx} of {total_words}")

    # Format data for display
    display_df = df.iloc[start_idx:end_idx].copy()
    display_cols = ['id', 'word', 'domain', 'type', 'tier', 'language', 'nsm_prime_group']
    display_df_formatted = display_df[display_cols].copy()

    # Display table
    st.dataframe(
        display_df_formatted,
        width='stretch',
        height=600,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "word": st.column_config.TextColumn("Word", width="medium"),
            "domain": st.column_config.TextColumn("Domain", width="medium"),
            "type": st.column_config.TextColumn("Type", width="medium"),
            "tier": st.column_config.NumberColumn("Tier", width="small"),
            "language": st.column_config.TextColumn("Language", width="small"),
            "nsm_prime_group": st.column_config.TextColumn("NSM Group", width="large")
        }
    )

    return display_df_formatted

def main():
    st.subheader("📝 NSM Prime Words Browser")

    # Load dataset statistics from database
    stats = get_dataset_stats_cached()

    # Display dataset statistics
    col0, _, col1, col2, col3, col4 = st.columns([5,2, 2,2,2,2])
    with col0:
        st.markdown("Explore the ICML NSM Prime Words dataset with search, filtering, and pagination capabilities.")
    with col1:
        st.metric("Total Words", stats['total_words'])
    with col2:
        st.metric("NSM Groups", stats['nsm_groups'])
    with col3:
        st.metric("Domains", stats['unique_domains'])
    with col4:
        st.metric("Languages", stats['languages'])

    # Get filter options from database
    filter_options = get_filter_options_cached()

    # Filters section - all in one row
    st.markdown("#### 🔍 Filters & Search")

    # Create 6 columns for all filters and search in one row
    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5, filter_col6 = st.columns(6)

    with filter_col1:
        # Domain filter
        selected_domain = st.selectbox("Domain", filter_options['domains'], key="domain_filter")

    with filter_col2:
        # Type filter
        selected_type = st.selectbox("Type", filter_options['types'], key="type_filter")

    with filter_col3:
        # Language filter
        selected_language = st.selectbox("Language", filter_options['languages'], key="language_filter")

    with filter_col4:
        # Tier filter
        selected_tier = st.selectbox("Tier", filter_options['tiers'], key="tier_filter")

    with filter_col5:
        # NSM Group filter
        selected_nsm_group = st.selectbox("NSM Group", filter_options['nsm_groups'], key="nsm_group_filter")

    with filter_col6:
        # Word search
        word_search = st.text_input(
            "🔤 Word Search",
            placeholder="e.g., good, think, big",
            key="word_search"
        )

    # Build filter dictionary
    filters = {
        'domain': selected_domain,
        'type': selected_type,
        'language': selected_language,
        'tier': selected_tier,
        'nsm_prime_group': selected_nsm_group,
        'word_search': word_search.strip() if word_search else None
    }

    # Query database with filters
    display_df = apply_filters_and_query(filters)

    if display_df.empty:
        st.warning("No data found. Please check your filters or database connection.")
        st.stop()

    # st.markdown("---")


    # Show filtered results count
    if len(display_df) != stats['total_words']:
        st.info(f"📊 **Showing {len(display_df)} words** (filtered from {stats['total_words']} total)")
    else:
        st.info(f"📊 **Showing all {len(display_df)} words**")

    if len(display_df) == 0:
        st.warning("No words match your current filters. Try adjusting the criteria.")
        return

    # Display table
    if HAS_AGGRID:
        table_result = create_nsm_table_aggrid(display_df)
    else:
        table_result = create_nsm_table_basic(display_df)

    # st.markdown("---")

    col1, col2 = st.columns([3, 3])
    with col1:
        # Export options
        with st.expander("📥 Export Filtered Results", expanded=False):
            if st.button("Export", key="download_filtered_button"):
                # Export filtered data in the required format
                export_data = []

                # Group by tier and language for separate files
                grouped = display_df.groupby(['tier', 'language'])

                export_dir = DATA_PATH / "input"
                export_dir.mkdir(parents=True, exist_ok=True)

                exported_files = []

                for (tier, language), group in grouped:
                    # Create filename: ICML-NSM-Prime-Words-<tier>-<lang_cd>.txt
                    lang_code = language.lower()
                    filename = f"ICML-NSM-Prime-Words-{tier}-{lang_code}.txt"
                    filepath = export_dir / filename

                    # Prepare export data with required columns mapping
                    export_rows = ["word,domain,type,note"]  # Add header row
                    for _, row in group.iterrows():
                        # Map columns: "Word" as "word", "NSM Group" as "domain", "Type" as "type", "Domain" as "note"
                        export_row = f"{row['word']},{row['nsm_prime_group']},{row['type']},{row['domain']}"
                        export_rows.append(export_row)

                    # Write to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(export_rows))

                    exported_files.append({
                        'filename': filename,
                        'count': len(export_rows),
                        'tier': tier,
                        'language': language
                    })

                # Show success message with file details
                if exported_files:
                    st.success(f"✅ Successfully exported {len(exported_files)} files:")
                    for file_info in exported_files:
                        st.info(f"📄 `{file_info['filename']}` - {file_info['count']} words (Tier {file_info['tier']}, {file_info['language']})")
                    st.markdown(f"**Export location:** `{export_dir}`")
                else:
                    st.warning("No data to export with current filters.")


    with col2:
        # NSM Groups Export Section
        with st.expander("🏷️ Export NSM Groups", expanded=False):
            if st.button("Export", key="export_nsm_groups"):
                try:
                    # Get distinct NSM Group values from database
                    db = get_database()
                    with db.get_connection() as conn:
                        nsm_groups_df = pd.read_sql_query(
                            "SELECT DISTINCT nsm_prime_group FROM nsm_prime_words WHERE nsm_prime_group IS NOT NULL ORDER BY nsm_prime_group",
                            conn
                        )

                    # Prepare export
                    export_dir = DATA_PATH / "input"
                    export_dir.mkdir(parents=True, exist_ok=True)

                    # Export to CSV
                    nsm_groups_file = export_dir / "NSM_GROUP.csv"
                    nsm_groups_df.to_csv(nsm_groups_file, index=False)

                    # Success message
                    st.success(f"✅ Exported {len(nsm_groups_df)} distinct NSM Groups")
                    st.info(f"📄 File saved: `{nsm_groups_file}`")

                    # Show preview of exported data
                    st.markdown("👀 Preview of NSM_GROUP.csv")
                    st.dataframe(nsm_groups_df, width='stretch')

                except Exception as e:
                    st.error(f"Error exporting NSM Groups: {str(e)}")



    # Statistics section (commented out for now)
    # st.markdown("### 📊 Dataset Statistics")

    # # Show distribution by domain
    # stats_col1, stats_col2 = st.columns(2)

    # with stats_col1:
    #     st.markdown("**Words by Domain:**")
    #     domain_counts = display_df['domain'].value_counts()
    #     st.bar_chart(domain_counts.head(10))

    # with stats_col2:
    #     st.markdown("**Words by NSM Group:**")
    #     nsm_group_counts = display_df['nsm_prime_group'].value_counts()
    #     st.bar_chart(nsm_group_counts.head(10))

if __name__ == "__main__":
    main()