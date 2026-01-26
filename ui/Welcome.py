import streamlit as st
from semanscope.config import ST_APP_NAME, ST_ICON, SRC_DIR, DATA_PATH
from pathlib import Path

# Set page config
st.set_page_config(
    page_title=ST_APP_NAME,
    page_icon=ST_ICON,
    layout="wide"
)

def main():
    st.header(f"{ST_ICON} Semanscope")
    
    # Main introduction
    st.markdown("""
    This app helps you discover the geometry of meaning through:
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 🧭 **Semantic Analysis**: Visualize word relationships across languages  
        - 🔍 **Advanced Dual Views**: Simultaneous overview and detailed exploration
        """)
    with col2:
        st.markdown("""
        - 📊 **Publication-Quality Export**: High-resolution figures for research papers
        - 🌐 **Professional Translator**: Enable multilingual study
        """)

    st.markdown("---")
    # Feature overview in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Key Features")
        
        # Core features
        with st.expander("🧭 Semanscope", expanded=True):
            st.markdown("""
            **Core visualization engine**
            - Multiple embedding models (Sentence-BERT, Ollama)
            - Advanced dimensionality reduction (PHATE, t-SNE, UMAP, etc.)
            - Interactive 2D/3D plotting with clustering
            - Real-time rotation and geometric analysis
            - Publication settings with high-DPI export
            """)
        
        with st.expander("🔍 Dual View Explorer"):
            st.markdown("""
            **Advanced research interface**
            - Overview + Detail views simultaneously  
            - Interactive zoom and pan controls
            - Real-time statistics and word lists
            - Publication-ready export system
            - Enhanced geometric analysis
            """)
        
        with st.expander("🖼️ Review Images"):
            st.markdown("""
            **Visualization management**
            - Multi-image comparison (1, 2, or 3 per row)
            - Batch operations and organization
            - Full filename visibility
            - Download and delete functionality
            """)
        
        with st.expander("🌐 Translator"):
            st.markdown("""
            **Professional translation service**
            - High-quality translation API for 30+ languages
            - Auto-detection of source language
            - Research-focused multilingual dataset creation
            - Save translation pairs for reference
            - Editable results for research accuracy
            """)

        with st.expander("📐 Semantic Affinity Benchmark"):
            st.markdown("""
            **Embedding model benchmarking**
            - Quantitative metric for cross-lingual alignment quality
            - Benchmark models using ZiNets dataset (400+ 元字)
            - Compare translation clustering across 8 languages
            - Detect geometric collapse in embeddings
            - Export results for research papers
            """)
    
    with col2:
        st.subheader("🎯 Getting Started")
        
        # Quick start guide
        st.markdown("""
        **For First-Time Users:**
        1. Click **🧭 Semanscope** in the sidebar
        2. Try the pre-loaded sample data
        3. Experiment with different models and methods
        4. Enable clustering to discover patterns
        
        **For Research Applications:**
        1. Use **🔍 Semanscope-Dual View** for detailed analysis
        2. Enable **Publication Mode** for high-quality exports
        3. Use **🖼️ Review Images** to compare multiple visualizations
        4. Export publication-ready figures with standardized naming
        """)
        
        # Sample categories
        with st.expander("📊 Pre-built Semantic Categories"):
            st.markdown("""
            Explore the "Geometry of Meaning" with included datasets:
            - **🎨 Colors** - Perfect branching patterns (warm/cool families)
            - **🔢 Numbers** - **Linear sequence relationships** (major discovery!)
            - **😊 Emotions** - Positive/negative clustering patterns  
            - **🐾 Animals** - Taxonomic family structures
            - **🍎 Food** - Category-based groupings
            """)

    with st.expander("🎉 Release"):
        # Recent updates and achievements
        col_updates, col_metrics = st.columns(2)
        
        with col_updates:
            st.subheader("🎉 Recent Updates (v3.0)")
            st.markdown("""
            **🚀 Major New Features:**
            - ✅ Advanced Dual View Interface
            - ✅ Publication-Quality Export System  
            - ✅ Interactive Zoom & Pan Controls
            - ✅ Enhanced Geometric Analysis
            
            **🏗️ Code Quality Improvements:**
            - ✅ Eliminated 120+ lines of code duplication
            - ✅ Decomposed complex functions (500→220 lines)
            - ✅ Professional error handling
            - ✅ Comprehensive testing framework
            """)
        
        with col_metrics:
            st.subheader("📈 Application Statistics")
            
            # Count saved images
            images_dir = DATA_PATH / "images"
            image_count = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
            
            # Count input datasets  
            input_dir = DATA_PATH / "input"
            if input_dir.exists():
                dataset_names = set()
                for file_path in input_dir.glob("*.txt"):
                    name_part = file_path.stem.replace("-chn", "").replace("-enu", "")
                    dataset_names.add(name_part)
                dataset_count = len(dataset_names)
            else:
                dataset_count = 0
            
            # Display metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Saved Visualizations", image_count, help="Images in data/images/")
                st.metric("Available Datasets", max(dataset_count, 5), help="Pre-built + custom datasets")
            with col_m2:
                st.metric("Embedding Models", "4+", help="Sentence-BERT + Ollama models")
                st.metric("Reduction Methods", "9", help="PHATE, t-SNE, UMAP, etc.")


    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🌎<strong><em> As Rene Descartes gave us coordinates to quantify physical space</em></strong> 🌲</em></strong>  <strong><em>embeddings give us coordinates to digitize mental space</em></strong> 🧠</p>
        <p><small>Built with ❤️ using Claude and Streamlit</small></p>
        <p><small>© Digital Duck LLC</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
else:
    main()  # Always run main() when imported by Streamlit