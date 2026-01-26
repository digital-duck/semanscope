"""
Download helper functions for Streamlit application
"""
import streamlit as st


def handle_download_button(fig, model_name, method_name, dataset_name, chart_type="detail", page_type="dual_view"):
    """
    Create a standardized download button for figures
    
    Args:
        fig: Plotly figure object to download
        model_name: Name of the embedding model
        method_name: Name of the reduction method  
        dataset_name: Name of the dataset
        chart_type: Type of chart ("detail", "clustering", or "main")
        page_type: Type of page ("dual_view" or "main") - affects settings source
    """
    try:
        # Get settings based on page type
        if page_type == "dual_view":
            # Get publication settings if available, otherwise use defaults
            pub_settings = st.session_state.get('dual_view_publication_settings', {})
            textfont_size = pub_settings.get('textfont_size', 16)
            point_size = pub_settings.get('point_size', 12)
            export_dpi = pub_settings.get('export_dpi', 300)
            export_format = pub_settings.get('export_format', 'PNG')
            plot_width = pub_settings.get('plot_width', 800)
            plot_height = pub_settings.get('plot_height', 700)
        else:
            # Main page doesn't have publication settings, use defaults
            textfont_size = 16
            point_size = 12
            export_dpi = 300
            export_format = 'PNG'
            plot_width = 800
            plot_height = 700
        
        # Clean names for filename
        clean_method = (method_name or "unknown-method").lower().replace(" ", "-").replace(",", "").replace("_", "-")
        clean_model = (model_name or "unknown-model").lower().replace(" ", "-").replace(",", "").replace("_", "-")
        clean_dataset = dataset_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
        
        # Create filename based on chart type
        if chart_type == "detail":
            filename = f"{clean_method}-{clean_model}-{clean_dataset}-dpi-{export_dpi}-text-{textfont_size}-point-{point_size}-{chart_type}-1.{export_format.lower()}"
        else:
            filename = f"{clean_method}-{clean_model}-{clean_dataset}-dpi-{export_dpi}-text-{textfont_size}-point-{point_size}-{chart_type}.{export_format.lower()}"
        label = f"📥 Download Image {export_format} ({export_dpi} DPI)"
        key = f"download_{chart_type}_view_{page_type}"
        
        # Export figure based on format
        if export_format.upper() == 'SVG':
            img_bytes = fig.to_image(format="svg", width=plot_width, height=plot_height)
        elif export_format.upper() == 'PDF':
            img_bytes = fig.to_image(format="pdf", width=plot_width, height=plot_height)
        else:
            # Fallback to PNG
            img_bytes = fig.to_image(format="png", width=plot_width, height=plot_height, scale=export_dpi/96)
        
        # Create download button
        st.download_button(
            label=label,
            data=img_bytes,
            file_name=filename,
            mime=f"image/{export_format.lower()}",
            key=key
        )
        
    except Exception as e:
        st.warning(f"Could not create {chart_type} download button: {str(e)}")


def get_clean_filename_parts(model_name, method_name, dataset_name):
    """
    Helper function to get cleaned filename parts
    
    Returns:
        tuple: (clean_method, clean_model, clean_dataset)
    """
    clean_method = (method_name or "unknown-method").lower().replace(" ", "-").replace(",", "").replace("_", "-")
    clean_model = (model_name or "unknown-model").lower().replace(" ", "-").replace(",", "").replace("_", "-")
    clean_dataset = dataset_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
    
    return clean_method, clean_model, clean_dataset