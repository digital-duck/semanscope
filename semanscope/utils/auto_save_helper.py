"""
Standardized Auto-Save Helper for Semanscope
Provides consistent auto-save functionality across all Semanscope pages.
"""

import os
import streamlit as st
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

from utils.title_filename_helper import create_title_and_filename


def auto_save_visualization(
    visualizer: Any,
    method_name: str,
    model_name: str,
    dataset_name: str,
    active_languages: List[str],
    dimensions: str = "2D",
    page_prefix: str = "",
    custom_config: Optional[Dict] = None,
    enable_debug: bool = False
) -> Optional[List[str]]:
    """
    Standardized auto-save function for all Semanscope pages.

    Args:
        visualizer: The visualizer object with plot manager
        method_name: Dimensionality reduction method name
        model_name: Embedding model name
        dataset_name: Dataset/input name
        active_languages: List of language codes (e.g., ['chn', 'enu'])
        dimensions: "2D" or "3D" (auto-save only works for 2D)
        page_prefix: Optional prefix for filename (e.g., "echarts-", "comparison-")
        custom_config: Optional custom ECharts configuration
        enable_debug: Enable debug output

    Returns:
        List of saved filenames or None if auto-save failed/disabled
    """
    try:
        # Check if auto-save is enabled globally
        auto_save_settings = st.session_state.get('echarts_auto_save', {'enabled': False})
        if not auto_save_settings.get('enabled', False):
            if enable_debug:
                st.info("🔍 **Auto-save Debug**: Auto-save disabled in settings")
            return None

        # Only auto-save 2D visualizations for stability
        if dimensions != "2D":
            if enable_debug:
                st.info(f"🔍 **Auto-save Debug**: dimensions='{dimensions}', auto-save only works for 2D")
            return None

        # Detect if visualizer IS the ECharts plot manager or contains one
        plot_manager = None
        if hasattr(visualizer, 'get_auto_save_status'):
            # Visualizer IS the plot manager
            plot_manager = visualizer
        elif hasattr(visualizer, 'echarts_plot_manager'):
            # Visualizer contains a plot manager
            plot_manager = visualizer.echarts_plot_manager

        # Check if selenium/auto-save is available
        auto_save_status = None
        if plot_manager and hasattr(plot_manager, 'get_auto_save_status'):
            auto_save_status = plot_manager.get_auto_save_status()
            if not auto_save_status.get('available', False):
                if enable_debug:
                    st.warning("⚠️ Auto-save not available: Selenium WebDriver not found")
                return None

        saved_files = []

        # Get ECharts configuration
        echarts_config = custom_config
        if not echarts_config:
            if plot_manager and hasattr(plot_manager, 'last_chart_config'):
                echarts_config = plot_manager.last_chart_config
            elif 'current_echarts_config' in st.session_state:
                echarts_config = st.session_state.current_echarts_config

        if not echarts_config:
            if enable_debug:
                st.warning("🔍 **Debug**: No ECharts config available for auto-save")
            return None

        # Determine format based on publication settings (always respect Publication tab format)
        pub_settings = st.session_state.get('global_settings', {}).get('publication', {})
        export_format = pub_settings.get('export_format', 'PNG').upper()

        if enable_debug:
            st.info(f"🔍 **Auto-save Format**: Using {export_format} (from Publication settings)")

        # Generate standardized filename using centralized helper with correct format
        _, standardized_filename = create_title_and_filename(
            [method_name],
            [model_name],
            dataset_name,
            active_languages,
            export_format.lower()
        )

        # Apply page prefix
        standardized_filename_str = str(standardized_filename)
        if page_prefix:
            final_filename = f"{page_prefix}{standardized_filename_str}"
        else:
            final_filename = standardized_filename_str

        # Create both PNG and PDF versions for fallback
        png_filename = final_filename.replace('.pdf', '.png') if final_filename.endswith('.pdf') else final_filename
        pdf_filename = final_filename.replace('.png', '.pdf') if final_filename.endswith('.png') else final_filename

        # Perform the auto-save using the detected plot_manager
        if export_format == 'PDF' and plot_manager and hasattr(plot_manager, 'save_echarts_as_pdf'):
            # Auto-save as PDF for publication mode
            with st.spinner("📄 Auto-saving PDF..."):
                saved_result = plot_manager.save_echarts_as_pdf(
                    echarts_config,
                    [],  # Empty filename_parts since we're using external_filename
                    dimensions,
                    external_filename=pdf_filename
                )
        else:
            # Auto-save as PNG (default behavior)
            with st.spinner("📸 Auto-saving PNG..."):
                if plot_manager and hasattr(plot_manager, 'save_echarts_as_png_auto'):
                    saved_result = plot_manager.save_echarts_as_png_auto(
                        echarts_config,
                        [],  # Empty filename_parts since we're using external_filename
                        dimensions,
                        width=auto_save_settings.get('width', 1200),
                        height=auto_save_settings.get('height', 800),
                        external_filename=png_filename
                    )
                elif plot_manager and hasattr(plot_manager, 'save_echarts_as_png'):
                    # Fallback to basic PNG save
                    saved_result = plot_manager.save_echarts_as_png(
                        echarts_config,
                        [],  # Empty filename_parts since we're using external_filename
                        dimensions,
                        external_filename=png_filename
                    )
                else:
                    if enable_debug:
                        st.warning("🔍 **Debug**: No compatible save method found on plot manager")
                    return None

        # Handle successful save
        if saved_result:
            if isinstance(saved_result, dict):
                filename = saved_result.get('filename', png_filename)
                saved_files.append(filename)
                # Store the filepath for later display
                st.session_state['last_auto_save_path'] = saved_result.get('filepath', '')
            else:
                saved_files.append(str(saved_result))
                # Try to construct filepath for older format
                if plot_manager and hasattr(plot_manager, 'images_dir'):
                    images_dir = plot_manager.images_dir
                else:
                    images_dir = os.path.join("src", "data", "images", "echarts")
                st.session_state['last_auto_save_path'] = os.path.join(images_dir, str(saved_result))

            if enable_debug:
                st.info(f"🔍 **Auto-save Success**: Saved as {saved_files}")

        return saved_files

    except Exception as auto_save_error:
        if enable_debug:
            st.error(f"🔍 **Auto-save Error Details**: {auto_save_error}")
        st.warning(f"Could not auto-save visualization: {str(auto_save_error)}")
        return None


def auto_save_legacy_plot(
    visualizer: Any,
    current_input: str,
    model_name: str,
    method_name: str,
    chinese_selected: bool,
    english_selected: bool,
    dimensions: str = "2D"
) -> Optional[str]:
    """
    Auto-save using legacy plot image method for compatibility.

    Args:
        visualizer: The visualizer object
        current_input: Input dataset name
        model_name: Embedding model name
        method_name: Dimensionality reduction method name
        chinese_selected: Whether Chinese is selected
        english_selected: Whether English is selected
        dimensions: "2D" or "3D"

    Returns:
        Saved filename or None if failed
    """
    try:
        if hasattr(visualizer, 'save_plot_image'):
            saved_filename = visualizer.save_plot_image(
                current_input, model_name, method_name,
                chinese_selected, english_selected, dimensions
            )
            return saved_filename
        else:
            st.warning("Legacy save_plot_image method not available")
            return None

    except Exception as error:
        st.warning(f"Could not auto-save using legacy method: {str(error)}")
        return None


def display_auto_save_success(saved_files: List[str], page_emoji: str = "📊"):
    """
    Display standardized success message for auto-saved files.

    Args:
        saved_files: List of saved filenames
        page_emoji: Emoji to use for the page (default: 📊)
    """
    if saved_files:
        files_display = ' | '.join(saved_files)
        st.success(f"{page_emoji} **Auto-saved**: {files_display}")


def get_auto_save_settings() -> Dict[str, Any]:
    """
    Get current auto-save settings from session state.

    Returns:
        Dictionary with auto-save settings
    """
    # Try to get from global_settings first (Settings page format)
    global_settings = st.session_state.get('global_settings', {})
    echarts_settings = global_settings.get('echarts', {})

    # Check both the global settings and the legacy echarts_auto_save location
    enabled = echarts_settings.get('auto_save_enabled', False)

    # Also check the legacy location for backward compatibility
    legacy_auto_save = st.session_state.get('echarts_auto_save', {})
    if not enabled and legacy_auto_save.get('enabled', False):
        enabled = True

    return {
        'enabled': enabled,
        'width': echarts_settings.get('png_width', 1200),
        'height': echarts_settings.get('png_height', 800),
        'available': False  # Will be updated by selenium check
    }


def check_auto_save_availability(visualizer: Any) -> Dict[str, Any]:
    """
    Check if auto-save functionality is available.

    Args:
        visualizer: The visualizer object

    Returns:
        Dictionary with availability status and details
    """
    status = {
        'available': False,
        'has_plot_manager': False,
        'has_selenium': False,
        'reason': 'Unknown'
    }

    if hasattr(visualizer, 'echarts_plot_manager'):
        status['has_plot_manager'] = True
        if hasattr(visualizer.echarts_plot_manager, 'get_auto_save_status'):
            plot_status = visualizer.echarts_plot_manager.get_auto_save_status()
            status.update(plot_status)
        else:
            status['reason'] = 'Auto-save status method not available'
    else:
        status['reason'] = 'ECharts plot manager not available'

    return status