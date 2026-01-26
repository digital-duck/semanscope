#!/usr/bin/env python3
"""
Generate Figure 3: Full Performance Heatmap for NeurIPS Paper

Creates a publication-quality heatmap showing RA scores across 17 models and 11 categories.

Usage:
    python generate_figure3_full_heatmap.py \
        --input cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \
        --output neurips_figure3-full-heatmap.pdf \
        [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path


# Dataset name mapping for cleaner labels
DATASET_NAMES = {
    'DS1': 'Family Relations',
    'DS2': 'Royalty Hierarchy',
    'DS3': 'Gendered Occupations',
    'DS4': 'Comparative-Superlative',
    'DS5': 'Opposite Relations',
    'DS6': 'Animal Gender',
    'DS7': 'Sequential-Math',
    'DS8': 'Hierarchical Relations',
    'DS9': 'Part-Whole',
    'DS10': 'Size-Scale',
    'DS11': 'Cause-Effect',
}


def parse_score(val: str) -> float:
    """Parse score from 'mean ± sem' format."""
    if pd.isna(val) or val == 'N/A':
        return np.nan
    try:
        return float(val.split('±')[0].strip())
    except:
        return np.nan


def create_full_heatmap(
    data_path: str,
    output_path: str,
    figsize: tuple = (14, 10),
    dpi: int = 300,
    verbose: bool = True
):
    """
    Create full performance heatmap showing RA scores across all models and categories.

    Args:
        data_path: Path to RA pivot CSV
        output_path: Path to save figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output
        verbose: Print progress messages
    """
    if verbose:
        click.echo(f"📊 Generating Figure 3: Full Performance Heatmap")
        click.echo(f"   Reading data: {data_path}")

    # Read data
    df = pd.read_csv(data_path)

    # Parse scores
    score_matrix = []
    models = []
    categories = []

    for idx, row in df.iterrows():
        # Clean model name: remove (OpenRouter) and (Ollama) suffixes
        model_name = row['model']
        model_name = model_name.replace(' (OpenRouter)', '').replace(' (Ollama)', '')
        models.append(model_name)
        scores = []
        for col in df.columns[1:]:  # Skip 'model' column
            scores.append(parse_score(row[col]))
        score_matrix.append(scores)

    # Get category labels
    for col in df.columns[1:]:
        ds_id = col.split('[')[0].strip()
        categories.append(DATASET_NAMES.get(ds_id, ds_id))

    # Create numpy array
    data = np.array(score_matrix)

    # Swap columns 3 and 5 (DS4 and DS6) - 0-indexed, so index 3 and 5
    # DS4 (Comparative-Superlative) was at index 3, DS6 (Animal Gender) was at index 5
    # User swapped them conceptually, so swap the data values
    data[:, [3, 5]] = data[:, [5, 3]]

    if verbose:
        click.echo(f"   Heatmap dimensions: {len(models)} models × {len(categories)} categories")
        click.echo(f"   RA score range: {np.nanmin(data):.4f} to {np.nanmax(data):.4f}")

    # Set publication style - use white style without grid
    plt.style.use('seaborn-v0_8-white')
    sns.set_context("paper", font_scale=1.1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap with custom colormap (red→yellow→green)
    # Use vmin=0, vmax=0.8 to highlight the differences
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.8)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    # Rotate Y-axis labels 45 degrees like X-axis, use multi-line if needed
    ax.set_yticklabels(models, rotation=45, ha='right', fontsize=9)

    # Disable all grid lines
    ax.grid(False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relational Affinity (RA)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Annotate cells with values - use all black text with larger font
    for i in range(len(models)):
        for j in range(len(categories)):
            val = data[i, j]
            if not np.isnan(val):
                # Use all black text, much larger font size for maximum visibility
                text = ax.text(j, i, f'{val:.3f}',
                             ha="center", va="center", color='black', fontsize=14, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Semantic Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Embedding Model', fontsize=13, fontweight='bold')
    ax.set_title(
        'Figure 3: Relational Affinity Performance Heatmap',
        fontsize=15,
        fontweight='bold',
        pad=20
    )

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    if verbose:
        click.echo(f"\n   ✅ Figure saved: {output_path}")
        click.echo(f"   Resolution: {dpi} DPI")
        click.echo(f"   Size: {figsize[0]}\" × {figsize[1]}\"")

    # Also save as PNG for quick preview
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    if verbose:
        click.echo(f"   ✅ Preview saved: {png_path}")

    plt.close()

    # Print summary statistics
    if verbose:
        click.echo(f"\n📊 Performance Summary:")
        click.echo(f"   Overall mean RA: {np.nanmean(data):.4f}")
        click.echo(f"   Overall std RA:  {np.nanstd(data):.4f}")

        # Find best and worst categories
        cat_means = np.nanmean(data, axis=0)
        best_cat_idx = np.nanargmax(cat_means)
        worst_cat_idx = np.nanargmin(cat_means)
        click.echo(f"\n   Best category:  {categories[best_cat_idx]} (mean RA={cat_means[best_cat_idx]:.4f})")
        click.echo(f"   Worst category: {categories[worst_cat_idx]} (mean RA={cat_means[worst_cat_idx]:.4f})")

        # Find best and worst models
        model_means = np.nanmean(data, axis=1)
        best_model_idx = np.nanargmax(model_means)
        worst_model_idx = np.nanargmin(model_means)
        click.echo(f"\n   Best model:  {models[best_model_idx]} (mean RA={model_means[best_model_idx]:.4f})")
        click.echo(f"   Worst model: {models[worst_model_idx]} (mean RA={model_means[worst_model_idx]:.4f})")


@click.command()
@click.option(
    '--input',
    '-i',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to RA pivot CSV file'
)
@click.option(
    '--output',
    '-o',
    required=True,
    type=click.Path(dir_okay=False),
    help='Path to save output figure (PDF recommended)'
)
@click.option(
    '--dpi',
    default=300,
    type=int,
    help='Resolution for output figure (default: 300)'
)
@click.option(
    '--figsize',
    default=(14, 10),
    type=(float, float),
    help='Figure size in inches (width, height) (default: 14 10)'
)
@click.option(
    '--verbose/--quiet',
    '-v/-q',
    default=True,
    help='Print progress messages'
)
def main(
    input: str,
    output: str,
    dpi: int,
    figsize: tuple,
    verbose: bool
):
    """
    Generate Figure 3: Full performance heatmap for NeurIPS paper.

    Creates a publication-quality heatmap showing RA scores across all 17 models
    and 11 semantic categories, visually highlighting catastrophic failures
    (Sequential-Math column should be uniformly red).

    Examples:

        # Basic usage
        python generate_figure3_full_heatmap.py \\
            --input cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \\
            --output neurips_figure3-full-heatmap.pdf

        # High resolution
        python generate_figure3_full_heatmap.py \\
            --input cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \\
            --output neurips_figure3-full-heatmap.pdf \\
            --dpi 600
    """
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("FIGURE 3 GENERATION: Full Performance Heatmap")
        click.echo("=" * 80 + "\n")

    create_full_heatmap(
        data_path=input,
        output_path=output,
        figsize=figsize,
        dpi=dpi,
        verbose=verbose
    )

    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("✅ Figure 3 generation complete!")
        click.echo(f"   Main figure: {output}")
        click.echo(f"   Preview: {Path(output).with_suffix('.png')}")
        click.echo("=" * 80 + "\n")


if __name__ == '__main__':
    main()
