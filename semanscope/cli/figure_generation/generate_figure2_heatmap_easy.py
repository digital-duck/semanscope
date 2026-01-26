#!/usr/bin/env python3
"""
Generate Figure 2: Performance Heatmap on Easy Datasets for NeurIPS Paper

Shows RA scores for 17 models across 4 easy datasets (DS1-DS4).

Usage:
    python generate_figure2_heatmap_easy.py \
        --input cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \
        --output neurips_figure2_heatmap_easy_datasets.pdf \
        [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path


# Dataset name mapping
DATASET_NAMES = {
    'DS1': 'Family Relations',
    'DS2': 'Royalty Hierarchy',
    'DS3': 'Gendered Occupations',
    'DS4': 'Animal Gender',  # Note: Swapped with DS6 in actual data
}


def parse_score(val: str) -> float:
    """Parse score from 'mean ± sem' format."""
    if pd.isna(val) or val == 'N/A':
        return np.nan
    try:
        return float(val.split('±')[0].strip())
    except:
        return np.nan


def create_heatmap_easy_datasets(
    data_path: str,
    output_path: str,
    figsize: tuple = (10, 10),
    dpi: int = 300,
    verbose: bool = True
):
    """
    Create heatmap showing RA scores on easy datasets (DS1-DS4).

    Args:
        data_path: Path to RA pivot CSV
        output_path: Path to save figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output
        verbose: Print progress messages
    """
    if verbose:
        click.echo(f"📊 Generating Figure 2: Easy Datasets Heatmap")
        click.echo(f"   Reading data: {data_path}")

    # Read data
    df = pd.read_csv(data_path)

    # Easy datasets columns (with corrected 2025-12-29 data: DS1-DS4)
    # DS1: Family Relations (column 1)
    # DS2: Royalty Hierarchy (column 2)
    # DS3: Gendered Occupations (column 3)
    # DS4: Comparative-Superlative (column 4) - corrected naming
    easy_col_indices = [1, 2, 3, 4]
    easy_labels = ['Family Relations', 'Royalty Hierarchy', 'Gendered Occupations', 'Comparative-Superlative']

    # Parse scores
    score_matrix = []
    models = []

    for idx, row in df.iterrows():
        model_name = row['model'].replace(' (OpenRouter)', '').replace(' (Ollama)', '')
        models.append(model_name)

        scores = []
        for col_idx in easy_col_indices:
            col = df.columns[col_idx]
            scores.append(parse_score(row[col]))

        score_matrix.append(scores)

    # Create numpy array
    data = np.array(score_matrix)

    # Calculate average RA for tier assignment
    avg_ra = np.nanmean(data, axis=1)
    sorted_indices = np.argsort(avg_ra)[::-1]  # Sort descending

    # Reorder data and models by average RA
    data = data[sorted_indices]
    models = [models[i] for i in sorted_indices]
    avg_ra = avg_ra[sorted_indices]

    if verbose:
        click.echo(f"   Heatmap dimensions: {len(models)} models × {len(easy_labels)} categories")
        click.echo(f"   RA score range: {np.nanmin(data):.4f} to {np.nanmax(data):.4f}")

    # Set publication style
    plt.style.use('seaborn-v0_8-white')
    sns.set_context("paper", font_scale=1.1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.8)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(easy_labels)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(easy_labels, rotation=20, ha='right', fontsize=11)
    ax.set_yticklabels(models, rotation=45, ha='right', fontsize=10)

    # Disable grid
    ax.grid(False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relational Affinity (RA)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Annotate cells with values
    for i in range(len(models)):
        for j in range(len(easy_labels)):
            val = data[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}',
                             ha="center", va="center", color='black', fontsize=14, fontweight='bold')

    # Add tier separation lines (without text labels)
    # Find tier boundaries
    tier1_boundary = np.where(avg_ra > 0.50)[0]
    tier2_boundary = np.where((avg_ra >= 0.35) & (avg_ra <= 0.50))[0]

    if len(tier1_boundary) > 0:
        tier1_end = tier1_boundary[-1] + 0.5
        ax.axhline(y=tier1_end, color='black', linestyle='--', linewidth=2, alpha=0.7)

    if len(tier2_boundary) > 0:
        tier2_end = tier2_boundary[-1] + 0.5
        ax.axhline(y=tier2_end, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # Labels and title
    ax.set_xlabel('Relational Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Embedding Model (Sorted by Avg RA)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Figure 2: Performance Heatmap on DS1-DS4',
        fontsize=14,
        fontweight='bold',
        pad=15
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

    # Save PNG version
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
        click.echo(f"\n   Best category:  {easy_labels[best_cat_idx]} (mean RA={cat_means[best_cat_idx]:.4f})")
        click.echo(f"   Worst category: {easy_labels[worst_cat_idx]} (mean RA={cat_means[worst_cat_idx]:.4f})")

        # Best and worst models
        click.echo(f"\n   Best model:  {models[0]} (avg RA={avg_ra[0]:.4f})")
        click.echo(f"   Worst model: {models[-1]} (avg RA={avg_ra[-1]:.4f})")


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
    default=(10, 10),
    type=(float, float),
    help='Figure size in inches (width, height) (default: 10 10)'
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
    Generate Figure 2: Performance heatmap on easy datasets.

    Shows RA scores for all 17 models across 4 easy categories (DS1-DS4),
    with tier boundaries marked.
    """
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("FIGURE 2 GENERATION: Easy Datasets Heatmap")
        click.echo("=" * 80 + "\n")

    create_heatmap_easy_datasets(
        data_path=input,
        output_path=output,
        figsize=figsize,
        dpi=dpi,
        verbose=verbose
    )

    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("✅ Figure 2 generation complete!")
        click.echo(f"   Main figure: {output}")
        click.echo(f"   Preview: {Path(output).with_suffix('.png')}")
        click.echo("=" * 80 + "\n")


if __name__ == '__main__':
    main()
