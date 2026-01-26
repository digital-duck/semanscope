#!/usr/bin/env python3
"""
Generate Figure 2: Performance Heatmap on Easy Datasets by Language

Shows RA scores for 17 models across 4 easy datasets for a specific language.

Usage:
    python generate_figure2_heatmap_easy_by_language.py \
        --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \
        --output neurips_figure2_heatmap_easy_datasets_EN.pdf \
        --language EN \
        [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path


# Easy datasets for Figure 2
# v2.5: Updated dataset names with -v2.5 suffix
EASY_DATASETS = [
    'NeurIPS-01-family-relations-v2.5',
    'NeurIPS-02-royalty-hierarchy-v2.5',
    'NeurIPS-03-gendered-occupations-v2.5',
    'NeurIPS-04-comparative-superlative-v2.5',
]

DATASET_LABELS = [
    'Family Relations',
    'Royalty Hierarchy',
    'Gendered Occupations',
    'Comparative-Superlative',
]


def create_heatmap_easy_datasets(
    data_path: str,
    output_path: str,
    language: str,  # 'EN', 'ZH', or 'cross'
    figsize: tuple = (10, 10),
    dpi: int = 300,
    verbose: bool = True
):
    """
    Create heatmap showing RA scores on easy datasets for a specific language.

    Args:
        data_path: Path to RA raw CSV
        output_path: Path to save figure
        language: 'EN', 'ZH', or 'cross'
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output
        verbose: Print progress messages
    """
    if verbose:
        lang_name = 'English' if language == 'EN' else 'Chinese' if language == 'ZH' else 'Cross-lingual'
        click.echo(f"📊 Generating Figure 2 ({lang_name}): Easy Datasets Heatmap")
        click.echo(f"   Reading data: {data_path}")

    # Read raw RA data
    df = pd.read_csv(data_path)

    # Select RA column based on language
    ra_col = f'cos_{language}'

    if verbose:
        click.echo(f"   Using RA column: {ra_col}")
        click.echo(f"   Easy datasets: {', '.join(EASY_DATASETS)}")

    # Get unique models
    models = df['model'].unique().tolist()

    # Create matrix: rows=models, columns=easy datasets
    score_matrix = []
    model_labels = []

    for model in models:
        model_name = model.replace(' (OpenRouter)', '').replace(' (Ollama)', '')
        model_labels.append(model_name)

        row_scores = []
        for dataset in EASY_DATASETS:
            # Find RA score for this model-dataset pair
            match = df[(df['model'] == model) & (df['dataset'] == dataset)]
            if len(match) > 0:
                score = match.iloc[0][ra_col]
                row_scores.append(score)
            else:
                row_scores.append(np.nan)
        score_matrix.append(row_scores)

    # Create numpy array
    data = np.array(score_matrix)

    # Calculate average RA for tier assignment
    avg_ra = np.nanmean(data, axis=1)
    sorted_indices = np.argsort(avg_ra)[::-1]  # Sort descending

    # Reorder data and models by average RA
    data = data[sorted_indices]
    model_labels = [model_labels[i] for i in sorted_indices]
    avg_ra = avg_ra[sorted_indices]

    if verbose:
        click.echo(f"   Heatmap dimensions: {len(model_labels)} models × {len(DATASET_LABELS)} datasets")
        click.echo(f"   RA score range: {np.nanmin(data):.4f} to {np.nanmax(data):.4f}")

    # Set publication style
    plt.style.use('seaborn-v0_8-white')
    sns.set_context("paper", font_scale=1.1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.8)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(DATASET_LABELS)))
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_xticklabels(DATASET_LABELS, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(model_labels, rotation=45, ha='right', fontsize=10)

    # Disable grid
    ax.grid(False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relational Affinity (RA)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Annotate cells with values
    for i in range(len(model_labels)):
        for j in range(len(DATASET_LABELS)):
            val = data[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}',
                             ha="center", va="center", color='black', fontsize=13, fontweight='bold')

    # Add tier separator lines
    # Find tier boundaries
    tier1_end = np.sum(avg_ra > 0.50)
    tier2_end = tier1_end + np.sum((avg_ra >= 0.30) & (avg_ra <= 0.50))

    if tier1_end > 0 and tier1_end < len(model_labels):
        ax.axhline(y=tier1_end - 0.5, color='black', linestyle='-', linewidth=2.5)
        # Add tier label
        ax.text(-0.5, tier1_end / 2 - 0.5, 'Tier 1', rotation=90, va='center', ha='center',
                fontsize=11, fontweight='bold', color='green')

    if tier2_end > tier1_end and tier2_end < len(model_labels):
        ax.axhline(y=tier2_end - 0.5, color='black', linestyle='-', linewidth=2.5)
        # Add tier label
        ax.text(-0.5, (tier1_end + tier2_end) / 2 - 0.5, 'Tier 2', rotation=90, va='center', ha='center',
                fontsize=11, fontweight='bold', color='orange')

    if tier2_end < len(model_labels):
        # Add tier label for Tier 3
        ax.text(-0.5, (tier2_end + len(model_labels)) / 2 - 0.5, 'Tier 3', rotation=90, va='center', ha='center',
                fontsize=11, fontweight='bold', color='red')

    # Labels and title
    lang_name = 'English' if language == 'EN' else 'Chinese' if language == 'ZH' else 'Cross-lingual'
    ax.set_xlabel('Semantic Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Embedding Model (Ranked by Average RA)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Figure 2: Performance Heatmap on DS01-DS04 ({lang_name})',
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

    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    if verbose:
        click.echo(f"   ✅ Preview saved: {png_path}")

    plt.close()

    # Print summary statistics
    if verbose:
        click.echo(f"\n📊 Performance Summary ({lang_name}):")
        click.echo(f"   Overall mean RA: {np.nanmean(data):.4f}")
        click.echo(f"   Overall std RA:  {np.nanstd(data):.4f}")

        # Find best and worst datasets
        dataset_means = np.nanmean(data, axis=0)
        best_ds_idx = np.nanargmax(dataset_means)
        worst_ds_idx = np.nanargmin(dataset_means)
        click.echo(f"\n   Best dataset:  {DATASET_LABELS[best_ds_idx]} (mean RA={dataset_means[best_ds_idx]:.4f})")
        click.echo(f"   Worst dataset: {DATASET_LABELS[worst_ds_idx]} (mean RA={dataset_means[worst_ds_idx]:.4f})")

        # Print tier counts
        tier1_count = np.sum(avg_ra > 0.50)
        tier2_count = np.sum((avg_ra >= 0.30) & (avg_ra <= 0.50))
        tier3_count = np.sum(avg_ra < 0.30)

        click.echo(f"\n   Tier 1 (RA > 0.50): {tier1_count} models")
        click.echo(f"   Tier 2 (0.30 ≤ RA ≤ 0.50): {tier2_count} models")
        click.echo(f"   Tier 3 (RA < 0.30): {tier3_count} models")


@click.command()
@click.option(
    '--input',
    '-i',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to RA raw CSV file'
)
@click.option(
    '--output',
    '-o',
    required=True,
    type=click.Path(dir_okay=False),
    help='Path to save output figure (PDF recommended)'
)
@click.option(
    '--language',
    '-l',
    required=True,
    type=click.Choice(['EN', 'ZH', 'ES', 'cross'], case_sensitive=False),
    help='Language to analyze: EN (English), ZH (Chinese), ES (Spanish), or cross (cross-lingual)'
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
    language: str,
    dpi: int,
    figsize: tuple,
    verbose: bool
):
    """
    Generate Figure 2: Easy datasets performance heatmap.

    Shows RA scores for all models on easy datasets (Family Relations,
    Royalty Hierarchy, Gendered Occupations, Comparative-Superlative).

    Examples:

        # English version
        python generate_figure2_heatmap_easy_by_language.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \\
            --output neurips_figure2_heatmap_easy_EN.pdf \\
            --language EN

        # Chinese version
        python generate_figure2_heatmap_easy_by_language.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \\
            --output neurips_figure2_heatmap_easy_ZH.pdf \\
            --language ZH
    """
    if verbose:
        lang_name = 'English' if language == 'EN' else 'Chinese' if language == 'ZH' else 'Cross-lingual'
        click.echo("\n" + "=" * 80)
        click.echo(f"FIGURE 2 GENERATION: Easy Datasets Heatmap ({lang_name})")
        click.echo("=" * 80 + "\n")

    create_heatmap_easy_datasets(
        data_path=input,
        output_path=output,
        language=language,
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
