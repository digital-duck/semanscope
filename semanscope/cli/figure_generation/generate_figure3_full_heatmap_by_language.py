#!/usr/bin/env python3
"""
Generate Figure 3: Full Performance Heatmap by Language for NeurIPS Paper

Creates language-specific heatmaps showing RA scores for EN, ZH, or cross-lingual.

Usage:
    python generate_figure3_full_heatmap_by_language.py \
        --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \
        --output neurips_figure3-full-heatmap-EN.pdf \
        --language EN \
        [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path


# Dataset name mapping for cleaner labels (v2.5 dataset names)
DATASET_NAME_MAP = {
    'NeurIPS-01-family-relations-v2.5': 'Family Relations',
    'NeurIPS-02-royalty-hierarchy-v2.5': 'Royalty Hierarchy',
    'NeurIPS-03-gendered-occupations-v2.5': 'Gendered Occupations',
    'NeurIPS-04-comparative-superlative-v2.5': 'Comparative-Superlative',
    'NeurIPS-05-opposite-relations-v2.5': 'Opposite Relations',
    'NeurIPS-06-animal-gender-v2.5': 'Animal Gender',
    'NeurIPS-07-sequential-mathematical-v2.5': 'Sequential-Math',
    'NeurIPS-08-hierarchical-relations-v2.5': 'Hierarchical Relations',
    'NeurIPS-09-part-whole-v2.5': 'Part-Whole',
    'NeurIPS-10-size-scale-v2.5': 'Size-Scale',
    'NeurIPS-11-cause-effect-v2.5': 'Cause-Effect',
}

# Desired order for datasets (11 main categories for v2.5)
DATASET_ORDER = [
    'NeurIPS-01-family-relations-v2.5',
    'NeurIPS-02-royalty-hierarchy-v2.5',
    'NeurIPS-03-gendered-occupations-v2.5',
    'NeurIPS-04-comparative-superlative-v2.5',
    'NeurIPS-05-opposite-relations-v2.5',
    'NeurIPS-06-animal-gender-v2.5',
    'NeurIPS-07-sequential-mathematical-v2.5',
    'NeurIPS-08-hierarchical-relations-v2.5',
    'NeurIPS-09-part-whole-v2.5',
    'NeurIPS-10-size-scale-v2.5',
    'NeurIPS-11-cause-effect-v2.5',
]


def create_full_heatmap(
    data_path: str,
    output_path: str,
    language: str,  # 'EN', 'ZH', or 'cross'
    figsize: tuple = (14, 10),
    dpi: int = 300,
    verbose: bool = True
):
    """
    Create full performance heatmap for a specific language.

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
        click.echo(f"📊 Generating Figure 3 ({lang_name}): Full Performance Heatmap")
        click.echo(f"   Reading data: {data_path}")

    # Read raw RA data
    df = pd.read_csv(data_path)

    # Select RA column based on language
    ra_col = f'cos_{language}'

    if verbose:
        click.echo(f"   Using RA column: {ra_col}")

    # Get unique models and datasets
    models = df['model'].unique().tolist()
    datasets_in_data = df['dataset'].unique().tolist()

    # Filter to only include datasets in our desired order
    datasets = [ds for ds in DATASET_ORDER if ds in datasets_in_data]

    if verbose:
        click.echo(f"   Models: {len(models)}")
        click.echo(f"   Datasets: {len(datasets)}")

    # Create matrix: rows=models, columns=datasets
    score_matrix = []
    model_labels = []

    for model in models:
        # Clean model name
        model_name = model.replace(' (OpenRouter)', '').replace(' (Ollama)', '')
        model_labels.append(model_name)

        row_scores = []
        for dataset in datasets:
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

    # Get category labels
    category_labels = [DATASET_NAME_MAP.get(ds, ds) for ds in datasets]

    if verbose:
        click.echo(f"   Heatmap dimensions: {len(model_labels)} models × {len(category_labels)} categories")
        click.echo(f"   RA score range: {np.nanmin(data):.4f} to {np.nanmax(data):.4f}")

    # Set publication style
    plt.style.use('seaborn-v0_8-white')
    sns.set_context("paper", font_scale=1.1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap with custom colormap (red→yellow→green)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.8)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(category_labels)))
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_xticklabels(category_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(model_labels, rotation=45, ha='right', fontsize=9)

    # Disable grid
    ax.grid(False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relational Affinity (RA)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Annotate cells with values
    for i in range(len(model_labels)):
        for j in range(len(category_labels)):
            val = data[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}',
                             ha="center", va="center", color='black', fontsize=14, fontweight='bold')

    # Labels and title
    lang_name = 'English' if language == 'EN' else 'Chinese' if language == 'ZH' else 'Cross-lingual'
    ax.set_xlabel('Semantic Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Embedding Model', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Figure 3: Relational Affinity Performance Heatmap ({lang_name})',
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

        # Find best and worst categories
        cat_means = np.nanmean(data, axis=0)
        best_cat_idx = np.nanargmax(cat_means)
        worst_cat_idx = np.nanargmin(cat_means)
        click.echo(f"\n   Best category:  {category_labels[best_cat_idx]} (mean RA={cat_means[best_cat_idx]:.4f})")
        click.echo(f"   Worst category: {category_labels[worst_cat_idx]} (mean RA={cat_means[worst_cat_idx]:.4f})")

        # Find best and worst models
        model_means = np.nanmean(data, axis=1)
        best_model_idx = np.nanargmax(model_means)
        worst_model_idx = np.nanargmin(model_means)
        click.echo(f"\n   Best model:  {model_labels[best_model_idx]} (mean RA={model_means[best_model_idx]:.4f})")
        click.echo(f"   Worst model: {model_labels[worst_model_idx]} (mean RA={model_means[worst_model_idx]:.4f})")


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
    language: str,
    dpi: int,
    figsize: tuple,
    verbose: bool
):
    """
    Generate Figure 3: Language-specific full performance heatmap.

    Creates a publication-quality heatmap showing RA scores for a specific
    language (English, Chinese, or cross-lingual) across all 17 models
    and 11 semantic categories.

    Examples:

        # English heatmap
        python generate_figure3_full_heatmap_by_language.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \\
            --output neurips_figure3-full-heatmap-EN.pdf \\
            --language EN

        # Chinese heatmap
        python generate_figure3_full_heatmap_by_language.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \\
            --output neurips_figure3-full-heatmap-ZH.pdf \\
            --language ZH

        # High resolution
        python generate_figure3_full_heatmap_by_language.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \\
            --output neurips_figure3-full-heatmap-EN.pdf \\
            --language EN \\
            --dpi 600
    """
    if verbose:
        lang_name = 'English' if language == 'EN' else 'Chinese' if language == 'ZH' else 'Cross-lingual'
        click.echo("\n" + "=" * 80)
        click.echo(f"FIGURE 3 GENERATION: Full Performance Heatmap ({lang_name})")
        click.echo("=" * 80 + "\n")

    create_full_heatmap(
        data_path=input,
        output_path=output,
        language=language,
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
