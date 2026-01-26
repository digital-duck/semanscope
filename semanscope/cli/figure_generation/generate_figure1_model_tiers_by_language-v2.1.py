#!/usr/bin/env python3
"""
Generate Figure 1: Three-Tier Model Performance Structure by Language
Version 2.1 - Preserves exact plotting style from original optimized version

Shows model performance tiers on easy datasets for a specific language.
Uses VERTICAL bars with colored background regions (optimized style).

Usage:
    python generate_figure1_model_tiers_by_language-v2.1.py \
        --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \
        --output neurips_figure1_model_tiers_EN.pdf \
        --language EN \
        [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path


# Easy datasets for Figure 1 (v2.5 dataset names)
EASY_DATASETS = [
    'NeurIPS-01-family-relations-v2.5',
    'NeurIPS-02-royalty-hierarchy-v2.5',
    'NeurIPS-03-gendered-occupations-v2.5',
    'NeurIPS-04-comparative-superlative-v2.5',
]


def create_model_tiers_figure(
    data_path: str,
    output_path: str,
    language: str,  # 'EN', 'ZH', or 'cross'
    figsize: tuple = (14, 8),
    dpi: int = 300,
    verbose: bool = True
):
    """
    Create three-tier model performance visualization (exact style from original).

    Args:
        data_path: Path to RA raw CSV
        output_path: Path to save figure
        language: 'EN', 'ZH', or 'cross'
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output
        verbose: Print progress messages
    """
    if verbose:
        lang_name = {'EN': 'English', 'ZH': 'Chinese', 'ES': 'Spanish', 'cross': 'Cross-lingual'}.get(language, language)
        click.echo(f"📊 Generating Figure 1 ({lang_name}): Three-Tier Model Performance")
        click.echo(f"   Reading data: {data_path}")

    # Read raw RA data
    df = pd.read_csv(data_path)

    # Select RA column based on language
    ra_col = f'cos_{language}'

    if verbose:
        click.echo(f"   Using RA column: {ra_col}")
        click.echo(f"   Easy datasets: {', '.join(EASY_DATASETS)}")

    # Calculate average RA for each model on easy datasets
    models = df['model'].unique().tolist()
    model_avg_ra = []

    for model in models:
        model_data = df[df['model'] == model]
        easy_data = model_data[model_data['dataset'].isin(EASY_DATASETS)]

        if len(easy_data) > 0:
            avg_ra = easy_data[ra_col].mean()
            model_avg_ra.append({
                'Model': model.replace(' (OpenRouter)', '').replace(' (Ollama)', ''),
                'Avg_RA': avg_ra
            })

    # Create dataframe and sort
    result_df = pd.DataFrame(model_avg_ra).sort_values('Avg_RA', ascending=False)

    if verbose:
        click.echo(f"   Models: {len(result_df)}")
        click.echo(f"   Average RA range: {result_df['Avg_RA'].min():.4f} to {result_df['Avg_RA'].max():.4f}")

    # Define tiers
    tier1 = result_df[result_df['Avg_RA'] > 0.50]
    tier2 = result_df[(result_df['Avg_RA'] >= 0.30) & (result_df['Avg_RA'] <= 0.50)]
    tier3 = result_df[result_df['Avg_RA'] < 0.30]

    if verbose:
        click.echo(f"\n   Tier 1 (RA > 0.50): {len(tier1)} models")
        click.echo(f"   Tier 2 (0.30 ≤ RA ≤ 0.50): {len(tier2)} models")
        click.echo(f"   Tier 3 (RA < 0.30): {len(tier3)} models")

    # Set publication style - EXACT MATCH TO ORIGINAL
    plt.style.use('seaborn-v0_8-white')
    sns.set_context("paper", font_scale=1.2)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Add colored background regions for tiers - EXACT MATCH TO ORIGINAL
    y_max = 0.70
    ax.axhspan(0.50, y_max, facecolor='lightgreen', alpha=0.3, label='Tier 1: Top BERT')
    ax.axhspan(0.30, 0.50, facecolor='lightblue', alpha=0.3, label='Tier 2: LLM Plateau')
    ax.axhspan(0, 0.30, facecolor='lightcoral', alpha=0.3, label='Tier 3: Failed')

    # Add horizontal threshold lines - EXACT MATCH TO ORIGINAL
    ax.axhline(y=0.50, color='darkgreen', linestyle='--', linewidth=2,
               alpha=0.6, label='RA = 0.50 (Top threshold)')
    ax.axhline(y=0.30, color='darkred', linestyle='--', linewidth=2,
               alpha=0.6, label='RA = 0.30 (Failure threshold)')

    # Create VERTICAL bar chart - EXACT MATCH TO ORIGINAL
    x_positions = np.arange(len(result_df))
    colors = []

    for ra in result_df['Avg_RA']:
        if ra > 0.50:
            colors.append('#2ca02c')  # Green - Tier 1
        elif ra >= 0.30:
            colors.append('#5588dd')  # Blue - Tier 2
        else:
            colors.append('#e74c3c')  # Red - Tier 3

    bars = ax.bar(x_positions, result_df['Avg_RA'], color=colors,
                  alpha=0.8, edgecolor='black', linewidth=0.8)

    # Annotate bars with values on top - EXACT MATCH TO ORIGINAL
    for i, (idx, row) in enumerate(result_df.iterrows()):
        ax.text(i, row['Avg_RA'] + 0.01, f"{row['Avg_RA']:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Labels and formatting - EXACT MATCH TO ORIGINAL
    ax.set_ylabel('Average RA Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(result_df['Model'], rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0, y_max)
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=13, frameon=True, shadow=True,
              markerscale=1.4, handlelength=2.5, borderpad=1.2)

    # Title - adapted for language
    lang_name = {'EN': 'English', 'ZH': 'Chinese', 'ES': 'Spanish', 'cross': 'Cross-lingual'}.get(language, language)
    ax.set_title(f'Figure 3: Three-Tier Performance Structure ({lang_name})',
                 fontsize=15, fontweight='bold', pad=15)

    # Adjust layout
    plt.tight_layout()

    # Save figure - EXACT MATCH TO ORIGINAL
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=dpi)

    if verbose:
        click.echo(f"\n   ✅ Figure saved: {output_path}")
        click.echo(f"   Resolution: {dpi} DPI")
        click.echo(f"   Size: {figsize[0]}\" × {figsize[1]}\"")

    # Save PNG version
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150)

    if verbose:
        click.echo(f"   ✅ Preview saved: {png_path}")

    plt.close()

    # Print tier details
    if verbose:
        click.echo(f"\n📊 Tier Breakdown:")

        click.echo(f"\n   Tier 1 (Top Performers, RA > 0.50): {len(tier1)} models")
        for _, row in tier1.iterrows():
            click.echo(f"      {row['Model']}: {row['Avg_RA']:.4f}")

        click.echo(f"\n   Tier 2 (Moderate Performers, 0.30 ≤ RA ≤ 0.50): {len(tier2)} models")
        for _, row in tier2.iterrows():
            click.echo(f"      {row['Model']}: {row['Avg_RA']:.4f}")

        click.echo(f"\n   Tier 3 (Failed Models, RA < 0.30): {len(tier3)} models")
        for _, row in tier3.iterrows():
            click.echo(f"      {row['Model']}: {row['Avg_RA']:.4f}")


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
    default=(14, 8),
    type=(float, float),
    help='Figure size in inches (width, height) (default: 14 8)'
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
    Generate Figure 1: Three-tier model performance structure.
    Version 2.1 - Preserves exact plotting style from original.

    Examples:

        # English version
        python generate_figure1_model_tiers_by_language-v2.1.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \\
            --output neurips_figure1_model_tiers_EN.pdf \\
            --language EN
    """
    if verbose:
        lang_name = {'EN': 'English', 'ZH': 'Chinese', 'ES': 'Spanish', 'cross': 'Cross-lingual'}.get(language, language)
        click.echo("\n" + "=" * 80)
        click.echo(f"FIGURE 1 GENERATION: Three-Tier Model Performance ({lang_name})")
        click.echo("=" * 80 + "\n")

    create_model_tiers_figure(
        data_path=input,
        output_path=output,
        language=language,
        figsize=figsize,
        dpi=dpi,
        verbose=verbose
    )

    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("✅ Figure 1 generation complete!")
        click.echo(f"   Main figure: {output}")
        click.echo(f"   Preview: {Path(output).with_suffix('.png')}")
        click.echo("=" * 80 + "\n")


if __name__ == '__main__':
    main()
