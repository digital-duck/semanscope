#!/usr/bin/env python3
"""
Generate Figure 1: Three-Tier Model Performance Structure by Language

Shows model performance tiers on easy datasets (Family Relations, Royalty Hierarchy,
Gendered Occupations, Comparative-Superlative) for a specific language.

Usage:
    python generate_figure1_model_tiers_by_language.py \
        --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \
        --output neurips_figure1_model_tiers_easy_datasets_EN.pdf \
        --language EN \
        [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path


# Easy datasets for Figure 1
EASY_DATASETS = [
    'NeurIPS-01-family-relations',
    'NeurIPS-02-royalty-hierarchy',
    'NeurIPS-03-gendered-occupations',
    'NeurIPS-04-comparative-superlative',
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
    Create three-tier model performance visualization on easy datasets.

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

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create colors based on tiers
    colors = []
    for _, row in result_df.iterrows():
        if row['Avg_RA'] > 0.50:
            colors.append('#2ecc71')  # Green
        elif row['Avg_RA'] >= 0.30:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red

    # Create horizontal bar chart
    y_positions = np.arange(len(result_df))
    bars = ax.barh(y_positions, result_df['Avg_RA'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Set labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(result_df['Model'], fontsize=10)
    ax.invert_yaxis()  # Top model at top

    # Set x-axis
    ax.set_xlabel('Average Relational Affinity (RA) on Easy Datasets', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 0.8)

    # Add tier boundary lines
    ax.axvline(x=0.50, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Tier 1 threshold (0.50)')
    ax.axvline(x=0.30, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Tier 2 threshold (0.30)')

    # Add value labels on bars
    for i, (idx, row) in enumerate(result_df.iterrows()):
        ax.text(row['Avg_RA'] + 0.01, i, f"{row['Avg_RA']:.3f}",
                va='center', ha='left', fontsize=9, fontweight='bold')

    # Add tier labels
    ax.text(0.65, 0.95, 'Tier 1: Strong Performance', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='green', ha='left', va='top')
    ax.text(0.65, 0.90, 'Tier 2: Moderate Performance', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='orange', ha='left', va='top')
    ax.text(0.65, 0.85, 'Tier 3: Weak Performance', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='red', ha='left', va='top')

    # Title
    lang_name = 'English' if language == 'EN' else 'Chinese' if language == 'ZH' else 'Cross-lingual'
    ax.set_title(
        f'Figure 1: Three-Tier Model Performance Structure ({lang_name})',
        fontsize=15,
        fontweight='bold',
        pad=20
    )

    # Grid
    ax.grid(True, alpha=0.3, axis='x')

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

    # Print tier details
    if verbose:
        click.echo(f"\n📊 Tier Details ({lang_name}):")

        if len(tier1) > 0:
            click.echo(f"\n   Tier 1 Models (RA > 0.50):")
            for _, row in tier1.iterrows():
                click.echo(f"      {row['Model']:<50} {row['Avg_RA']:.4f}")

        if len(tier2) > 0:
            click.echo(f"\n   Tier 2 Models (0.30 ≤ RA ≤ 0.50):")
            for _, row in tier2.iterrows():
                click.echo(f"      {row['Model']:<50} {row['Avg_RA']:.4f}")

        if len(tier3) > 0:
            click.echo(f"\n   Tier 3 Models (RA < 0.30):")
            for _, row in tier3.iterrows():
                click.echo(f"      {row['Model']:<50} {row['Avg_RA']:.4f}")


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
    type=click.Choice(['EN', 'ZH', 'cross'], case_sensitive=False),
    help='Language to analyze: EN (English), ZH (Chinese), or cross (cross-lingual)'
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

    Shows model performance on easy datasets (Family Relations, Royalty Hierarchy,
    Gendered Occupations, Comparative-Superlative) for a specific language.

    Examples:

        # English version
        python generate_figure1_model_tiers_by_language.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \\
            --output neurips_figure1_model_tiers_EN.pdf \\
            --language EN

        # Chinese version
        python generate_figure1_model_tiers_by_language.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \\
            --output neurips_figure1_model_tiers_ZH.pdf \\
            --language ZH
    """
    if verbose:
        lang_name = 'English' if language == 'EN' else 'Chinese' if language == 'ZH' else 'Cross-lingual'
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
