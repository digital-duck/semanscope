#!/usr/bin/env python3
"""
Generate Figure 5: SA vs RA Scatter Plots by Language (EN and ZH separately)

Creates two separate scatter plots:
- SA_EN vs RA_EN (English)
- SA_ZH vs RA_ZH (Chinese)

This allows comparison of how semantic and relational affinity correlate
within each language.

Usage:
    python generate_figure5_sa_ra_scatter_by_language.py \
        --input sa-ra-aggregated-2025-12-29.csv \
        --output-en neurips_figure5_sa_ra_scatter_EN.pdf \
        --output-zh neurips_figure5_sa_ra_scatter_ZH.pdf \
        [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path
from scipy.stats import pearsonr


# Dataset name mapping for legend
DATASET_NAMES = {
    'NeurIPS-01-family-relations': 'Family Relations',
    'NeurIPS-02-royalty-hierarchy': 'Royalty Hierarchy',
    'NeurIPS-03-gendered-occupations': 'Gendered Occupations',
    'NeurIPS-04-comparative-superlative': 'Comparative/Superlative',
    'NeurIPS-05-opposite-relations': 'Opposite Relations',
    'NeurIPS-06-animal-gender': 'Animal Gender',
    'NeurIPS-07-sequential-mathematical': 'Sequential Math',
    'NeurIPS-08-hierarchical-relations': 'Hierarchical Relations',
    'NeurIPS-09-part-whole': 'Part-Whole',
    'NeurIPS-10-size-scale': 'Size/Scale',
    'NeurIPS-11-cause-effect': 'Cause-Effect',
    'NeurIPS-07-math-numbers': 'Math Numbers',
}

# Dataset colors and markers
DATASET_COLORS = {
    'NeurIPS-01-family-relations': '#1f77b4',
    'NeurIPS-02-royalty-hierarchy': '#ff7f0e',
    'NeurIPS-03-gendered-occupations': '#2ca02c',
    'NeurIPS-04-comparative-superlative': '#d62728',
    'NeurIPS-05-opposite-relations': '#9467bd',
    'NeurIPS-06-animal-gender': '#8c564b',
    'NeurIPS-07-sequential-mathematical': '#e377c2',
    'NeurIPS-08-hierarchical-relations': '#7f7f7f',
    'NeurIPS-09-part-whole': '#bcbd22',
    'NeurIPS-10-size-scale': '#17becf',
    'NeurIPS-11-cause-effect': '#ff9896',
    'NeurIPS-07-math-numbers': '#c5b0d5',
}


def create_scatter_plot(
    data_path: str,
    output_path: str,
    language: str,  # 'EN' or 'ZH'
    sa_metric: str = 'sa_cos',  # 'sa_cos', 'sa_cos_intra', or 'sa_cos_inter'
    figsize: tuple = (10, 8),
    dpi: int = 300,
    sa_threshold: float = 0.5,
    ra_threshold: float = 0.3,
    verbose: bool = True
):
    """
    Create SA vs RA scatter plot for a specific language.

    Args:
        data_path: Path to aggregated SA+RA CSV
        output_path: Path to save figure
        language: 'EN' or 'ZH'
        sa_metric: SA column to use ('sa_cos', 'sa_cos_intra', or 'sa_cos_inter')
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output
        sa_threshold: Threshold for "high" SA (quadrant boundary)
        ra_threshold: Threshold for "high" RA (quadrant boundary)
        verbose: Print progress messages
    """
    if verbose:
        click.echo(f"📊 Generating Figure 5 ({language}): SA vs RA Scatter Plot")
        click.echo(f"   Reading data: {data_path}")

    # Read aggregated data
    df = pd.read_csv(data_path)

    # Select columns based on language
    ra_col = f'ra_cos_{language}'
    sa_col = sa_metric  # SA doesn't have language-specific columns

    # Drop rows with missing values
    df_clean = df[[ra_col, sa_col, 'dataset', 'model']].dropna()

    if verbose:
        click.echo(f"   Total data points: {len(df_clean)}")
        click.echo(f"   Using columns: {ra_col}, {sa_col}")

    # Calculate correlation
    if len(df_clean) > 1:
        corr, p_value = pearsonr(df_clean[ra_col], df_clean[sa_col])
        if verbose:
            click.echo(f"   Pearson correlation: r = {corr:.4f} (p = {p_value:.4e})")
    else:
        corr = np.nan
        p_value = np.nan

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.3)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot quadrant backgrounds (subtle)
    ax.axhspan(sa_threshold, 1.0, xmin=0, xmax=ra_threshold/1.0,
               alpha=0.05, color='yellow', zorder=0, label='_nolegend_')  # Quadrant II
    ax.axhspan(sa_threshold, 1.0, xmin=ra_threshold/1.0, xmax=1.0,
               alpha=0.05, color='green', zorder=0, label='_nolegend_')   # Quadrant I
    ax.axhspan(0, sa_threshold, xmin=ra_threshold/1.0, xmax=1.0,
               alpha=0.05, color='gray', zorder=0, label='_nolegend_')    # Quadrant III
    ax.axhspan(0, sa_threshold, xmin=0, xmax=ra_threshold/1.0,
               alpha=0.05, color='red', zorder=0, label='_nolegend_')     # Quadrant IV

    # Plot data points grouped by dataset
    for dataset, color in DATASET_COLORS.items():
        dataset_data = df_clean[df_clean['dataset'] == dataset]
        if len(dataset_data) > 0:
            ax.scatter(
                dataset_data[ra_col],
                dataset_data[sa_col],
                c=color,
                label=DATASET_NAMES.get(dataset, dataset),
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                zorder=3
            )

    # Add quadrant boundary lines
    ax.axhline(y=sa_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)
    ax.axvline(x=ra_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)

    # Add diagonal line (y=x) to show where SA=RA
    max_val = max(df_clean[sa_col].max(), df_clean[ra_col].max())
    ax.plot([0, max_val], [0, max_val], 'k:', linewidth=1, alpha=0.3, label='SA = RA', zorder=1)

    # Set labels and title
    lang_name = 'English' if language == 'EN' else 'Chinese'
    ax.set_xlabel(f'Relational Affinity (RA) - {lang_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Semantic Affinity (SA) - {lang_name}', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Figure 5: Semantic vs Relational Affinity ({lang_name})',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Add correlation text
    if not np.isnan(corr):
        ax.text(
            0.05, 0.95,
            f'r = {corr:.3f}\n(p = {p_value:.2e})',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        )

    # Add quadrant labels
    ax.text(0.75, 0.95, 'I: Coherent', transform=ax.transAxes, fontsize=10,
            ha='center', va='top', color='darkgreen', fontweight='bold')
    ax.text(0.25, 0.95, 'II: Incoherent', transform=ax.transAxes, fontsize=10,
            ha='center', va='top', color='orange', fontweight='bold')
    ax.text(0.75, 0.05, 'III: Unusual', transform=ax.transAxes, fontsize=10,
            ha='center', va='bottom', color='gray', fontweight='bold')
    ax.text(0.25, 0.05, 'IV: Poor', transform=ax.transAxes, fontsize=10,
            ha='center', va='bottom', color='darkred', fontweight='bold')

    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, frameon=True, shadow=True)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Adjust layout
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

    # Print quadrant statistics
    if verbose:
        click.echo(f"\n📊 Quadrant Statistics ({lang_name}):")
        q1 = len(df_clean[(df_clean[sa_col] >= sa_threshold) & (df_clean[ra_col] >= ra_threshold)])
        q2 = len(df_clean[(df_clean[sa_col] >= sa_threshold) & (df_clean[ra_col] < ra_threshold)])
        q3 = len(df_clean[(df_clean[sa_col] < sa_threshold) & (df_clean[ra_col] >= ra_threshold)])
        q4 = len(df_clean[(df_clean[sa_col] < sa_threshold) & (df_clean[ra_col] < ra_threshold)])
        total = len(df_clean)

        click.echo(f"   Quadrant I  (High SA, High RA - Coherent):   {q1:>4} ({q1/total*100:>5.1f}%)")
        click.echo(f"   Quadrant II (High SA, Low RA - Incoherent):  {q2:>4} ({q2/total*100:>5.1f}%) ⚠️")
        click.echo(f"   Quadrant III (Low SA, High RA - Unusual):    {q3:>4} ({q3/total*100:>5.1f}%)")
        click.echo(f"   Quadrant IV (Low SA, Low RA - Poor):         {q4:>4} ({q4/total*100:>5.1f}%)")


@click.command()
@click.option(
    '--input',
    '-i',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to aggregated SA+RA CSV file'
)
@click.option(
    '--output-en',
    required=True,
    type=click.Path(dir_okay=False),
    help='Path to save English scatter plot (PDF recommended)'
)
@click.option(
    '--output-zh',
    required=True,
    type=click.Path(dir_okay=False),
    help='Path to save Chinese scatter plot (PDF recommended)'
)
@click.option(
    '--dpi',
    default=300,
    type=int,
    help='Resolution for output figure (default: 300)'
)
@click.option(
    '--figsize',
    default=(10, 8),
    type=(float, float),
    help='Figure size in inches (width, height) (default: 10 8)'
)
@click.option(
    '--sa-threshold',
    default=0.5,
    type=float,
    help='Threshold for "high" SA (default: 0.5)'
)
@click.option(
    '--ra-threshold',
    default=0.3,
    type=float,
    help='Threshold for "high" RA (default: 0.3)'
)
@click.option(
    '--sa-metric',
    default='sa_cos',
    type=click.Choice(['sa_cos', 'sa_cos_intra', 'sa_cos_inter'], case_sensitive=False),
    help='SA metric to use: sa_cos (overall), sa_cos_intra (within-language), or sa_cos_inter (cross-language) (default: sa_cos)'
)
@click.option(
    '--verbose/--quiet',
    '-v/-q',
    default=True,
    help='Print progress messages'
)
def main(
    input: str,
    output_en: str,
    output_zh: str,
    dpi: int,
    figsize: tuple,
    sa_threshold: float,
    ra_threshold: float,
    sa_metric: str,
    verbose: bool
):
    """
    Generate SA vs RA scatter plots separately for English and Chinese.

    Examples:

        # Basic usage
        python generate_figure5_sa_ra_scatter_by_language.py \\
            --input sa-ra-aggregated-2025-12-29.csv \\
            --output-en neurips_figure5_sa_ra_scatter_EN.pdf \\
            --output-zh neurips_figure5_sa_ra_scatter_ZH.pdf

        # High resolution
        python generate_figure5_sa_ra_scatter_by_language.py \\
            --input sa-ra-aggregated-2025-12-29.csv \\
            --output-en neurips_figure5_sa_ra_scatter_EN.pdf \\
            --output-zh neurips_figure5_sa_ra_scatter_ZH.pdf \\
            --dpi 600
    """
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("FIGURE 5 GENERATION: SA vs RA Scatter Plots by Language")
        click.echo("=" * 80 + "\n")

    # Generate English plot
    create_scatter_plot(
        data_path=input,
        output_path=output_en,
        language='EN',
        sa_metric=sa_metric,
        figsize=figsize,
        dpi=dpi,
        sa_threshold=sa_threshold,
        ra_threshold=ra_threshold,
        verbose=verbose
    )

    if verbose:
        click.echo("\n" + "-" * 80 + "\n")

    # Generate Chinese plot
    create_scatter_plot(
        data_path=input,
        output_path=output_zh,
        language='ZH',
        sa_metric=sa_metric,
        figsize=figsize,
        dpi=dpi,
        sa_threshold=sa_threshold,
        ra_threshold=ra_threshold,
        verbose=verbose
    )

    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("✅ Figure 5 generation complete!")
        click.echo(f"   English plot: {output_en}")
        click.echo(f"   Chinese plot: {output_zh}")
        click.echo("=" * 80 + "\n")


if __name__ == '__main__':
    main()
