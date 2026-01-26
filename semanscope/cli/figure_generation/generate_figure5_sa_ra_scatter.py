#!/usr/bin/env python3
"""
Generate Figure 5: SA vs RA Scatter Plot for NeurIPS Paper

This script creates a publication-quality scatter plot showing the relationship
between Semantic Affinity (SA) and Relational Affinity (RA) across 17 models
and 11 semantic categories, highlighting structural incoherence (Quadrant II).

Usage:
    python generate_figure5_sa_ra_scatter.py \\
        --input quadrant-analysis.csv \\
        --output figure5-sa-ra-scatter.pdf \\
        [--dpi 300] \\
        [--figsize 12 10]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Color palette for datasets (11 distinct colors)
DATASET_COLORS = {
    'DS1': '#1f77b4',  # Blue - Family (easy)
    'DS2': '#ff7f0e',  # Orange - Royalty (easy)
    'DS3': '#2ca02c',  # Green - Gendered Occ (easy)
    'DS4': '#d62728',  # Red - Comparative (easy)
    'DS5': '#9467bd',  # Purple - Opposites (hard)
    'DS6': '#8c564b',  # Brown - Animal Gender (hard)
    'DS7': '#e377c2',  # Pink - Sequential-Math (VERY hard)
    'DS8': '#7f7f7f',  # Gray - Hierarchical (hard)
    'DS9': '#bcbd22',  # Yellow-green - Part-Whole (VERY hard)
    'DS10': '#17becf', # Cyan - Size-Scale (hard)
    'DS11': '#ff0000', # Bright Red - Cause-Effect (VERY hard)
}


def create_sa_ra_scatter(
    data_path: str,
    output_path: str,
    figsize: tuple = (14, 10),
    dpi: int = 300,
    sa_threshold: float = 0.5,
    ra_threshold: float = 0.3,
    annotate_key_cases: bool = True,
    show_correlation: bool = True,
    verbose: bool = True
):
    """
    Create SA vs RA scatter plot with quadrant highlighting.

    Args:
        data_path: Path to quadrant analysis CSV
        output_path: Path to save figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output
        sa_threshold: Threshold line for "high" SA
        ra_threshold: Threshold line for "high" RA
        annotate_key_cases: Annotate extreme Quadrant II cases
        show_correlation: Show correlation coefficient
        verbose: Print progress messages
    """
    if verbose:
        click.echo(f"📊 Generating Figure 5: SA vs RA Scatter Plot")
        click.echo(f"   Reading data: {data_path}")

    # Read data
    df = pd.read_csv(data_path)

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
        dataset_data = df[df['dataset'] == dataset]
        if len(dataset_data) > 0:
            ax.scatter(
                dataset_data['ra_score'],
                dataset_data['sa_score'],
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
    max_val = max(df['sa_score'].max(), df['ra_score'].max())
    ax.plot([0, max_val], [0, max_val], 'k:', linewidth=1, alpha=0.3, label='SA = RA', zorder=1)

    # Annotate key Quadrant II cases (highest gaps)
    if annotate_key_cases:
        quadrant_ii = df[
            (df['sa_score'] >= sa_threshold) &
            (df['ra_score'] < ra_threshold)
        ].copy()
        quadrant_ii['gap'] = quadrant_ii['sa_score'] - quadrant_ii['ra_score']
        top_cases = quadrant_ii.nlargest(5, 'gap')

        for idx, row in top_cases.iterrows():
            ax.annotate(
                f"{row['dataset']}",
                xy=(row['ra_score'], row['sa_score']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1),
                zorder=4
            )

    # Calculate and display correlation
    if show_correlation:
        corr = df['sa_score'].corr(df['ra_score'])
        ax.text(
            0.05, 0.95,
            f'r = {corr:.3f}\nN = {len(df)}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        )

    # Add quadrant labels
    ax.text(0.85, 0.95, 'Quadrant I\n(Coherent)', transform=ax.transAxes,
            fontsize=11, ha='center', va='top', style='italic', alpha=0.6)
    ax.text(0.15, 0.95, 'Quadrant II\n(Incoherent)', transform=ax.transAxes,
            fontsize=11, ha='center', va='top', style='italic', alpha=0.6,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.text(0.85, 0.15, 'Quadrant III\n(Unusual)', transform=ax.transAxes,
            fontsize=11, ha='center', va='top', style='italic', alpha=0.6)
    ax.text(0.15, 0.15, 'Quadrant IV\n(Poor)', transform=ax.transAxes,
            fontsize=11, ha='center', va='top', style='italic', alpha=0.6)

    # Labels and title
    ax.set_xlabel('Relational Affinity (RA)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Semantic Affinity (SA)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Figure 5: Semantic vs Relational Affinity Across 17 Models and 11 Categories\n'
        'Exposing Structural Incoherence in Embedding Models',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Set axis limits with padding
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Grid
    ax.grid(True, alpha=0.3, zorder=0)

    # Legend (split into two columns for readability)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=10,
        frameon=True,
        shadow=True,
        title='Semantic Categories',
        title_fontsize=11
    )

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    if verbose:
        click.echo(f"   ✅ Figure saved: {output_path}")
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
        click.echo(f"\n📊 Quadrant Summary:")
        q1 = len(df[(df['sa_score'] >= sa_threshold) & (df['ra_score'] >= ra_threshold)])
        q2 = len(df[(df['sa_score'] >= sa_threshold) & (df['ra_score'] < ra_threshold)])
        q3 = len(df[(df['sa_score'] < sa_threshold) & (df['ra_score'] >= ra_threshold)])
        q4 = len(df[(df['sa_score'] < sa_threshold) & (df['ra_score'] < ra_threshold)])
        total = len(df)

        click.echo(f"   Quadrant I  (High SA, High RA): {q1:3} ({q1/total*100:5.1f}%) - Structurally Coherent")
        click.echo(f"   Quadrant II (High SA, Low RA):  {q2:3} ({q2/total*100:5.1f}%) - Structurally Incoherent ⚠️")
        click.echo(f"   Quadrant III (Low SA, High RA):  {q3:3} ({q3/total*100:5.1f}%) - Unusual")
        click.echo(f"   Quadrant IV (Low SA, Low RA):   {q4:3} ({q4/total*100:5.1f}%) - Universally Poor")


@click.command()
@click.option(
    '--input',
    '-i',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to quadrant analysis CSV file'
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
    '--annotate/--no-annotate',
    default=True,
    help='Annotate key Quadrant II cases (default: True)'
)
@click.option(
    '--correlation/--no-correlation',
    default=True,
    help='Show correlation coefficient (default: True)'
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
    sa_threshold: float,
    ra_threshold: float,
    annotate: bool,
    correlation: bool,
    verbose: bool
):
    """
    Generate Figure 5: SA vs RA scatter plot for NeurIPS paper.

    This script creates a publication-quality visualization showing the
    relationship between Semantic Affinity (SA) and Relational Affinity (RA)
    across 17 embedding models and 11 semantic categories.

    The plot highlights Quadrant II (High SA, Low RA) cases which demonstrate
    structural incoherence - models that cluster words semantically but fail
    to preserve relational structure.

    Examples:

        # Basic usage (generate PDF at 300 DPI)
        python generate_figure5_sa_ra_scatter.py \\
            --input quadrant-analysis-2025-12-29.csv \\
            --output figure5-sa-ra-scatter.pdf

        # High-resolution for publication (600 DPI)
        python generate_figure5_sa_ra_scatter.py \\
            --input quadrant-analysis-2025-12-29.csv \\
            --output figure5-sa-ra-scatter.pdf \\
            --dpi 600

        # Custom thresholds
        python generate_figure5_sa_ra_scatter.py \\
            --input quadrant-analysis-2025-12-29.csv \\
            --output figure5-sa-ra-scatter.pdf \\
            --sa-threshold 0.6 \\
            --ra-threshold 0.2
    """
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("FIGURE 5 GENERATION: SA vs RA Scatter Plot")
        click.echo("=" * 80 + "\n")

    create_sa_ra_scatter(
        data_path=input,
        output_path=output,
        figsize=figsize,
        dpi=dpi,
        sa_threshold=sa_threshold,
        ra_threshold=ra_threshold,
        annotate_key_cases=annotate,
        show_correlation=correlation,
        verbose=verbose
    )

    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("✅ Figure 5 generation complete!")
        click.echo(f"   Main figure: {output}")
        click.echo(f"   Preview: {Path(output).with_suffix('.png')}")
        click.echo("=" * 80 + "\n")


if __name__ == '__main__':
    main()
