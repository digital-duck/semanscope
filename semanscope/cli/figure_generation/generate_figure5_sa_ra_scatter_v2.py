#!/usr/bin/env python3
"""
Generate Figure 5: SA vs RA Scatter Plot for NeurIPS Paper (Version 2)

Uses markers for datasets and color gradient (red→green) for models.

Usage:
    python generate_figure5_sa_ra_scatter_v2.py \\
        --input quadrant-analysis.csv \\
        --output figure5-sa-ra-scatter.pdf \\
        [--dpi 300] \\
        [--figsize 16 10]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
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

# Markers for datasets (11 distinct shapes)
DATASET_MARKERS = {
    'DS1': 'o',      # Circle
    'DS2': 's',      # Square
    'DS3': '^',      # Triangle up
    'DS4': 'v',      # Triangle down
    'DS5': 'D',      # Diamond
    'DS6': 'P',      # Plus (filled)
    'DS7': 'X',      # X (filled)
    'DS8': '*',      # Star
    'DS9': '>',      # Triangle right
    'DS10': 'h',     # Hexagon
    'DS11': '<',     # Triangle left
}


def create_sa_ra_scatter(
    data_path: str,
    output_path: str,
    figsize: tuple = (16, 10),
    dpi: int = 300,
    sa_threshold: float = 0.5,
    ra_threshold: float = 0.3,
    annotate_key_cases: bool = True,
    show_correlation: bool = True,
    verbose: bool = True
):
    """
    Create SA vs RA scatter plot with markers for datasets and color gradient for models.

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
        click.echo(f"📊 Generating Figure 5: SA vs RA Scatter Plot (v2)")
        click.echo(f"   Reading data: {data_path}")

    # Read data
    df = pd.read_csv(data_path)

    # Rank models by average RA performance (for color assignment)
    model_avg_ra = df.groupby('model')['ra_score'].mean().sort_values()
    model_rank = {model: i for i, model in enumerate(model_avg_ra.index)}

    if verbose:
        click.echo(f"   Models ranked by avg RA (worst to best):")
        for i, (model, avg_ra) in enumerate(model_avg_ra.items()):
            click.echo(f"      {i+1:2}. {model:<45} RA={avg_ra:.4f}")

    # Create red→yellow→green colormap for models
    n_models = len(model_rank)
    cmap = LinearSegmentedColormap.from_list('performance',
                                             ['#d62728', '#ff7f0e', '#ffff00', '#90ee90', '#2ca02c'])
    model_colors = {model: cmap(rank / (n_models - 1)) for model, rank in model_rank.items()}

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)

    # Create figure with more space for dual legends
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot quadrant backgrounds (subtle) - counter-clockwise: I, II, III, IV
    # Calculate xmin/xmax fractions based on new axis limits (-0.025 to 0.75)
    x_range = 0.75 - (-0.025)
    ra_threshold_frac = (ra_threshold - (-0.025)) / x_range

    ax.axhspan(sa_threshold, 1.0, xmin=0, xmax=ra_threshold_frac,
               alpha=0.05, color='yellow', zorder=0, label='_nolegend_')  # Quadrant II (top-left)
    ax.axhspan(sa_threshold, 1.0, xmin=ra_threshold_frac, xmax=1.0,
               alpha=0.05, color='green', zorder=0, label='_nolegend_')   # Quadrant I (top-right)
    ax.axhspan(0, sa_threshold, xmin=0, xmax=ra_threshold_frac,
               alpha=0.05, color='red', zorder=0, label='_nolegend_')     # Quadrant III (bottom-left)
    ax.axhspan(0, sa_threshold, xmin=ra_threshold_frac, xmax=1.0,
               alpha=0.05, color='gray', zorder=0, label='_nolegend_')    # Quadrant IV (bottom-right)

    # Plot data points: iterate through each row
    for idx, row in df.iterrows():
        model = row['model']
        dataset = row['dataset']
        ra = row['ra_score']
        sa = row['sa_score']

        marker = DATASET_MARKERS.get(dataset, 'o')
        color = model_colors.get(model, 'gray')

        # Increase marker size for star (DS8) and hexagon (DS10) for better visibility
        marker_size = 120
        if dataset == 'DS8':  # Hierarchical Relations (star)
            marker_size = 180
        elif dataset == 'DS10':  # Size-Scale (hexagon)
            marker_size = 180

        ax.scatter(
            ra, sa,
            marker=marker,
            c=[color],
            s=marker_size,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
            zorder=3
        )

    # Add quadrant boundary lines
    ax.axhline(y=sa_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)
    ax.axvline(x=ra_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)

    # Annotate key Quadrant II cases (highest gaps) - exclude DS8 and DS11 (too crowded)
    if annotate_key_cases:
        quadrant_ii = df[
            (df['sa_score'] >= sa_threshold) &
            (df['ra_score'] < ra_threshold)
        ].copy()
        quadrant_ii['gap'] = quadrant_ii['sa_score'] - quadrant_ii['ra_score']
        top_cases = quadrant_ii.nlargest(5, 'gap')

        for idx, row in top_cases.iterrows():
            # Skip DS8 and DS11 to avoid overlapping annotations
            if row['dataset'] in ['DS8', 'DS11']:
                continue
            ax.annotate(
                f"{row['dataset']}",
                xy=(row['ra_score'], row['sa_score']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=0.8),
                zorder=4
            )

    # Calculate and display correlation
    if show_correlation:
        corr = df['sa_score'].corr(df['ra_score'])
        ax.text(
            0.35, 0.85,
            f'r = {corr:.3f}\nN = {len(df)}',
            fontsize=18,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.95,
                     edgecolor='black', linewidth=2)
        )

    # Add quadrant labels (counter-clockwise: I=top-right, II=top-left, III=bottom-left, IV=bottom-right)
    # Quadrant I: positioned at right-upper corner with light green background
    ax.text(0.98, 0.98, 'Quadrant I\n(Coherent)', transform=ax.transAxes,
            fontsize=13, ha='right', va='top', style='italic', alpha=0.7, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Quadrant II: positioned at left-upper corner with yellow background
    ax.text(0.02, 0.98, 'Quadrant II\n(Incoherent)', transform=ax.transAxes,
            fontsize=13, ha='left', va='top', style='italic', alpha=0.7, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Quadrant III: positioned at left-lower corner with pinkish background
    ax.text(0.02, 0.08, 'Quadrant III\n(Poor)', transform=ax.transAxes,
            fontsize=13, ha='left', va='bottom', style='italic', alpha=0.7, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ffcccb', alpha=0.3))

    # Quadrant IV: hidden behind model legend (no label)

    # Labels and title
    ax.set_xlabel('Relational Affinity (RA)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Affinity (SA)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Figure 5: Semantic vs Relational Affinity Across 17 Models and 11 Categories\n'
        'Exposing Structural Incoherence in Embedding Models',
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    # Set axis limits with padding
    ax.set_xlim(-0.025, 0.75)  # Optimized for data range
    ax.set_ylim(0.05, 0.9)

    # Grid
    ax.grid(True, alpha=0.3, zorder=0)

    # Create DATASET legend (markers) - LEFT side
    dataset_handles = []
    for ds_id in sorted(DATASET_MARKERS.keys()):
        # Increase marker size for star (DS8) and hexagon (DS10) in legend too
        marker_size = 8
        if ds_id == 'DS8':  # Hierarchical Relations (star)
            marker_size = 11
        elif ds_id == 'DS10':  # Size-Scale (hexagon)
            marker_size = 11

        handle = mlines.Line2D(
            [], [],
            marker=DATASET_MARKERS[ds_id],
            color='gray',
            markerfacecolor='gray',
            markersize=marker_size,
            linestyle='None',
            markeredgecolor='black',
            markeredgewidth=0.5,
            label=DATASET_NAMES.get(ds_id, ds_id)
        )
        dataset_handles.append(handle)

    legend1 = ax.legend(
        handles=dataset_handles,
        loc='lower left',
        bbox_to_anchor=(0.355, 0.02),
        ncol=2,
        fontsize=14,
        frameon=True,
        shadow=True,
        title='Datasets (Markers)',
        title_fontsize=16,
        framealpha=0.95
    )
    ax.add_artist(legend1)

    # Create MODEL legend (colors) - bottom area near X-axis at RA=0.57, showing gradient
    model_handles = []
    # Show top 7 and bottom 7 models (reversed: best at top, worst at bottom)
    models_to_show = list(model_avg_ra.index[-7:][::-1]) + list(model_avg_ra.index[:7][::-1])

    for model in models_to_show:
        avg_ra = model_avg_ra[model]
        # Shorten model names more aggressively for compact legend
        short_name = model.replace('OpenAI ', '').replace('(OpenRouter)', '').replace('(Ollama)', '').strip()
        short_name = short_name[:22] + '...' if len(short_name) > 22 else short_name
        handle = mlines.Line2D(
            [], [],
            marker='o',
            color='w',
            markerfacecolor=model_colors[model],
            markersize=11,
            linestyle='None',
            markeredgecolor='black',
            markeredgewidth=0.5,
            label=f"{short_name} ({avg_ra:.2f})"
        )
        model_handles.append(handle)

    legend2 = ax.legend(
        handles=model_handles,
        loc='lower left',
        bbox_to_anchor=(0.768, 0.02),
        ncol=1,
        fontsize=12,
        frameon=True,
        shadow=True,
        title='Models (Red→Green)\nTop 7 & Bottom 7 by RA',
        title_fontsize=14,
        framealpha=0.95
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
    default=(16, 10),
    type=(float, float),
    help='Figure size in inches (width, height) (default: 16 10)'
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
    Generate Figure 5: SA vs RA scatter plot (Version 2).

    Uses markers for datasets (11 shapes) and color gradient (red→green) for
    models ranked by average RA performance.

    Examples:

        # Basic usage
        python generate_figure5_sa_ra_scatter_v2.py \\
            --input quadrant-analysis-2025-12-29.csv \\
            --output figure5-sa-ra-scatter.pdf

        # High resolution
        python generate_figure5_sa_ra_scatter_v2.py \\
            --input quadrant-analysis-2025-12-29.csv \\
            --output figure5-sa-ra-scatter.pdf \\
            --dpi 600
    """
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("FIGURE 5 GENERATION (v2): SA vs RA Scatter Plot")
        click.echo("  - Markers: Datasets (11 shapes)")
        click.echo("  - Colors: Models (red→green gradient by avg RA)")
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
        click.echo("✅ Figure 5 (v2) generation complete!")
        click.echo(f"   Main figure: {output}")
        click.echo(f"   Preview: {Path(output).with_suffix('.png')}")
        click.echo("=" * 80 + "\n")


if __name__ == '__main__':
    main()
