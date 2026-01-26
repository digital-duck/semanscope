#!/usr/bin/env python3
"""
Generate Figure 4: Cross-Lingual Morphological Anisotropy Visualization for NeurIPS Paper

Shows the dramatic performance difference between Chinese (compositional morphology)
and English (lexical irregularity) for Animal Gender category, revealing directional
dependence of relational encoding across languages.

Usage:
    python generate_figure4_morphological_asymmetry.py \
        --output neurips_figure4-morphological-anisotropy.pdf \
        [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path


def parse_score(val: str) -> float:
    """Parse score from 'mean ± sem' format."""
    if pd.isna(val) or val == 'N/A':
        return np.nan
    try:
        return float(val.split('±')[0].strip())
    except:
        return np.nan


def create_morphological_asymmetry_figure(
    data_path: str,
    output_path: str,
    figsize: tuple = (12, 8),
    dpi: int = 300,
    verbose: bool = True
):
    """
    Create visualization showing cross-lingual morphological anisotropy for Animal Gender category.

    Reads actual RA scores from benchmark results:
    - Chinese (compositional): 公牛/母牛, 公鸡/母鸡 (systematic prefix marking)
    - English (irregular): bull/cow, rooster/hen (lexical suppletive pairs)

    Anisotropy: Directional dependence of relational encoding (ZH >> EN for gender relations)

    Args:
        data_path: Path to RA pivot CSV with EN and ZH columns
        output_path: Path to save figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output
        verbose: Print progress messages
    """
    if verbose:
        click.echo(f"📊 Generating Figure 4: Cross-Lingual Morphological Anisotropy")
        click.echo(f"   Reading data: {data_path}")

    # Read RA data (non-pivot format with rows per model-dataset combo)
    df = pd.read_csv(data_path)

    # Filter for Animal Gender dataset (NeurIPS-06-animal-gender or containing 'animal')
    df_animal = df[df['dataset'].str.contains('animal', case=False, na=False)].copy()

    if len(df_animal) == 0:
        raise ValueError(f"Could not find Animal Gender dataset rows. Available datasets: {df['dataset'].unique().tolist()}")

    if verbose:
        click.echo(f"   Found {len(df_animal)} rows for Animal Gender dataset")

    # Extract data for all models
    models = []
    english_ra = []
    chinese_ra = []

    for idx, row in df_animal.iterrows():
        model_name = row['model'].replace(' (OpenRouter)', '').replace(' (Ollama)', '')
        models.append(model_name)
        # Use cos_EN and cos_ZH columns (these are the EN-RA and ZH-RA scores)
        english_ra.append(row['cos_EN'] if not pd.isna(row['cos_EN']) else np.nan)
        chinese_ra.append(row['cos_ZH'] if not pd.isna(row['cos_ZH']) else np.nan)

    # Create dataframe and sort by Chinese RA descending, take top 10
    df = pd.DataFrame({
        'Model': models,
        'English_RA': english_ra,
        'Chinese_RA': chinese_ra
    })
    df = df.sort_values('Chinese_RA', ascending=False).head(10).reset_index(drop=True)

    # Calculate performance ratio
    df['Ratio'] = df['Chinese_RA'] / df['English_RA']
    avg_ratio = df['Ratio'].mean()

    if verbose:
        click.echo(f"   Average Chinese/English ratio: {avg_ratio:.1f}x")
        click.echo(f"   Mean English RA: {df['English_RA'].mean():.4f}")
        click.echo(f"   Mean Chinese RA: {df['Chinese_RA'].mean():.4f}")

    # Set publication style
    plt.style.use('seaborn-v0_8-white')
    sns.set_context("paper", font_scale=1.2)

    # Create figure with 2 subplots - DON'T share Y-axis (causes label issues)
    # Left chart wider (65%), right chart narrower (35%) - right is just illustrative
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                    gridspec_kw={'wspace': 0.05, 'width_ratios': [2, 1]})

    # === Left panel: Horizontal grouped bars (Chinese above, English below for each model) ===
    # Keep original order (top-down: LaBSE first)
    n_models = len(df)
    y_positions = np.arange(n_models)
    bar_height = 0.35

    # With inverted Y-axis, negative offset = above, positive offset = below
    # Chinese bars (above - use negative offset)
    bars_chinese = ax1.barh(y_positions - bar_height/2, df['Chinese_RA'], bar_height,
                            label='Chinese (compositional)',
                            color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
    # English bars (below - use positive offset)
    bars_english = ax1.barh(y_positions + bar_height/2, df['English_RA'], bar_height,
                            label='English (non-compositional)',
                            color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Relational Affinity (RA)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Embedding Model', fontsize=12, fontweight='bold')
    # Removed sub-chart title
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(df['Model'], rotation=45, ha='right', fontsize=10)  # 45 degree rotation
    ax1.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    # Increase xlim to accommodate BGE-M3 Chinese RA (0.8062) with margin
    max_ra = max(df['Chinese_RA'].max(), df['English_RA'].max())
    ax1.set_xlim(0, max(0.85, max_ra * 1.05))  # At least 0.85, or 5% margin above max
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()  # Top-down order

    # === Right panel: Ratio bars (mono-color grey) ===
    # Narrower bars with ratio text inside
    bars_ratio = ax2.barh(y_positions, df['Ratio'], bar_height * 1.2,
                          color='grey', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('RA Ratio', fontsize=12, fontweight='bold')
    # Removed sub-chart title
    ax2.set_xlim(0, max(df['Ratio']) * 1.1)
    ax2.grid(True, alpha=0.3, axis='x')

    # Sync Y-axis with left chart - same order, hide labels on right
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([])  # Empty labels
    ax2.invert_yaxis()  # Same orientation as left - top-down order

    # Annotate bars with ratio values INSIDE the bars (blue text on grey background)
    for i, ratio in enumerate(df['Ratio']):
        ax2.text(ratio * 0.95, i, f"{ratio:.1f}×",
                 va='center', ha='right', fontsize=9, fontweight='bold', color='blue')

    # Add main title
    fig.suptitle('Figure 5: Cross-Lingual Morphological Anisotropy (Animal Gender)',
                  fontsize=15, fontweight='bold', y=0.96)

    # Adjust layout with extra space on left for Y-axis labels
    plt.subplots_adjust(left=0.20, right=0.98, top=0.92, bottom=0.08, wspace=0.05)

    # Save figure - don't use bbox_inches='tight' to avoid clipping labels
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=dpi)

    if verbose:
        click.echo(f"\n   ✅ Figure saved: {output_path}")
        click.echo(f"   Resolution: {dpi} DPI")
        click.echo(f"   Size: {figsize[0]}\" × {figsize[1]}\"")

    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150)

    if verbose:
        click.echo(f"   ✅ Preview saved: {png_path}")

    plt.close()

    # Print key statistics
    if verbose:
        click.echo(f"\n📊 Key Findings:")
        click.echo(f"   Chinese compositional morphology: {df['Chinese_RA'].mean():.4f} RA")
        click.echo(f"   English lexical irregularity:      {df['English_RA'].mean():.4f} RA")
        click.echo(f"   Performance advantage: {avg_ratio:.0f}× ({(avg_ratio-1)*100:.0f}% better)")
        click.echo(f"\n   Best Chinese RA:  {df['Chinese_RA'].max():.4f} (LaBSE)")
        click.echo(f"   Best English RA:  {df['English_RA'].max():.4f}")
        click.echo(f"   Peak advantage:   {df['Ratio'].max():.0f}×")


@click.command()
@click.option(
    '--input',
    '-i',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to RA CSV file (non-pivot format with cos_EN/cos_ZH columns)'
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
    default=(12, 8),
    type=(float, float),
    help='Figure size in inches (width, height) (default: 12 8)'
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
    Generate Figure 4: Cross-lingual morphological anisotropy visualization.

    Shows the dramatic performance difference between Chinese compositional
    morphology (公牛/母牛 - systematic prefix marking) and English lexical
    irregularity (bull/cow - suppletive pairs) for Animal Gender category.

    Examples:

        # Basic usage
        python generate_figure4_morphological_asymmetry.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \\
            --output neurips_figure4-morphological-asymmetry.pdf

        # High resolution
        python generate_figure4_morphological_asymmetry.py \\
            --input cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \\
            --output neurips_figure4-morphological-asymmetry.pdf \\
            --dpi 600
    """
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("FIGURE 4 GENERATION: Cross-Lingual Morphological Anisotropy")
        click.echo("=" * 80 + "\n")

    create_morphological_asymmetry_figure(
        data_path=input,
        output_path=output,
        figsize=figsize,
        dpi=dpi,
        verbose=verbose
    )

    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("✅ Figure 4 generation complete!")
        click.echo(f"   Main figure: {output}")
        click.echo(f"   Preview: {Path(output).with_suffix('.png')}")
        click.echo("=" * 80 + "\n")


if __name__ == '__main__':
    main()
