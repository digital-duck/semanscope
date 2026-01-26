#!/usr/bin/env python3
"""
Generate figures for ICML paper from SA metric CSV files.

Uses SA (Cosine) as PRIMARY metric (v2.0).

Figure 1: Three-tier bar chart (average SA cosine across 4 datasets)
Figure 2: Heatmap (13 models × 4 datasets, SA cosine)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Model alias mapping (full name → short alias)
MODEL_ALIASES = {
    'LaBSE': 'LaBSE',
    'Universal-Sentence-Encoder-Multilingual': 'USE',
    'Sentence-BERT Multilingual': 'Sentence-BERT',
    'Gemini-Embedding-001 (OpenRouter)': 'Gemini-001',
    'Qwen3-Embedding-8B (OpenRouter)': 'Qwen3-8B',
    'Qwen3-Embedding-4B (OpenRouter)': 'Qwen3-4B',
    'Qwen3-Embedding-0.6B': 'Qwen3-0.6B',
    'OpenAI Text-Embedding-3-Large (OpenRouter)': 'OpenAI-3-Large',
    'OpenAI Text-Embedding-3-Small (OpenRouter)': 'OpenAI-3-Small',
    'OpenAI Text-Embedding-Ada-002 (OpenRouter)': 'OpenAI-Ada-002',
    'Multilingual-E5-Large-Instruct-v2': 'E5-Large',
    'mBERT': 'mBERT',
    'XLM-RoBERTa-v2': 'XLM-R'
}

# Three-tier classification
TIER_1_MODELS = ['LaBSE', 'USE', 'Sentence-BERT']
TIER_2_MODELS = ['OpenAI-3-Large', 'Gemini-001', 'OpenAI-3-Small', 'Qwen3-8B',
                 'Qwen3-4B', 'OpenAI-Ada-002', 'E5-Large']
TIER_3_MODELS = ['mBERT', 'XLM-R', 'Qwen3-0.6B']

TIER_COLORS = {
    'Tier 1: Top BERT': '#2ecc71',  # Green
    'Tier 2: LLM Plateau': '#3498db',  # Blue
    'Tier 3: Failed': '#e74c3c'  # Red
}


def parse_sa_value(value_str):
    """Parse 'SA ± SEM' string to extract SA score."""
    if isinstance(value_str, str) and '±' in value_str:
        return float(value_str.split('±')[0].strip())
    return float(value_str)


def load_and_process_csv(csv_path):
    """Load CSV and extract SA scores."""
    df = pd.read_csv(csv_path)

    # Apply model aliases
    df['model_alias'] = df['model'].map(MODEL_ALIASES)

    # Parse SA scores from all dataset columns
    dataset_cols = [col for col in df.columns if col.startswith('DS')]

    data = []
    for _, row in df.iterrows():
        model_alias = row['model_alias']
        for col in dataset_cols:
            sa_score = parse_sa_value(row[col])
            dataset_name = col.split('[')[0].strip()  # Extract 'DS1', 'DS2', etc.
            data.append({
                'model_alias': model_alias,
                'dataset': dataset_name,
                'sa_score': sa_score
            })

    return pd.DataFrame(data)


def assign_tier(model_alias):
    """Assign tier to model."""
    if model_alias in TIER_1_MODELS:
        return 'Tier 1: Top BERT'
    elif model_alias in TIER_2_MODELS:
        return 'Tier 2: LLM Plateau'
    elif model_alias in TIER_3_MODELS:
        return 'Tier 3: Failed'
    else:
        return 'Unknown'


def generate_figure1(df, output_path):
    """
    Figure 1: Three-tier bar chart showing average SA across 4 datasets.
    """
    # Calculate average SA per model
    avg_df = df.groupby('model_alias')['sa_score'].mean().reset_index()
    avg_df.columns = ['model_alias', 'avg_sa']
    avg_df['tier'] = avg_df['model_alias'].apply(assign_tier)

    # Sort by tier and then by SA score within tier
    tier_order = ['Tier 1: Top BERT', 'Tier 2: LLM Plateau', 'Tier 3: Failed']
    avg_df['tier_order'] = avg_df['tier'].map({t: i for i, t in enumerate(tier_order)})
    avg_df = avg_df.sort_values(['tier_order', 'avg_sa'], ascending=[True, False])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars with tier-specific colors
    colors = [TIER_COLORS[tier] for tier in avg_df['tier']]
    bars = ax.bar(range(len(avg_df)), avg_df['avg_sa'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, sa) in enumerate(zip(bars, avg_df['avg_sa'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{sa:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Formatting
    ax.set_xticks(range(len(avg_df)))
    ax.set_xticklabels(avg_df['model_alias'], rotation=45, ha='right')
    ax.set_ylabel('Average SA Score (Cosine)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Figure 1: Three-Tier Performance Structure\n(Average SA Cosine across 4 Datasets)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add horizontal lines for tier thresholds (COSINE)
    ax.axhline(y=0.60, color='darkgreen', linestyle=':', linewidth=2.0, alpha=0.7, label='SA = 0.60 (Great/Good threshold)')
    ax.axhline(y=0.50, color='darkred', linestyle=':', linewidth=2.0, alpha=0.7, label='SA = 0.50 (Alignment/Non-alignment)')

    # Add tier labels as background regions
    tier1_count = len([m for m in avg_df['model_alias'] if m in TIER_1_MODELS])
    tier2_count = len([m for m in avg_df['model_alias'] if m in TIER_2_MODELS])

    ax.axvspan(-0.5, tier1_count - 0.5, alpha=0.1, color='green', label='Tier 1: Top BERT')
    ax.axvspan(tier1_count - 0.5, tier1_count + tier2_count - 0.5, alpha=0.1, color='blue', label='Tier 2: LLM Plateau')
    ax.axvspan(tier1_count + tier2_count - 0.5, len(avg_df) - 0.5, alpha=0.1, color='red', label='Tier 3: Failed')

    ax.set_ylim(0.40, 0.75)  # Extended for cosine range
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()

    # Save both PNG and PDF versions
    png_path = output_path
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 1 saved: {png_path}")
    print(f"✅ Figure 1 saved: {pdf_path}")
    plt.close()


def generate_figure2(df, output_path):
    """
    Figure 2: Heatmap showing SA scores for 13 models × 4 datasets.
    """
    # Pivot data to create heatmap matrix
    pivot_df = df.pivot(index='model_alias', columns='dataset', values='sa_score')

    # Reorder columns (DS1, DS2, DS3, DS4)
    pivot_df = pivot_df[['DS1', 'DS2', 'DS3', 'DS4']]

    # Reorder rows by tier and SA score
    model_order = []
    for tier_models in [TIER_1_MODELS, TIER_2_MODELS, TIER_3_MODELS]:
        tier_df = pivot_df.loc[[m for m in tier_models if m in pivot_df.index]]
        tier_avg = tier_df.mean(axis=1).sort_values(ascending=False)
        model_order.extend(tier_avg.index.tolist())

    pivot_df = pivot_df.loc[model_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap (COSINE: wider range, center at 0.60)
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.60,
                vmin=0.40, vmax=0.80, cbar_kws={'label': 'SA Score (Cosine)'},
                linewidths=1.5, linecolor='black', ax=ax,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})

    # Formatting
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Dataset Difficulty Gradient Heatmap\n(13 Models × 4 Datasets, SA Cosine)',
                 fontsize=14, fontweight='bold', pad=20)

    # Update column labels with full names
    dataset_labels = [
        'DS1\n[349 words]\nPeterg',
        'DS2\n[341 words]\nCultural',
        'DS3\n[177 words]\nChallenge',
        'DS4\n[769 words]\nZiNets'
    ]
    ax.set_xticklabels(dataset_labels, rotation=0, ha='center')

    # Add tier separators
    tier1_count = len([m for m in model_order if m in TIER_1_MODELS])
    tier2_count = len([m for m in model_order if m in TIER_2_MODELS])

    ax.axhline(y=tier1_count, color='black', linewidth=3)
    ax.axhline(y=tier1_count + tier2_count, color='black', linewidth=3)

    plt.tight_layout()

    # Save both PNG and PDF versions
    png_path = output_path
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 2 saved: {png_path}")
    print(f"✅ Figure 2 saved: {pdf_path}")
    plt.close()


def main():
    """Main execution."""
    # File paths
    # Use relative path from semanscope package root
    ROOT_DIR = Path(__file__).parent.parent.parent.parent / 'data'
    csv_path = f'{ROOT_DIR}/sa-metric-enu-chn-cosine.csv'  # Changed to cosine (PRIMARY)
    output_dir = Path(ROOT_DIR)

    fig1_path = output_dir / 'figure1_three_tier_bar_chart.png'
    fig2_path = output_dir / 'figure2_dataset_difficulty_heatmap.png'

    print("📊 Generating ICML paper figures...")
    print(f"   Input: {csv_path}")

    # Load data
    df = load_and_process_csv(csv_path)
    print(f"   Loaded data: {len(df)} rows ({df['model_alias'].nunique()} models × {df['dataset'].nunique()} datasets)")

    # Generate figures
    generate_figure1(df, fig1_path)
    generate_figure2(df, fig2_path)

    print("\n" + "="*60)
    print("✅ All figures generated successfully!")
    print(f"   Figure 1 (PNG): {fig1_path}")
    print(f"   Figure 1 (PDF): {str(fig1_path).replace('.png', '.pdf')}")
    print(f"   Figure 2 (PNG): {fig2_path}")
    print(f"   Figure 2 (PDF): {str(fig2_path).replace('.png', '.pdf')}")
    print("="*60)


if __name__ == '__main__':
    main()
