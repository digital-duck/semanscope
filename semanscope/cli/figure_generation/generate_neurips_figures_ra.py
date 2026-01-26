#!/usr/bin/env python3
"""
Generate figures for NeurIPS 2026 paper from RA metric CSV files.

Uses RA (Cosine) as PRIMARY metric for Relational Affinity.

Figure 1: Three-tier bar chart (average RA cosine across EASY datasets: DS1-DS4)
Figure 2: Heatmap (17 models × 4 EASY datasets, RA cosine)

Key Insight: All models perform poorly on DS5-DS11 (hard relational structure).
This shows Semanscope acts as a "microscope" revealing fundamental limitations
invisible to traditional benchmarks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Model alias mapping
MODEL_ALIASES = {
    'LaBSE': 'LaBSE',
    'Universal-Sentence-Encoder-Multilingual': 'USE',
    'Sentence-BERT Multilingual': 'Sentence-BERT',
    'Gemini-Embedding-001 (OpenRouter)': 'Gemini-001',
    'Qwen3-Embedding-8B (OpenRouter)': 'Qwen3-8B',
    'Qwen3-Embedding-4B (Ollama)': 'Qwen3-4B',
    'Qwen3-Embedding-0.6B': 'Qwen3-0.6B',
    'OpenAI Text-Embedding-3-Large (OpenRouter)': 'OpenAI-3-Large',
    'OpenAI Text-Embedding-3-Small (OpenRouter)': 'OpenAI-3-Small',
    'OpenAI Text-Embedding-Ada-002 (OpenRouter)': 'OpenAI-Ada-002',
    'Multilingual-E5-Large-Instruct-v2': 'E5-Large',
    'mBERT': 'mBERT',
    'XLM-RoBERTa-v2': 'XLM-R',
    'EmbeddingGemma-300M': 'EmbeddingGemma',
    'DistilBERT Multilingual': 'DistilBERT',
    'BGE-M3 (Ollama)': 'BGE-M3',
    'Snowflake-Arctic-Embed2 (Ollama)': 'Snowflake-Arctic'
}

# Easy datasets (DS1-DS4)
EASY_DATASETS = ['DS1', 'DS2', 'DS3', 'DS4']

TIER_COLORS = {
    'Tier 1: Top Performers': '#2ecc71',
    'Tier 2: Mid-Range': '#3498db',
    'Tier 3: Low Performance': '#e74c3c'
}


def parse_ra_value(value_str):
    """Parse 'RA ± SEM' string to extract RA score."""
    if isinstance(value_str, str) and '±' in value_str:
        return float(value_str.split('±')[0].strip())
    return float(value_str)


def load_and_process_csv(csv_path):
    """Load CSV and extract RA scores for EASY datasets only."""
    df = pd.read_csv(csv_path)
    df['model_alias'] = df['model'].map(MODEL_ALIASES)
    
    dataset_cols = [col for col in df.columns if any(ds in col for ds in EASY_DATASETS)]
    
    data = []
    for _, row in df.iterrows():
        model_alias = row['model_alias']
        for col in dataset_cols:
            for ds in EASY_DATASETS:
                if ds in col:
                    dataset_name = ds
                    break
            else:
                continue
            
            ra_score = parse_ra_value(row[col])
            data.append({
                'model_alias': model_alias,
                'dataset': dataset_name,
                'ra_score': ra_score
            })
    
    return pd.DataFrame(data)


def assign_tier(avg_ra):
    """Assign tier based on average RA score on easy datasets."""
    if avg_ra > 0.50:
        return 'Tier 1: Top Performers'
    elif avg_ra >= 0.30:  # Changed from 0.35 to 0.30 to include OpenAI models
        return 'Tier 2: Mid-Range'
    else:
        return 'Tier 3: Low Performance'


def generate_figure1(df, output_path):
    """Figure 1: Three-tier bar chart."""
    avg_df = df.groupby('model_alias')['ra_score'].mean().reset_index()
    avg_df.columns = ['model_alias', 'avg_ra']
    avg_df['tier'] = avg_df['avg_ra'].apply(assign_tier)
    
    tier_order = ['Tier 1: Top Performers', 'Tier 2: Mid-Range', 'Tier 3: Low Performance']
    avg_df['tier_order'] = avg_df['tier'].map({t: i for i, t in enumerate(tier_order)})
    avg_df = avg_df.sort_values(['tier_order', 'avg_ra'], ascending=[True, False])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = [TIER_COLORS[tier] for tier in avg_df['tier']]
    bars = ax.bar(range(len(avg_df)), avg_df['avg_ra'], color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.2)
    
    for i, (bar, ra) in enumerate(zip(bars, avg_df['avg_ra'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{ra:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(avg_df)))
    ax.set_xticklabels(avg_df['model_alias'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel(r'Average $RA_{\mathrm{cos}}$', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title(r'Model Performance Tiers (Average $RA_{\mathrm{cos}}$ across DS1-DS4: Family, Royalty, Occupations, Gradation)',
                 fontsize=15, fontweight='bold', pad=20)

    ax.axhline(y=0.50, color='darkgreen', linestyle=':', linewidth=2.5, alpha=0.7,
               label='RA = 0.50 (Top/Mid threshold)')
    ax.axhline(y=0.30, color='darkred', linestyle=':', linewidth=2.5, alpha=0.7,
               label='RA = 0.30 (Mid/Low threshold)')

    tier1_count = len(avg_df[avg_df['tier'] == 'Tier 1: Top Performers'])
    tier2_count = len(avg_df[avg_df['tier'] == 'Tier 2: Mid-Range'])

    if tier1_count > 0:
        ax.axvspan(-0.5, tier1_count - 0.5, alpha=0.1, color='green',
                   label='Tier 1: Top Performers')
    if tier2_count > 0:
        ax.axvspan(tier1_count - 0.5, tier1_count + tier2_count - 0.5,
                   alpha=0.1, color='blue', label='Tier 2: Mid-Range')
    ax.axvspan(tier1_count + tier2_count - 0.5, len(avg_df) - 0.5,
               alpha=0.1, color='red', label='Tier 3: Low Performance')

    ax.set_ylim(0, 0.65)  # Changed to start from 0
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    png_path = output_path
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 1 saved: {png_path}")
    print(f"✅ Figure 1 saved: {pdf_path}")
    plt.close()


def generate_figure2(df, output_path):
    """Figure 2: Heatmap."""
    # Aggregate any duplicates by taking mean
    df_agg = df.groupby(['model_alias', 'dataset'])['ra_score'].mean().reset_index()
    pivot_df = df_agg.pivot(index='model_alias', columns='dataset', values='ra_score')
    pivot_df = pivot_df[EASY_DATASETS]

    avg_ra = pivot_df.mean(axis=1).sort_values(ascending=False)
    pivot_df = pivot_df.loc[avg_ra.index]

    # Create larger figure with better aspect ratio
    fig, ax = plt.subplots(figsize=(12, 15))

    # Create heatmap WITHOUT automatic annotations first
    sns.heatmap(pivot_df, annot=False, cmap='RdYlGn', center=0.45,
                vmin=0.20, vmax=0.70, cbar_kws={'label': r'$RA_{\mathrm{cos}}$'},
                linewidths=1.5, linecolor='black', ax=ax,
                square=False, cbar=True)

    # Manually add annotations to ENSURE they all render
    for i in range(len(pivot_df)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f'{value:.3f}',
                   ha='center', va='center',
                   fontsize=16, fontweight='bold',
                   color='black',
                   clip_on=False)

    ax.set_xlabel('Dataset (Easy Categories)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title(r'Model Performance Heatmap (17 Models × 4 Datasets, $RA_{\mathrm{cos}}$)',
                 fontsize=15, fontweight='bold', pad=20)

    dataset_labels = [
        'DS1 [n=20]\nFamily\nRelations',
        'DS2 [n=15]\nRoyalty\nHierarchy',
        'DS3 [n=20]\nGendered\nOccupations',
        'DS4 [n=15]\nComparative\nSuperlative'
    ]
    ax.set_xticklabels(dataset_labels, rotation=0, ha='center', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Draw threshold lines at RA = 0.50 and 0.30
    for i, model_avg in enumerate(avg_ra):
        if i > 0 and avg_ra.iloc[i-1] > 0.50 and model_avg <= 0.50:
            ax.axhline(y=i, color='darkgreen', linewidth=3, linestyle='--', label='RA = 0.50 threshold' if i == 2 else '')
        elif i > 0 and avg_ra.iloc[i-1] >= 0.30 and model_avg < 0.30:
            ax.axhline(y=i, color='darkred', linewidth=3, linestyle='--', label='RA = 0.30 threshold' if i > 2 else '')
    
    plt.tight_layout()
    
    png_path = output_path
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 2 saved: {png_path}")
    print(f"✅ Figure 2 saved: {pdf_path}")
    plt.close()


def main():
    """Main execution."""
    # Use relative path from package root
    ROOT_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'batch_results'
    # Use the cleaned CSV with only easy datasets
    csv_path = f'{ROOT_DIR}/cli_batch_benchmark_ra-2025-12-27-all-models-easy-datasets-EN-ZH-cos-cross-pivot.csv'
    output_dir = Path(ROOT_DIR)

    fig1_path = output_dir / 'neurips_figure1_model_tiers_easy_datasets.png'
    fig2_path = output_dir / 'neurips_figure2_heatmap_easy_datasets.png'

    print("📊 Generating NeurIPS 2026 paper figures...")
    print(f"   Input: {csv_path}")
    print(f"   Using EASY datasets only: {EASY_DATASETS} (DS1-DS4)")

    df = load_and_process_csv(csv_path)
    print(f"   Loaded: {len(df)} rows ({df['model_alias'].nunique()} models × {df['dataset'].nunique()} datasets)")

    generate_figure1(df, fig1_path)
    generate_figure2(df, fig2_path)

    print("\n" + "="*70)
    print("✅ All NeurIPS figures generated successfully!")
    print(f"   Figure 1: {fig1_path} (+ PDF)")
    print(f"   Figure 2: {fig2_path} (+ PDF)")
    print("="*70)


if __name__ == '__main__':
    main()
