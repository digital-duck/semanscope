#!/usr/bin/env python3
"""
Create Figure 3: 2×2 PHATE grid for ICML paper
Combines 4 DS1 PHATE plots (LaBSE, OpenAI-3-Large, Qwen3-0.6B, mBERT)

Generates both PNG and PDF versions for publication.
"""

import matplotlib.pyplot as plt
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os

# File paths - source PDFs - use relative path from package root
source_pdf_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'images' / 'PDF'

pdf_files = {
    'labse': source_pdf_dir / 'echarts-icml-1-peterg-phate-labse-enu-chn.pdf',
    'openai': source_pdf_dir / 'echarts-icml-1-peterg-phate-openai-3-large-enu-chn.pdf',
    'qwen3': source_pdf_dir / 'echarts-icml-1-peterg-phate-qwen3-06b-enu-chn.pdf',
    'mbert': source_pdf_dir / 'echarts-icml-1-peterg-phate-mbert-enu-chn.pdf'
}

# Output directory - save to data folder
output_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'images'
output_png = output_dir / 'figure3_phate_comparison_2x2.png'
output_pdf = output_dir / 'figure3_phate_comparison_2x2.pdf'

print("📊 Creating Figure 3: 2×2 PHATE Grid")
print(f"   Output PNG: {output_png}")
print(f"   Output PDF: {output_pdf}")

# Convert PDFs to images using pdftoppm
print("\n🔄 Converting PDFs to images (DPI=300)...")
images = {}

with tempfile.TemporaryDirectory() as tmpdir:
    for key, pdf_path in pdf_files.items():
        print(f"   Processing {key}...")

        # Convert PDF to PNG using pdftoppm
        output_prefix = os.path.join(tmpdir, f"{key}")
        subprocess.run([
            'pdftoppm',
            '-png',
            '-r', '300',  # 300 DPI
            '-singlefile',
            str(pdf_path),
            output_prefix
        ], check=True, capture_output=True)

        # Load the generated PNG
        png_file = f"{output_prefix}.png"
        images[key] = np.array(Image.open(png_file))

# Create 2×2 grid
print("\n🎨 Creating 2×2 grid layout...")
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.suptitle('Figure 3: PHATE Manifolds with Integrated SA Legends (DS1: Peterg, 349 words)',
             fontsize=20, fontweight='bold', y=0.995)

# Layout: [Top-left, Top-right]
#         [Bottom-left, Bottom-right]
grid_layout = [
    ('labse', 'LaBSE (Tier 1, SA=0.692)'),
    ('openai', 'OpenAI-3-Large (Tier 2, SA=0.582)'),
    ('qwen3', 'Qwen3-0.6B (Tier 3, SA=0.502)'),
    ('mbert', 'mBERT (Tier 3, SA=0.507)')
]

for idx, (key, title) in enumerate(grid_layout):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    ax.imshow(images[key])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save both PNG and PDF versions
print(f"\n💾 Saving outputs...")
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Figure 3 PNG saved: {output_png}")
print(f"   Size: {output_png.stat().st_size / 1024 / 1024:.2f} MB")
print(f"✅ Figure 3 PDF saved: {output_pdf}")
print(f"   Size: {output_pdf.stat().st_size / 1024 / 1024:.2f} MB")

plt.close()

print("\n" + "="*60)
print("✅ Figure 3 generation complete!")
print(f"   PNG: {output_png}")
print(f"   PDF: {output_pdf}")
print("="*60)
