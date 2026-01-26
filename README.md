# Semanscope

**Multilingual Semantic Embedding Visualization and Analysis Toolkit**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Semanscope is a comprehensive toolkit for visualizing and analyzing semantic embeddings across multiple languages. It features advanced metrics for measuring semantic consistency (Semantic Affinity) and relational structure preservation (Relational Affinity) in multilingual embedding models.

## Key Features

- **Multi-Model Support**: LaBSE, SONAR, Gemma, OpenAI, Voyage AI, Google Gemini, Ollama, and 30+ models
- **Advanced Dimensionality Reduction**: UMAP, PHATE, t-SNE, PaCMAP, TriMap
- **Semantic Affinity (SA)**: Novel metric for measuring semantic consistency across embeddings
- **Relational Affinity (RA)**: Metric for evaluating relational structure preservation
- **Interactive UI**: Streamlit-based interface with 11 specialized pages
- **Batch Benchmarking**: CLI tools for research-grade evaluation
- **Multilingual**: Support for 70+ languages
- **Visualization**: Interactive plots with Plotly and ECharts

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/semanscope/semanscope.git
cd semanscope

# Create conda environment
conda create -n semanscope python=3.11
conda activate semanscope

# Install package with UI support
pip install -e ".[ui]"

# Or install with all dependencies (including API integrations)
pip install -e ".[all]"
```

### Launch the UI

```bash
# Option 1: Using the launcher script
python run_app.py

# Option 2: Using the CLI command (after installation)
semanscope-ui
```

### Basic Usage (Python API)

```python
from semanscope.models.model_manager import get_model
from semanscope.components.embedding_viz import EmbeddingVisualizer

# Load a model
model = get_model("LaBSE")

# Create visualizer
viz = EmbeddingVisualizer(model=model)

# Visualize embeddings
words = ["hello", "world", "friend", "peace"]
viz.plot_words(words, method="UMAP", dimension=2)
```

### Batch Benchmarking

```bash
# Semantic Affinity benchmark
semanscope-benchmark-sa \
    --dataset data/input/NeurIPS-01-family-relations-v2.5-SA.csv \
    --models LaBSE SONAR \
    --output results/sa_benchmark.csv

# Relational Affinity benchmark
semanscope-benchmark-ra \
    --dataset data/input/NeurIPS-01-family-relations-v2.5-RA.csv \
    --models LaBSE SONAR \
    --languages english chinese \
    --output results/ra_benchmark.csv
```

## Features in Detail

### Semantic Affinity (SA) Metric

Measures how consistently a model represents semantic relationships:

```python
from semanscope.components.semantic_affinity import calculate_semantic_affinity

sa_score = calculate_semantic_affinity(
    model=model,
    word_pairs=[("cat", "dog"), ("happy", "sad")],
    metric="cosine"
)
```

**SA Formula**:
```
SA = 1 - std(similarities) / mean(similarities)
```

Higher SA (→1.0) = more consistent semantic representations

### Relational Affinity (RA) Metric

Evaluates preservation of relational structure across languages:

```python
from semanscope.components import calculate_relational_affinity

ra_score = calculate_relational_affinity(
    model=model,
    word_quadruples=[("king", "queen", "man", "woman")],
    languages=["english", "chinese"],
    metric="cosine"
)
```

**RA Formula** (Cosine):
```
rel_vec(w1, w2) = emb(w2) - emb(w1)
RA = cosine_similarity(rel_vec_lang1, rel_vec_lang2)
```

Higher RA (→1.0) = better relational structure preservation

### Interactive UI Pages

1. **Settings** (0_🔧_Settings.py): Configure models, methods, cache
2. **Semanscope** (1_🧭_Semanscope.py): Main visualization interface
3. **Semanscope ECharts** (2_📊_Semanscope-ECharts.py): ECharts-based visualization
4. **Compare** (3_⚖️_Semanscope-Compare.py): Side-by-side model comparison
5. **Multilingual** (4_🌐_Semanscope-Multilingual.py): Multi-language visualization
6. **Zoom** (5_🔍_Semanscope-Zoom.py): Interactive zoom and exploration
7. **Semantic Affinity** (6_📐_Semantic_Affinity.py): SA metric calculator
8. **Relational Affinity** (6_🔗_Relational_Affinity.py): RA metric calculator
9. **Translator** (8_🌐_Translator.py): Translation utilities
10. **NSM Prime Words** (9_📝_NSM_Prime_Words.py): Natural Semantic Metalanguage
11. **Review Images** (9_🖼️_Review_Images.py): Visualization gallery

### Supported Models

**Open Source**:
- LaBSE (Language-agnostic BERT Sentence Embedding)
- SONAR (Seamless Communication models)
- XLM-RoBERTa variants
- mBERT (Multilingual BERT)
- And 20+ more...

**API-based** (requires API keys):
- OpenAI (text-embedding-ada-002, text-embedding-3-small, etc.)
- Voyage AI (voyage-multilingual-2, voyage-code-2)
- Google Gemini (text-embedding-004)
- Ollama (local models)

See `semanscope/config.py` for complete model catalog.

### Dimensionality Reduction Methods

- **UMAP**: Uniform Manifold Approximation and Projection
- **PHATE**: Potential of Heat-diffusion for Affinity-based Transition Embedding
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding
- **PaCMAP**: Pairwise Controlled Manifold Approximation
- **TriMap**: Triplet-based dimensionality reduction
- **PCA**: Principal Component Analysis

## Datasets

Semanscope includes 60+ representative datasets across 7 categories:

- **ACL-0**: Chinese morphology (Zinets, Radicals)
- **ACL-1**: Alphabets (15+ languages)
- **ACL-2**: PeterG vocabulary (semantic primes)
- **ACL-3**: Morphological networks
- **ACL-4**: Semantic categories (numbers, emotions, animals)
- **ACL-5**: Poetry corpora (Li Bai, Du Fu, Frost, Wordsworth)
- **ACL-6**: Visual semantics (emoji, pictographs)
- **NeurIPS-01 to NeurIPS-11**: Research benchmarks for SA/RA metrics

See `data/input/README.md` for complete dataset documentation.

## Documentation

- **[Usage Guide](docs/USAGE.md)**: Detailed usage instructions
- **[API Reference](docs/API.md)**: Python API documentation
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[GPU Setup](docs/GPU_SETUP.md)**: CUDA configuration for acceleration

## Architecture

```
semanscope/
├── semanscope/          # Core Python package
│   ├── components/      # Analysis components (SA, RA, viz)
│   ├── models/          # Model managers and integrations
│   ├── utils/           # Utilities (caching, text processing)
│   ├── services/        # External API integrations
│   └── cli/             # Command-line tools
├── ui/                  # Streamlit UI
├── data/                # Datasets and visualizations
├── tests/               # Test suite
├── demo/                # Usage examples
├── scripts/             # Utility scripts
└── docs/                # Documentation
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_semantic_affinity.py -v

# Code formatting
black semanscope/ ui/ tests/
ruff check semanscope/ ui/
```

## Configuration

Create a `.env` file for API keys and settings:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your API keys
OPENROUTER_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Performance Tips

1. **Use GPU**: Set `CUDA_VISIBLE_DEVICES=0` for GPU acceleration
2. **Enable caching**: Embeddings are cached automatically to `~/projects/embedding_cache/`
3. **Batch processing**: Use CLI tools for large-scale benchmarking
4. **Model selection**: Start with smaller models (LaBSE, mBERT) for exploration

## Citation

If you use Semanscope in your research, please cite:

```bibtex
@software{semanscope2026,
  title={Semanscope: Multilingual Semantic Embedding Visualization Toolkit},
  author={Semanscope Contributors},
  year={2026},
  url={https://github.com/semanscope/semanscope}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- **Language Models**: Thanks to Google (LaBSE), Meta (SONAR), and the open-source community
- **Dimensionality Reduction**: UMAP, PHATE, t-SNE, PaCMAP, TriMap libraries
- **Visualization**: Plotly, Streamlit, ECharts
- **Datasets**: Computational linguistics research community

## Support

- **Documentation**: [GitHub Wiki](https://github.com/semanscope/semanscope/wiki)
- **Issues**: [GitHub Issues](https://github.com/semanscope/semanscope/issues)
- **Discussions**: [GitHub Discussions](https://github.com/semanscope/semanscope/discussions)

## Roadmap

- [ ] PyPI publication
- [ ] Additional embedding models (Cohere, Anthropic)
- [ ] Enhanced visualization options
- [ ] Expanded benchmark datasets
- [ ] Interactive tutorials and examples
- [ ] Web deployment (Streamlit Cloud)

---

**Built with ❤️ for the multilingual NLP community**