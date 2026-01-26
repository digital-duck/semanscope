# Semanscope Datasets

This directory contains representative datasets for semantic embedding visualization and benchmarking.

## Dataset Categories

### ACL-0: Chinese Morphology
- **Zinets**: Chinese character components (子-network)
- **Radicals**: Kangxi radicals for morphological analysis

### ACL-1: Alphabets
Letter names across multiple writing systems:
- English (enu), Chinese (chn), Spanish (spa), Arabic (ara)
- Korean (kor), Japanese (jpn), Russian (rus), Turkish (tur)
- French (fra), German (deu), Persian (fas), Greek (grk)
- Hebrew (heb), Hindi (hin), Armenian (hye), Georgian (kat)
- Thai (tha), Vietnamese (vie)

### ACL-2: PeterG Vocabulary
Core vocabulary sets from Natural Semantic Metalanguage research:
- **ALL**: Complete semantic primes (~65 words)
- **Adj**: Adjectives subset
- **Verb**: Verbs subset
Available in English (enu), Chinese (chn), German (deu)

### ACL-3: Morphological Networks
Word formation networks:
- Chinese 子-network (child/son morphology)
- German haus-arbeit network (house-work compounds)

### ACL-4: Semantic Categories
Thematic word lists:
- **Numbers**: Number words (1-100) in English and Chinese
- **Emotions**: Emotion vocabulary
- **Animals**: Animal names
- **Physics-Math**: Scientific terminology

### ACL-5: Poetry Corpora
Poetic texts for semantic analysis:
- **Chinese**: Li Bai (李白), Du Fu (杜甫)
- **English**: Robert Frost, William Wordsworth, Percy Shelley
- **German**: Goethe, Schiller

### ACL-6: Visual Semantics
- **Emoji**: Emoji characters and descriptions
- **Pictographs**: Pictographic symbols

### NeurIPS Benchmark Datasets (v2.5)

Research-grade benchmarks for Semantic Affinity (SA) and Relational Affinity (RA) metrics:

1. **NeurIPS-01**: family-relations (kinship terms)
2. **NeurIPS-02**: royalty-hierarchy (social ranks)
3. **NeurIPS-03**: gendered-occupations (profession terms)
4. **NeurIPS-04**: comparative-superlative (degree adjectives)
5. **NeurIPS-05**: antonym-pairs (opposites)
6. **NeurIPS-06**: verb-tenses (temporal forms)
7. **NeurIPS-07**: plural-singular (number forms)
8. **NeurIPS-08**: capitals-countries (geographic relations)
9. **NeurIPS-09**: currency-countries (economic relations)
10. **NeurIPS-10**: sports-equipment (thematic associations)
11. **NeurIPS-11**: food-categories (semantic taxonomies)

Each benchmark includes:
- **-SA.csv**: Semantic Affinity test format
- **-RA.csv**: Relational Affinity test format

## File Formats

### Text Files (.txt)
Simple newline-delimited word lists:
```
word1
word2
word3
```

### CSV Files
Structured data for benchmarking:
- **SA format**: word1, word2 pairs for semantic similarity
- **RA format**: word1, word2, word3, word4 quadruples for relational analogy

### Color Code Files (.color-code.csv)
Optional visualization color mappings:
```
word,color
example,#FF5733
```

## Language Codes

Semanscope uses 3-letter ISO 639-3 inspired codes:
- `enu` = English
- `chn` = Chinese (Mandarin)
- `spa` = Spanish
- `ara` = Arabic
- `deu` = German
- `fra` = French
- `jpn` = Japanese
- `kor` = Korean
- `rus` = Russian
- `tur` = Turkish

(See `semanscope.config.LANGUAGE_CODE_MAP` for complete mapping)

## Dataset Statistics

- **Total files**: ~60+ representative datasets
- **Languages**: 15+ major languages
- **Categories**: 7 main categories (ACL-0 through ACL-6)
- **Benchmarks**: 11 NeurIPS datasets × 2 metrics = 22 benchmark files
- **Color codes**: 26 visualization mappings

## Full Dataset Collection

This directory contains a curated selection of representative datasets. To download the complete collection (2000+ files):

```bash
# Option 1: Using download script
python scripts/download_datasets.py --full

# Option 2: From GitHub releases
# (Coming soon - check repository releases page)

# Option 3: From original source
# Contact repository maintainers for access to full archive
```

## Usage Examples

### Loading a dataset in Python
```python
from pathlib import Path
from semanscope.config import DATA_PATH

# Read a simple word list
dataset_path = DATA_PATH / "input" / "ACL-1-Alphabets-enu.txt"
with open(dataset_path) as f:
    words = [line.strip() for line in f]

# Read a benchmark CSV
import pandas as pd
benchmark_path = DATA_PATH / "input" / "NeurIPS-01-family-relations-v2.5-SA.csv"
df = pd.read_csv(benchmark_path)
```

### Using in Streamlit UI
1. Launch UI: `python run_app.py` or `semanscope-ui`
2. Navigate to any Semanscope page
3. Select dataset from dropdown (automatically populated from this directory)

### Batch Benchmarking
```bash
# Semantic Affinity benchmark
semanscope-benchmark-sa --dataset NeurIPS-01-family-relations-v2.5-SA.csv --model LaBSE

# Relational Affinity benchmark
semanscope-benchmark-ra --dataset NeurIPS-01-family-relations-v2.5-RA.csv --model LaBSE
```

## Adding Custom Datasets

To add your own datasets:

1. **Create word list** (.txt format):
   ```
   word1
   word2
   word3
   ```

2. **Follow naming convention**:
   - `CategoryName-SubCategory-language.txt`
   - Example: `MyData-Animals-enu.txt`

3. **Place in this directory**: `data/input/`

4. **Optional color coding**: Create matching `.color-code.csv`:
   ```csv
   word,color
   cat,#FF5733
   dog,#33C3FF
   ```

5. **Restart UI**: Dataset will appear in dropdown automatically

## Dataset Attribution

- **ACL datasets**: Research collections from computational linguistics studies
- **NeurIPS benchmarks**: Purpose-built evaluation datasets for SA/RA metrics
- **PeterG vocabulary**: Based on Natural Semantic Metalanguage (NSM) theory
- **Poetry corpora**: Public domain literary works

## License

Datasets included are either:
- Public domain (poetry, alphabets)
- Research use (benchmarks)
- Creative Commons (where applicable)

See individual dataset documentation for specific licensing.

## Citation

If you use these datasets in research, please cite:

```bibtex
@software{semanscope2026,
  title={Semanscope: Multilingual Semantic Embedding Visualization Toolkit},
  author={Semanscope Contributors},
  year={2026},
  url={https://github.com/semanscope/semanscope}
}
```

## Questions?

- **Missing dataset**: Use download script or contact maintainers
- **Custom datasets**: See "Adding Custom Datasets" above
- **Format questions**: Check examples in this directory
- **Bug reports**: https://github.com/semanscope/semanscope/issues
