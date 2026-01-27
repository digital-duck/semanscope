# Semanscope - Input Data Format

This folder contains input datasets for semantic geometry analysis using the Semanscope tool.

## File Format Specification

### CSV Format (Required)
All input data files must be in **CSV format** with exactly **4 columns**:

```csv
word,domain,type,note
```

### Column Definitions

| Column | Description | Purpose | Examples |
|--------|-------------|---------|----------|
| **word** | The term to analyze | Primary data for embedding analysis | `force`, `+`, `孔子`, `work` |
| **domain** | Semantic domain category | Color coding and grouping | `mechanics`, `operations`, `morphological_子` |
| **type** | Classification type | Additional categorization | `concept`, `symbol`, `particle`, `character` |
| **note** | Brief description | Context and documentation | `push or pull`, `addition operator` |

### Color Coding System

The **domain** column maps to colors defined in `src/config.py`:

#### Mathematical & Physics Domains
- `numbers` → 🔴 Bright red (numerical spiral pattern)
- `operations` → 🟠 Orange red (mathematical operators)
- `algebra` → 🔵 Royal blue (variables, equations)
- `geometry` → 🔵 Blue (shapes, spatial concepts)
- `mechanics` → 🟢 Forest green (force, energy)
- `quantum` → 🟣 Dark violet (quantum mechanics)
- `constants` → 🟠 Dark orange (π, e, c, h)

#### Linguistic Domains
- `people` → 🔴 Royal blue (persons, agents)
- `nature` → 🟢 Forest green (animals, plants)
- `objects` → 🟠 Dark orange (tools, furniture)
- `morphological_子` → 🟣 Purple (Chinese character families)
- `morphological_work` → 🟣 Purple (English derivational families)

#### Custom Domains
Add new domains to `SEMANTIC_DOMAIN_COLORS` in `src/config.py` for custom color mapping.

## File Naming Convention

```
{PROJECT}-{DATASET}-{VERSION}-{LANGUAGE}.txt
```

**Examples:**
- `ACL-word-v2-enu.txt` - ACL paper core vocabulary (English)
- `ACL-word-v2-chn.txt` - ACL paper core vocabulary (Chinese)
- `ACL-Physics-v2-enu.txt` - Mathematical and physics concepts
- `ACL-network-子-v2-chn.txt` - Chinese morphological network
- `ACL-numbers-enu.txt` - Numerical concepts dataset

## Sample Datasets

### 1. ACL Core Vocabulary (Multilingual)
Cross-linguistic semantic analysis datasets:
- **English**: `ACL-word-v2-enu.txt` (278 words)
- **Chinese**: `ACL-word-v2-chn.txt` (265 words)
- **German**: `ACL-word-v2-deu.txt` (265 words)

**Domains**: people, nature, objects, abstract, activity, places, time, body, food

### 2. Mathematical & Physics Concepts
Comprehensive scientific vocabulary:
- **Physics v2**: `ACL-Physics-v2-enu.txt` (263 terms)

**Domains**: numbers, operations, algebra, geometry, calculus, mechanics, quantum, etc.

### 3. Morphological Networks
Language-specific word formation patterns:
- **Chinese**: `ACL-network-子-v2-chn.txt` (123 characters)
- **English**: `ACL-network-work-light-v2-enu.txt` (62 words)
- **German**: `ACL-network-haus-arbeit-v2-deu.txt` (90 compounds)

**Domains**: morphological_子, morphological_work, morphological_light, etc.

### 4. Numerical Progression
Mathematical spiral geometry analysis:
- **English**: `ACL-numbers-enu.txt` (92 terms)

**Domains**: numbers, operations, symbols, constants

## Data Quality Guidelines

### Word Selection
- ✅ **Single words or established terms**: `force`, `E=mc²`, `子`
- ✅ **Domain-specific vocabulary**: technical terms, scientific concepts
- ✅ **Cross-linguistic equivalents**: parallel terms across languages
- ❌ **Full sentences**: avoid complete sentences
- ❌ **Very rare terms**: focus on established vocabulary
- ❌ **Highly ambiguous words**: prefer clear, unambiguous terms

### Domain Assignment
- **Consistent categorization**: Use established domain categories
- **Semantic coherence**: Group semantically related terms
- **Avoid overlap**: Each word should have one primary domain
- **Document rationale**: Use descriptive domain names

### Type Classification
Common type categories:
- `concept` - Abstract ideas, principles
- `symbol` - Mathematical/logical symbols
- `particle` - Physical particles, atoms
- `property` - Physical/chemical properties
- `process` - Actions, procedures
- `unit` - Measurement units
- `character` - Chinese characters
- `word` - Natural language words

## Usage in Semanscope

1. **Load Dataset**: Select file from data input folder
2. **Embedding Generation**: Words are converted to embeddings using selected model
3. **Geometric Analysis**: PHATE/other methods create 2D/3D visualization
4. **Color Mapping**: Domain column determines point colors
5. **Interactive Exploration**: 3D visualization with domain-based coloring

## Research Applications

### Cross-Linguistic Analysis
Compare geometric organization across languages using parallel datasets.

### Domain-Specific Studies
Analyze how specialized vocabularies (mathematics, physics, linguistics) organize in embedding space.

### Morphological Research
Study language-specific word formation patterns through geometric analysis.

### Educational Applications
Visualize conceptual relationships for learning optimization.

## File Creation Tips

### CSV Formatting
- **Header required**: First line must be `word,domain,type,note`
- **No spaces in domain names**: Use underscores (`morphological_work`)
- **Consistent encoding**: UTF-8 for multilingual support
- **Quote complex terms**: Use quotes for terms with commas: `"mass-energy"`

### Domain Strategy
- **Start broad**: Use major categories (numbers, mechanics, people)
- **Refine gradually**: Add subcategories as needed (quantum, thermodynamics)
- **Color coordination**: Consider visual clarity in final visualization
- **Documentation**: Keep notes about domain rationale

### Example Entry
```csv
word,domain,type,note
E=mc²,relativity,equation,Einstein's mass-energy equivalence
孔子,morphological_子,character,Confucius - philosophical compound
derivative,calculus,concept,rate of change in calculus
```

---

## Technical Integration

This data format integrates with:
- **Embedding Models**: Sentence-BERT Multilingual, BGE-M3, etc.
- **Dimensionality Reduction**: PHATE, UMAP, t-SNE, TriMap, PaCMAP
- **Visualization**: 2D/3D interactive plots with domain-based coloring
- **Export**: Results can be saved for academic papers and analysis

**For technical details see**: `src/config.py` (color mappings) and `src/components/` (processing logic)

---

*This format enables systematic geometric analysis of semantic organization across languages, domains, and conceptual categories - supporting research in computational semantics, multilingual NLP, and cognitive science.*