# Semanscope Migration Documentation

## Overview

This document tracks the migration of semanscope from the research repository (`st_semantics`) to a standalone, production-ready public repository. The migration follows the successful maniscope pattern to create a self-contained Python package for multilingual semantic embedding visualization and analysis.

## Migration Context

- **Source**: `/home/papagame/projects/Proj-Geometry-of-Meaning/st_semantics`
- **Target**: `/home/papagame/projects/Proj-Geometry-of-Meaning/semanscope` (this repository)
- **Archive**: `/home/papagame/projects/Proj-Geometry-of-Meaning/st_semantics/archive/docs`
- **Conda Environment**: `semanscope`

## Critical Decisions (User Approved)

1. **Dataset Strategy**: Include ~80-100 representative datasets covering all categories (ACL-0 through ACL-6, NeurIPS-01 through NeurIPS-11). Provide download script for full collection.

2. **Config.py**: Keep as single 1,438-line file to minimize migration risk. Can refactor later if needed.

3. **UI Scope**: Migrate all 11 Streamlit pages for complete researcher functionality.

4. **Batch CLI**: Integrate as `semanscope.cli` package with entry points via pyproject.toml.

## Target Structure

```
semanscope/
├── semanscope/              # Core Python package
│   ├── __init__.py          # Package exports
│   ├── config.py            # Central configuration (1,438 lines)
│   ├── components/          # Analysis components
│   │   ├── __init__.py
│   │   ├── embedding_viz.py
│   │   ├── dimension_reduction.py
│   │   ├── plotting.py
│   │   ├── plotting_echarts.py
│   │   ├── semantic_affinity.py
│   │   ├── relational_affinity.py
│   │   ├── geometric_analysis.py
│   │   └── shared/
│   ├── models/              # Embedding model managers
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   ├── model_manager_v2o.py
│   │   ├── openrouter_model.py
│   │   ├── gemini_model.py
│   │   ├── voyage_model.py
│   │   └── ollama_model.py
│   ├── utils/               # Utilities
│   │   ├── __init__.py
│   │   ├── embedding_cache.py
│   │   ├── error_handling.py
│   │   ├── global_settings.py
│   │   ├── text_decomposer.py
│   │   ├── download_helpers.py
│   │   └── title_filename_helper.py
│   ├── services/            # External API integrations
│   │   ├── __init__.py
│   │   └── [service modules]
│   └── cli/                 # CLI tools (batch benchmark)
│       ├── __init__.py
│       ├── batch_benchmark_sa.py
│       ├── batch_benchmark_ra.py
│       ├── parse_results_sa.py
│       ├── parse_results_ra.py
│       └── figure_generation/
├── ui/                      # Streamlit UI
│   ├── Welcome.py           # Main entry point
│   ├── config.py            # UI-specific config
│   └── pages/               # All 11 Streamlit pages
│       ├── 0_🔧_Settings.py
│       ├── 1_🧭_Semanscope.py
│       ├── 2_📊_Semanscope_ECharts.py
│       ├── 3_⚖️_Semanscope_Compare.py
│       ├── 4_🌐_Semanscope_Multilingual.py
│       ├── 5_🔍_Semanscope_Zoom.py
│       ├── 6_📐_Semantic_Affinity.py
│       ├── 6_🔗_Relational_Affinity.py
│       ├── 8_🌐_Translator.py
│       ├── 9_📝_NSM_Prime_Words.py
│       └── 9_🖼️_Review_Images.py
├── data/                    # Representative datasets
│   ├── input/               # ~80-100 curated text datasets
│   │   └── README.md        # Dataset documentation
│   └── color-codes/         # Color coding files
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_semantic_affinity.py
│   ├── test_all_apis.py
│   ├── test_parallel_performance.py
│   └── test_echarts_auto_png.py
├── demo/                    # Usage examples
│   ├── basic_visualization.py
│   ├── batch_benchmark_example.py
│   └── semantic_affinity_demo.py
├── scripts/                 # Utility scripts
│   ├── download_datasets.py
│   ├── verify_setup.py
│   └── quick_test.sh
├── docs/                    # Documentation
│   ├── USAGE.md
│   ├── TROUBLESHOOTING.md
│   ├── GPU_SETUP.md
│   └── API.md
├── pyproject.toml           # Modern Python packaging
├── setup.py                 # Legacy support
├── requirements.txt         # Core dependencies
├── requirements-dev.txt     # Development dependencies
├── README.md                # Main documentation
├── LICENSE                  # MIT or appropriate
├── .env.example             # Environment variables template
├── .gitignore               # Git exclusions
└── run_app.py               # Streamlit launcher
```

## Migration Phases

### Phase 1: Archive Preparation
- Move Claude-generated .md files from st_semantics to archive
- Keep only essential documentation in source repository

### Phase 2: Core Package Migration
- Create `semanscope/` package structure
- Migrate config.py (update paths)
- Migrate components/ (update imports)
- Migrate models/ (update imports)
- Migrate utils/ (update imports, cache paths)
- Migrate services/ (update imports)
- Migrate batch_benchmark/ → cli/ (rename files, update imports)

### Phase 3: UI Migration
- Create `ui/` directory structure
- Copy Welcome.py and pages/
- Update all imports from `src.*` to `semanscope.*`
- Create ui/config.py for UI-specific settings
- Create run_app.py launcher

### Phase 4: Data Migration
- Select ~80-100 representative datasets
- Copy to data/input/
- Create data/input/README.md
- Copy color-code files to data/color-codes/
- Create scripts/download_datasets.py for full collection

### Phase 5: Testing & Documentation
- Migrate and update tests/
- Create demo scripts
- Write comprehensive README.md
- Create docs/ with USAGE, TROUBLESHOOTING, GPU_SETUP, API
- Create scripts/verify_setup.py and quick_test.sh

### Phase 6: Packaging & Configuration
- Create pyproject.toml with dependencies and entry points
- Create requirements.txt and requirements-dev.txt
- Create .env.example
- Create .gitignore
- Add LICENSE

### Phase 7: Validation & Testing
- Install package: `pip install -e ".[all]"`
- Run import tests
- Run pytest suite
- Test demo scripts
- Launch UI
- Test CLI tools

## File Migration Mapping

### Core Package Files
| Source (st_semantics) | Target (semanscope) | Action |
|----------------------|---------------------|--------|
| `src/config.py` | `semanscope/config.py` | Copy + update paths |
| `src/components/` | `semanscope/components/` | Copy + update imports |
| `src/models/` | `semanscope/models/` | Copy + update imports |
| `src/utils/` | `semanscope/utils/` | Copy + update imports |
| `src/services/` | `semanscope/services/` | Copy + update imports |
| `src/batch_benchmark/` | `semanscope/cli/` | Copy + rename + update imports |

### UI Files
| Source | Target | Action |
|--------|--------|--------|
| `src/Welcome.py` | `ui/Welcome.py` | Copy + update imports |
| `src/pages/*.py` | `ui/pages/*.py` | Copy + update imports |

### Data Files
| Source | Target | Action |
|--------|--------|--------|
| `data/input/` (selected) | `data/input/` | Copy curated selection |
| `data/*.color-code.csv` | `data/color-codes/` | Copy all |

### Test Files
| Source | Target | Action |
|--------|--------|--------|
| `tests/*.py` | `tests/*.py` | Copy + update imports |

## Import Path Changes

All imports must be updated:
- `from src.` → `from semanscope.`
- `from src.components.` → `from semanscope.components.`
- `from src.models.` → `from semanscope.models.`
- `from src.utils.` → `from semanscope.utils.`
- etc.

## Cache Path Updates

Embedding cache paths in config.py must be updated to work with new structure while maintaining access to existing cache.

## CLI Entry Points

Entry points to be configured in pyproject.toml:
- `semanscope-ui` → launches Streamlit UI
- `semanscope-benchmark-sa` → Semantic Affinity batch benchmark
- `semanscope-benchmark-ra` → Relational Affinity batch benchmark

## Representative Datasets

Selection covers all categories (~80-100 files):
- **ACL-0**: Zinets, Radicals (Chinese morphology)
- **ACL-1**: Alphabets (English, Chinese, Spanish, Arabic, Korean, Japanese)
- **ACL-2**: PeterG vocabulary (ALL, Adj, Verbs - English, Chinese, German)
- **ACL-3**: Morphological networks (子-network, haus-arbeit)
- **ACL-4**: Numbers (English, Chinese), emotions, animals, physics-math
- **ACL-5**: Poems (Li Bai, Du Fu, Frost, Wordsworth, Shelley - representative)
- **ACL-6**: Emoji, Pictographs
- **NeurIPS-01 to NeurIPS-11**: Latest v2.5 versions for each category

## Verification Checklist

After migration, verify:
- [ ] Package imports work: `import semanscope`
- [ ] Core components accessible: `from semanscope.components import SemanticAffinity`
- [ ] Models load: `from semanscope.models import get_model; get_model("LaBSE")`
- [ ] Config accessible: `from semanscope import config`
- [ ] UI launches: `python run_app.py`
- [ ] Tests pass: `pytest tests/`
- [ ] Demo scripts run: `python demo/*.py`
- [ ] CLI works: `semanscope-benchmark-sa --help`
- [ ] Documentation complete
- [ ] .gitignore excludes archive/, logs/, cache/
- [ ] .env.example provided
- [ ] License included
- [ ] No st_semantics imports remain
- [ ] No absolute paths to old location
- [ ] Cache paths updated
- [ ] Embedding cache accessible

## Risk Mitigation

1. **Import Path Changes**: Systematic find/replace, test each module
2. **Cache Path Migration**: Update config, consider symlinks if needed
3. **Configuration Complexity**: Keep 1,438-line config.py as-is initially
4. **Dataset Size**: Include only representative datasets, provide download script
5. **Dependency Conflicts**: Fresh conda environment, pinned versions

## Success Criteria

Migration is successful when:
1. Package is pip-installable: `pip install -e .`
2. Core functionality works: SA/RA metrics compute correctly
3. UI launches and visualizes embeddings
4. CLI tools run batch benchmarks
5. Tests pass in new structure
6. Documentation is comprehensive and clear
7. No dependencies on parent repository
8. Repository follows Python packaging best practices
9. Ready for public release on GitHub

## Post-Migration Tasks

1. Git initialization and first commit
2. Create GitHub repository
3. Documentation polish (screenshots, examples)
4. Release preparation (tag v1.0.0)
5. Consider PyPI publication

## Notes

- Follow maniscope migration pattern
- Use `semanscope` conda environment
- Archive Claude-generated docs in st_semantics/archive/docs
- Exclude archive/, devs/, research/ from source
- Plan transcript available at: `/home/papagame/.claude/projects/-home-papagame-projects-Proj-Geometry-of-Meaning-semanscope/ec053dd0-bce0-467e-a6d0-c4e158b1a440.jsonl`
