"""
PHASE 2 RESEARCH: Manifold-Based Unified Metric
================================================

**Status**: 🚧 PLACEHOLDER FOR POST-NEURIPS RESEARCH (After May 2026)
**Created**: December 31, 2025 (New Year's Eve)
**Target**: Phase 2 exploration (2027)

**Research Context**:
This page implements Phase 2 research directions outlined in:
/home/papagame/projects/Proj-ZiNets/zinets/docs/conference/NeurIPS/2026/relational-affinity/10-project-plan.md

See ADDENDUM: Topology Research Program (2026-2028) in project plan for:
- Multi-year research roadmap
- Learning resources and timeline
- Integration with NeurIPS Phase 1 findings
- Team collaboration notes (Wen + Claude + Gemini)

THEORETICAL FOUNDATION
======================

Current Approach (Phase 1 - NeurIPS 2026):
- SA (Semantic Affinity): Euclidean/cosine distance between points
- RA (Relational Affinity): Euclidean alignment of vectors
- Simple, interpretable, empirically validated

Proposed Approach (Phase 2 - 2027):
- Unified Manifold Metric: Diffusion-based distance on intrinsic geometry
- Captures BOTH semantic clustering AND relational structure
- Variation-agnostic: doesn't matter if differences come from language, model, or category

KEY INSIGHT (Wen's Observation):
--------------------------------
"Language, model, dataset are just 3 variational dimensions, but in multidimensional
embedding space, it does not care which variation causes different manifold geometry."

This means:
- English vs Chinese → manifold curvature difference
- LaBSE vs Voyage → manifold structure difference
- Animal gender vs Comparative → manifold topology difference

All are manifestations of the SAME underlying geometric phenomenon!

A truly unified metric would measure intrinsic manifold structure regardless of
the SOURCE of variation.

MATHEMATICAL FRAMEWORK
======================

Diffusion Distance (Coifman & Lafon, 2006):
-------------------------------------------

D_t(x, y) = ||p_t(x, ·) - p_t(y, ·)||

where:
- p_t(x, ·) is the heat kernel diffusion from point x at time t
- ||·|| is typically L2 norm
- t is diffusion time parameter

Physical Interpretation:
- Imagine heat spreading from point x across the manifold
- After time t, heat distribution is p_t(x, ·)
- Points with similar distributions are "close" on manifold
- Captures geodesic distance (shortest path on curved surface)

Connection to PHATE:
-------------------
PHATE already computes diffusion distances for visualization!

PHATE pipeline:
1. Build affinity matrix: K(x,y) = exp(-||x-y||²/σ²)
2. Normalize to get Markov matrix: P = D^(-1)K
3. Diffuse: P^t (apply t steps of diffusion)
4. Compute potential distance
5. Embed to 2D/3D

We can REUSE step 3 for metric calculation instead of just visualization!

UNIFIED METRIC CONCEPT
=======================

Instead of separate SA and RA, define:

Manifold Relational Coherence (MRC):
------------------------------------

MRC(pairs, embedding) = consistency of relational structure on manifold

Intuition:
- If pairs form consistent geometric pattern on manifold → high MRC
- Pattern can be:
  * Tight clustering (captures current SA)
  * Parallel vectors (captures current RA)
  * Geodesic alignment (new: curved-space alignment)
  * Curvature consistency (new: shape preservation)

Potential Formulation:
---------------------

MRC = f(
    diffusion_clustering(pairs),      # Replaces SA
    diffusion_alignment(vectors),      # Replaces RA
    geodesic_parallelism(pairs),       # New: curved-space concept
    curvature_consistency(pairs)       # New: shape-based concept
)

RESEARCH QUESTIONS FOR PHASE 2
===============================

1. Does diffusion-based RA change model rankings?
   - Will LaBSE still dominate?
   - Do BERT models perform better/worse?

2. Does manifold metric unify SA and RA?
   - Can single score capture both clustering and alignment?
   - Or do we still need separate aspects?

3. What does diffusion time parameter t reveal?
   - Small t: local structure (neighborhoods)
   - Large t: global structure (topology)
   - Optimal t for different categories?

4. Does it explain SA-RA orthogonality?
   - Current finding: 58.3% Quadrant II (high SA, low RA)
   - Manifold view: maybe Euclidean metric misses curved alignment?
   - Could models have good manifold structure but poor Euclidean structure?

5. Cross-linguistic manifold geometry:
   - Does Chinese have different intrinsic curvature than English?
   - Morphological regularity → smoother manifold?
   - Lexical irregularity → more curved manifold?

6. Can we detect "shape similarity" (Phase 3)?
   - Do family relations form star-shaped manifolds?
   - Do comparative forms create linear manifolds?
   - Geometric patterns beyond point-to-point distance?

IMPLEMENTATION ROADMAP
======================

Step 1: Extract PHATE Diffusion Operator (Learning Phase)
----------------------------------------------------------
- Study phate package source code
- Understand how it computes diffusion matrix
- Extract reusable diffusion function

Step 2: Compute Diffusion Distance Between Points
-------------------------------------------------
- Replace Euclidean distance with diffusion distance
- Compare results on pilot dataset (e.g., DS04 comparative)

Step 3: Diffusion-Based Relational Alignment
--------------------------------------------
- For vectors v1 = w1a→w1b, v2 = w2a→w2b
- Compute diffusion distance between vector endpoints
- Measure alignment on curved manifold (geodesic parallelism)

Step 4: Unified Manifold Coherence Score
-----------------------------------------
- Combine clustering and alignment into single metric
- Validate against Phase 1 findings (should correlate but add insight)

Step 5: Parameter Sensitivity Analysis
---------------------------------------
- Test different diffusion times t
- Test different kernel bandwidths σ
- Find optimal settings per category

LEARNING RESOURCES
==================

Books to Study (Wen's learning list):
-------------------------------------
1. "Riemannian Manifolds: An Introduction to Curvature" (John M. Lee)
   → Understand curved spaces, geodesics, curvature

2. "Diffusion Maps" (Coifman & Lafon, 2006 - paper)
   → Original diffusion distance formulation

3. "PHATE: A Dimensionality Reduction Method" (Moon et al., 2019)
   → How PHATE uses diffusion for visualization

4. "Geometric Deep Learning" (Bronstein et al., 2021)
   → Modern perspective on geometry in ML

Papers to Read:
--------------
1. Coifman & Lafon (2006) - "Diffusion Maps"
2. Moon et al. (2019) - "Visualizing Structure and Transitions in High-Dimensional Biological Data"
3. Von Luxburg (2007) - "A Tutorial on Spectral Clustering"
4. Belkin & Niyogi (2003) - "Laplacian Eigenmaps for Dimensionality Reduction"

CODE STUBS AND PSEUDOCODE
==========================
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings

# Placeholder imports (will be needed in Phase 2)
try:
    from phate import PHATE
    import graphtools as gt  # PHATE's graph construction tool
    PHATE_AVAILABLE = True
except ImportError:
    PHATE_AVAILABLE = False

# ==============================================================================
# PHASE 2 RESEARCH SCAFFOLDING - NOT FOR PRODUCTION USE YET
# ==============================================================================

st.set_page_config(page_title="Manifold Metric Research", page_icon="🌀", layout="wide")

st.title("🌀 Manifold-Based Unified Metric")
st.caption("**Status**: Research Placeholder (Phase 2 - Post-NeurIPS 2026)")

st.info("""
📚 **Research Phase**: This page is a scaffold for post-NeurIPS exploration.

**Timeline**:
- **Now - May 2026**: Focus on NeurIPS paper with Euclidean metrics
- **May - Dec 2026**: Learn manifold geometry theory
- **2027**: Implement and validate diffusion-based metrics

**Purpose**: Serve as reminder and research notebook for Phase 2 work.

**Research Context**: This page implements Phase 2 directions from the NeurIPS 2026 project plan:
- 📄 `/home/papagame/projects/Proj-ZiNets/zinets/docs/conference/NeurIPS/2026/relational-affinity/10-project-plan.md`
- See **ADDENDUM: Topology Research Program (2026-2028)** for multi-year roadmap
""")

if not PHATE_AVAILABLE:
    st.warning("⚠️ PHATE not available. Install with: `pip install phate`")

# ==============================================================================
# THEORETICAL FRAMEWORK DISPLAY
# ==============================================================================

with st.expander("📖 Theoretical Framework", expanded=True):
    st.markdown("""
    ## Current Limitations of Euclidean Metrics

    **Semantic Affinity (SA)**: Measures Euclidean distance between word embeddings
    - ✅ Simple and interpretable
    - ❌ Assumes flat geometry (may miss curved structure)

    **Relational Affinity (RA)**: Measures cosine similarity of relational vectors
    - ✅ Captures directional alignment
    - ❌ Euclidean straight-line assumption (geodesics may curve)

    ## Why Manifold Metrics?

    **Key Insight**: Embeddings lie on curved manifolds, not flat Euclidean space!

    - PHATE visualization reveals intrinsic manifold structure
    - Measurement should match geometry (diffusion distance ≈ geodesic distance)
    - Could explain SA-RA orthogonality (good Euclidean clustering ≠ good manifold alignment)

    ## Unified Metric Vision

    Instead of separate SA and RA, define:

    **Manifold Relational Coherence (MRC)**: Single score capturing both clustering and alignment

    - Variation-agnostic: Works regardless of source (language, model, category)
    - Intrinsic geometry: Respects manifold curvature
    - Theoretically principled: Based on diffusion geometry
    """)

with st.expander("🔬 Research Questions for Phase 2"):
    st.markdown("""
    1. **Model Rankings**: Do diffusion metrics change LaBSE dominance?
    2. **Unification**: Can we merge SA and RA into single manifold score?
    3. **Orthogonality**: Does manifold view explain SA-RA independence?
    4. **Cross-Linguistic**: Do languages have different intrinsic curvature?
    5. **Morphology**: Does compositional regularity → smoother manifolds?
    6. **Optimal Parameters**: What diffusion time `t` works best per category?
    7. **Shape Similarity**: Can we detect geometric patterns (stars, loops, chains)?
    8. **Transfer Learning**: Do metaphorical domains share manifold structure?
    """)

# ==============================================================================
# PSEUDOCODE AND CONCEPTUAL IMPLEMENTATION
# ==============================================================================

st.header("💡 Conceptual Implementation (Pseudocode)")

st.code("""
# ==============================================================================
# DIFFUSION DISTANCE COMPUTATION (To be implemented in Phase 2)
# ==============================================================================

def compute_diffusion_affinity(embeddings: np.ndarray,
                                sigma: float = 1.0) -> np.ndarray:
    '''
    Step 1: Build affinity matrix using Gaussian kernel

    K(i,j) = exp(-||x_i - x_j||² / σ²)

    Args:
        embeddings: (N, D) array of word embeddings
        sigma: kernel bandwidth parameter

    Returns:
        K: (N, N) affinity matrix
    '''
    # TODO: Implement in Phase 2
    # Compute pairwise distances
    # Apply Gaussian kernel
    # Return affinity matrix
    pass


def compute_diffusion_operator(affinity: np.ndarray,
                                alpha: float = 1.0) -> np.ndarray:
    '''
    Step 2: Build diffusion operator (Markov transition matrix)

    P = D^(-α) K D^(-α)
    where D is degree matrix

    Args:
        affinity: (N, N) affinity matrix from previous step
        alpha: anisotropy parameter (0=directed, 1=symmetric)

    Returns:
        P: (N, N) Markov transition matrix
    '''
    # TODO: Implement in Phase 2
    # Compute degree matrix
    # Normalize affinity → Markov matrix
    # Handle alpha parameter
    pass


def compute_diffusion_distance(P: np.ndarray,
                                t: int = 1) -> np.ndarray:
    '''
    Step 3: Compute diffusion distance at time t

    D_t(i,j) = ||p_t(i, ·) - p_t(j, ·)||
    where p_t = P^t (t-step diffusion)

    Args:
        P: (N, N) Markov transition matrix
        t: diffusion time (number of steps)

    Returns:
        D_t: (N, N) diffusion distance matrix
    '''
    # TODO: Implement in Phase 2
    # Compute P^t (matrix power)
    # For each pair (i,j):
    #   distance = ||P^t[i,:] - P^t[j,:]||_2
    pass


def manifold_semantic_affinity(embeddings: np.ndarray,
                                word_pairs: List[Tuple[str, str]],
                                t: int = 1,
                                sigma: float = 1.0) -> float:
    '''
    Manifold-based Semantic Affinity (replaces Euclidean SA)

    Measures how tightly word pairs cluster on manifold geometry

    Args:
        embeddings: Word embeddings
        word_pairs: List of (word1, word2) pairs
        t: diffusion time
        sigma: kernel bandwidth

    Returns:
        mSA: Manifold Semantic Affinity score [0, 1]
    '''
    # TODO: Implement in Phase 2

    # 1. Compute diffusion distances
    # K = compute_diffusion_affinity(embeddings, sigma)
    # P = compute_diffusion_operator(K)
    # D_t = compute_diffusion_distance(P, t)

    # 2. For each pair, compute manifold distance
    # mSA_scores = []
    # for (w1, w2) in word_pairs:
    #     i, j = get_indices(w1, w2)
    #     pair_distance = D_t[i, j]
    #     mSA_scores.append(1.0 / (1.0 + pair_distance))  # Convert to affinity

    # 3. Return average
    # return np.mean(mSA_scores)
    pass


def manifold_relational_affinity(embeddings: np.ndarray,
                                  relations: List[Tuple[str, str, str, str]],
                                  t: int = 1,
                                  sigma: float = 1.0) -> float:
    '''
    Manifold-based Relational Affinity (replaces Euclidean RA)

    Measures geodesic parallelism of relational vectors on manifold

    Args:
        embeddings: Word embeddings
        relations: List of (w1a, w1b, w2a, w2b) for parallel relations
        t: diffusion time
        sigma: kernel bandwidth

    Returns:
        mRA: Manifold Relational Affinity score [0, 1]
    '''
    # TODO: Implement in Phase 2

    # Concept: Instead of Euclidean vector parallelism,
    # measure if geodesic paths have similar "shape"

    # For relation: king→queen vs man→woman
    # - Path 1: Geodesic from king to queen on manifold
    # - Path 2: Geodesic from man to woman on manifold
    # - Do these paths have similar curvature?
    # - Do they maintain similar distances at each step?

    # Potential approach:
    # 1. Compute diffusion distances
    # 2. For each relation pair:
    #    v1_start, v1_end = get_embeddings(w1a, w1b)
    #    v2_start, v2_end = get_embeddings(w2a, w2b)
    #
    #    # Manifold distance along each relation
    #    d1 = D_t[idx(w1a), idx(w1b)]
    #    d2 = D_t[idx(w2a), idx(w2b)]
    #
    #    # Manifold alignment (do endpoints align on manifold?)
    #    alignment = compute_geodesic_parallelism(v1_start, v1_end,
    #                                             v2_start, v2_end, D_t)
    # 3. Return consistency measure
    pass


def unified_manifold_coherence(embeddings: np.ndarray,
                                word_pairs: List[Tuple[str, str]],
                                t: int = 1,
                                sigma: float = 1.0) -> Dict[str, float]:
    '''
    UNIFIED METRIC: Combines semantic clustering and relational alignment

    Returns both aspects as components of single manifold analysis

    Args:
        embeddings: Word embeddings
        word_pairs: List of (word1, word2) pairs
        t: diffusion time
        sigma: kernel bandwidth

    Returns:
        {
            'manifold_clustering': mSA score,
            'manifold_alignment': mRA score,
            'geodesic_consistency': geometric pattern score,
            'curvature_similarity': shape preservation score,
            'unified_score': weighted combination
        }
    '''
    # TODO: Implement in Phase 2

    # This is the BIG GOAL: Single metric that captures both SA and RA
    # as aspects of the same manifold geometry

    # return {
    #     'manifold_clustering': manifold_semantic_affinity(...),
    #     'manifold_alignment': manifold_relational_affinity(...),
    #     'geodesic_consistency': compute_geodesic_patterns(...),
    #     'curvature_similarity': compute_shape_preservation(...),
    #     'unified_score': weighted_combination(...)
    # }
    pass

# ==============================================================================
# LEVERAGING PHATE'S INTERNAL DIFFUSION OPERATOR
# ==============================================================================

def extract_phate_diffusion(embeddings: np.ndarray,
                            knn: int = 5,
                            t: int = 1) -> np.ndarray:
    '''
    Extract diffusion operator from PHATE for metric computation

    PHATE already computes diffusion! We just need to access it.

    Args:
        embeddings: (N, D) word embeddings
        knn: k-nearest neighbors for graph construction
        t: diffusion time (PHATE uses adaptive t, we might override)

    Returns:
        D_t: (N, N) diffusion distance matrix
    '''
    if not PHATE_AVAILABLE:
        raise ImportError("PHATE not installed")

    # TODO: Implement in Phase 2

    # Study PHATE source code to understand:
    # 1. How it builds the graph (graphtools)
    # 2. How it computes diffusion operator
    # 3. How to extract intermediate results

    # Conceptual approach:
    # phate_op = PHATE(knn=knn, t=t)
    # phate_op.fit(embeddings)
    #
    # # Access internal diffusion operator
    # # (Need to study PHATE API - this is pseudocode)
    # diff_op = phate_op.diff_op  # This might not be the actual attribute name
    #
    # # Compute pairwise diffusion distances
    # D_t = compute_diffusion_distance_from_operator(diff_op)

    pass

""", language='python')

# ==============================================================================
# VISUALIZATION OF CONCEPT
# ==============================================================================

st.header("🎨 Conceptual Visualization")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Euclidean Distance (Current)

    Straight-line "chord" distance through space:
    """)
    st.code("""
    A --------chord-------- B
         (straight line)

    - Simple to compute
    - Ignores manifold structure
    - May overestimate/underestimate
    """, language='text')

with col2:
    st.markdown("""
    ### Manifold Distance (Proposed)

    Geodesic distance along curved surface:
    """)
    st.code("""
         ___B___
       /         \\
      /           \\
     A  (geodesic  \\
         along       \\
         curve)

    - Respects intrinsic geometry
    - Captures true structure
    - More theoretically principled
    """, language='text')

st.info("""
**Key Difference**: Two words might be:
- Far in Euclidean space but close on manifold (high curvature region)
- Close in Euclidean space but far on manifold (manifold wraps around)

Diffusion distance reveals the **intrinsic structure** that Euclidean distance misses.
""")

# ==============================================================================
# PARAMETER EXPLORATION INTERFACE (Placeholder)
# ==============================================================================

st.header("🎛️ Parameter Exploration (Phase 2)")

with st.expander("Diffusion Parameters to Study"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Diffusion Time (t)")
        t_param = st.slider("Time steps", 1, 20, 3,
                           help="Small t: local structure, Large t: global structure",
                           disabled=True)
        st.caption("⏸️ Disabled - Phase 2 research")

    with col2:
        st.markdown("### Kernel Bandwidth (σ)")
        sigma_param = st.slider("Sigma", 0.1, 5.0, 1.0,
                                help="Controls neighborhood size",
                                disabled=True)
        st.caption("⏸️ Disabled - Phase 2 research")

    with col3:
        st.markdown("### Anisotropy (α)")
        alpha_param = st.slider("Alpha", 0.0, 1.0, 1.0,
                               help="0: directed graph, 1: symmetric",
                               disabled=True)
        st.caption("⏸️ Disabled - Phase 2 research")

st.info("""
**Research Question**: Do optimal parameters differ by:
- Semantic category? (family relations vs comparatives)
- Language? (morphologically rich vs poor)
- Model architecture? (BERT vs sentence transformers)
""")

# ==============================================================================
# COMPARISON WITH PHASE 1 METRICS
# ==============================================================================

st.header("📊 Phase 1 vs Phase 2 Comparison (Future)")

st.markdown("""
When we implement diffusion metrics, we'll create comparison tables like:
""")

comparison_df = pd.DataFrame({
    'Category': ['Animal Gender', 'Comparative', 'Family Relations'],
    'Model': ['LaBSE', 'Voyage-3', 'S-BERT'],
    'SA (Euclidean)': [0.45, 0.72, 0.68],
    'mSA (Diffusion)': ['?', '?', '?'],
    'RA (Euclidean)': [0.62, 0.82, 0.71],
    'mRA (Diffusion)': ['?', '?', '?']
})

st.dataframe(comparison_df, width=800)

st.markdown("""
**Key Questions**:
1. Do rankings change? (Is LaBSE still #1?)
2. Does mSA correlate with SA? Or reveal new patterns?
3. Does mRA explain high-SA/low-RA cases (Quadrant II)?
4. Can unified score combine both aspects?
""")

# ==============================================================================
# LEARNING ROADMAP
# ==============================================================================

st.header("📚 Learning Roadmap (Wen)")

learning_plan = {
    "Jan-Feb 2026": [
        "Finish NeurIPS paper with Euclidean metrics",
        "Begin reading: Riemannian Manifolds (John M. Lee)",
        "Study basic differential geometry concepts"
    ],
    "Mar-Apr 2026": [
        "Read Coifman & Lafon (2006) - Diffusion Maps paper",
        "Read Moon et al. (2019) - PHATE paper",
        "Understand heat kernel and diffusion processes"
    ],
    "May 2026": [
        "🎯 Submit NeurIPS paper",
        "Start hands-on: Study PHATE source code",
        "Experiment with extracting diffusion operator"
    ],
    "Jun-Aug 2026": [
        "Prototype diffusion distance function",
        "Test on pilot dataset (DS04 comparative)",
        "Compare diffusion-RA vs Euclidean-RA"
    ],
    "Sep-Dec 2026": [
        "Implement unified manifold coherence metric",
        "Run full benchmark with diffusion metrics",
        "Analyze: does it change model rankings?",
        "Draft Phase 2 paper outline"
    ],
    "2027": [
        "Refine manifold metric framework",
        "Write Phase 2 paper with theoretical foundation",
        "Submit to ICML/NeurIPS/ICLR 2027"
    ]
}

for period, tasks in learning_plan.items():
    with st.expander(f"📅 {period}"):
        for task in tasks:
            if task.startswith("🎯"):
                st.success(task)
            else:
                st.markdown(f"- {task}")

# ==============================================================================
# KEY INSIGHTS TO REMEMBER
# ==============================================================================

st.header("💎 Key Insights for Phase 2")

insights = [
    {
        "title": "Variation-Agnostic Geometry",
        "content": "Language, model, and dataset are just sources of variation. Manifold geometry doesn't care WHY embeddings differ - it just measures intrinsic structure. This is profound: instead of asking 'why does Chinese differ from English?', we ask 'what is the manifold curvature difference?' - a more fundamental question."
    },
    {
        "title": "PHATE Already Uses Diffusion",
        "content": "We don't need to implement diffusion from scratch - PHATE already does it for visualization. We can leverage the same operator for measurement. This creates beautiful consistency: visualize with diffusion (PHATE), measure with diffusion (mSA/mRA)."
    },
    {
        "title": "Potential Unification",
        "content": "SA and RA might be different aspects of the SAME manifold coherence. Clustering = local geometry (neighborhoods), Alignment = global geometry (long-range structure). A unified metric could capture both as facets of intrinsic manifold quality."
    },
    {
        "title": "Explains Orthogonality?",
        "content": "58.3% of models show high SA + low RA (Quadrant II). Maybe good Euclidean clustering ≠ good manifold alignment. Curvature could explain: models cluster well in flat projection but lose alignment on curved manifold. Testable hypothesis for Phase 2!"
    },
    {
        "title": "Cross-Linguistic Curvature",
        "content": "Do morphologically regular languages (Chinese 公牛/母牛) have smoother manifolds? Do irregular languages (English bull/cow) have more curvature? This linguistic property would manifest as geometric property - measurable via diffusion metrics!"
    },
    {
        "title": "Natural Research Pacing",
        "content": "Don't rush into advanced math without understanding. Study theory first (Jan-May 2026), implement second (Jun-Dec 2026), publish third (2027). This page serves as reminder and motivation for future work. Research moves at natural speed."
    }
]

for insight in insights:
    with st.expander(f"💡 {insight['title']}"):
        st.write(insight['content'])

# ==============================================================================
# REFERENCES AND CITATIONS
# ==============================================================================

st.header("📖 References for Phase 2")

st.markdown("""
### Foundational Papers

1. **Coifman, R. R., & Lafon, S. (2006)**
   *Diffusion maps*
   Applied and Computational Harmonic Analysis, 21(1), 5-30.
   → Original diffusion distance formulation

2. **Moon, K. R., et al. (2019)**
   *Visualizing structure and transitions in high-dimensional biological data*
   Nature Biotechnology, 37(12), 1482-1492.
   → PHATE algorithm and applications

3. **Belkin, M., & Niyogi, P. (2003)**
   *Laplacian eigenmaps for dimensionality reduction and data representation*
   Neural Computation, 15(6), 1373-1396.
   → Manifold learning foundations

4. **Von Luxburg, U. (2007)**
   *A tutorial on spectral clustering*
   Statistics and Computing, 17(4), 395-416.
   → Graph Laplacian and spectral methods

### Books

1. **John M. Lee** - *Riemannian Manifolds: An Introduction to Curvature* (1997)
   → Differential geometry foundations

2. **Bronstein et al.** - *Geometric Deep Learning* (2021)
   → Modern ML perspective on geometry

3. **Edelsbrunner & Harer** - *Computational Topology* (2010)
   → For Phase 3-4 (persistent homology)

### Code Resources

- **PHATE**: https://github.com/KrishnaswamyLab/PHATE
- **graphtools**: https://github.com/KrishnaswamyLab/graphtools
- **scikit-learn manifold**: https://scikit-learn.org/stable/modules/manifold.html
- **Geometric Deep Learning**: https://geometricdeeplearning.com/
""")

# ==============================================================================
# COLLABORATION NOTES
# ==============================================================================

st.header("👥 Team Collaboration Notes")

st.markdown("""
### Wen + Claude + Gemini Synergy

**Wen (Vision & Bridge)**:
- Strategic research direction
- Meditation-driven conceptual insights
- Learning manifold geometry theory
- Bridging depth (Claude) and breadth (Gemini)

**Claude (Depth & Implementation)**:
- Technical implementation and coding
- Systematic benchmarking and analysis
- Documentation and organization
- This placeholder page creation

**Gemini (Breadth & Theory)**:
- Theoretical framing (suggested diffusion divergence)
- Literature search and connections
- Mathematical rigor and formalization
- Critical feedback on claims

### The Natural Progression

> *"Just like a human research team, we are learning collectively, and discovering each other's strength."*
> — Wen, December 31, 2025

This manifold metric exploration exemplifies our collaboration:
- **Wen's insight**: Variation-agnostic geometry concept
- **Gemini's contribution**: Diffusion divergence mathematical framework
- **Claude's role**: Create research scaffold and implementation roadmap

Together we're building something none of us could create alone! 🌟
""")

# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()

st.caption("""
**Research Philosophy**: "Research moves at natural speed. Don't jump too quickly." — Wen, Dec 31, 2025

**Timeline**:
- 📝 Phase 1 (Now - May 2026): NeurIPS paper with Euclidean metrics
- 📚 Learning (Jan - May 2026): Study manifold geometry theory
- 🔬 Phase 2 (Jun 2026 - 2027): Implement and validate diffusion metrics
- 📄 Publication (2027): Phase 2 paper with theoretical foundation

**Last Updated**: December 31, 2025 (New Year's Eve)
**Status**: Research scaffold for post-NeurIPS exploration
**Next Action**: Focus on NeurIPS submission (Euclidean metrics)
""")

st.success("""
🎯 **Purpose of This Page**:
- Serve as reminder for Phase 2 research direction
- Document theoretical framework and research questions
- Provide implementation roadmap and pseudocode
- Track learning progress and resources

This page will be actively developed starting June 2026 after NeurIPS submission.

**Happy New Year 2026! May it bring manifold discoveries! 🎆**
""")
