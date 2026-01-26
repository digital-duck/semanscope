#!/usr/bin/env python3
"""
NSM Groups Color Mapping Preview
Displays the color scheme for Natural Semantic Metalanguage categories
"""

import sys
sys.path.append('../src')
from config import CUSTOM_SEMANTIC_DOMAINS
import pandas as pd

def generate_color_preview():
    """Generate a color preview for NSM Groups"""

    # Read NSM groups
    nsm_df = pd.read_csv('NSM_GROUP.csv')
    nsm_groups = nsm_df['nsm_prime_group'].tolist()

    print("🎨 NSM PRIME GROUPS - SEMANTIC COLOR MAPPING")
    print("=" * 80)
    print()

    # Group by semantic families
    color_families = {
        "Core Concepts (Blue Family)": [
            ("SUBSTANTIVES", "Core entities (I, you, someone, something)"),
            ("RELATIONAL_SUBSTANTIVES", "Relational concepts (kind, part)")
        ],
        "Mental & Communication (Purple Family)": [
            ("MENTAL_PREDICATES", "Mental processes (think, know, want, feel)"),
            ("SPEECH", "Communication (say, words, true)")
        ],
        "Physical Actions (Red/Orange Family)": [
            ("ACTIONS_EVENTS_MOVEMENT", "Actions and movement (do, happen, move, touch)"),
            ("LIFE_DEATH", "Life processes (live, die)")
        ],
        "Space & Time (Green Family)": [
            ("SPACE", "Spatial concepts (where, here, above, below)"),
            ("TIME", "Temporal concepts (when, now, before, after)")
        ],
        "Properties & Evaluation (Yellow/Amber Family)": [
            ("DESCRIPTORS", "Size/property (big, small)"),
            ("EVALUATORS", "Good/bad evaluation"),
            ("INTENSIFIER_AUGMENTOR", "Degree modifiers (very, more)")
        ],
        "Logical & Abstract (Gray Family)": [
            ("LOGICAL_CONCEPTS", "Logic (not, maybe, can, because, if)"),
            ("DETERMINERS", "Determiners (this, the same, other)"),
            ("QUANTIFIERS", "Quantities (one, two, some, all, much, many)"),
            ("SIMILARITY", "Similarity (like, as, way)")
        ],
        "Existence & Meta (Cyan/Pink Family)": [
            ("LOCATION_EXISTENCE", "Existence (be, there is, exist)"),
            ("CONTROL_WORD", "Meta-linguistic control words")
        ]
    }

    for family_name, groups in color_families.items():
        print(f"📂 {family_name}")
        print("-" * 60)

        for group, description in groups:
            if group.lower() in CUSTOM_SEMANTIC_DOMAINS:
                color = CUSTOM_SEMANTIC_DOMAINS[group.lower()]
                print(f"   {group:25} {color:8} | {description}")
            else:
                print(f"   {group:25} {'#------':8} | {description} [NOT MAPPED]")
        print()

    # Summary
    mapped_count = sum(1 for group in nsm_groups if group.lower() in CUSTOM_SEMANTIC_DOMAINS)
    print("=" * 80)
    print(f"TOTAL: {mapped_count}/{len(nsm_groups)} NSM Groups mapped to semantic color families")
    print("=" * 80)
    print()

    print("🎯 COLOR DESIGN PRINCIPLES:")
    print("   • Blue family: Foundational concepts (entities, relationships)")
    print("   • Purple family: Mental and communicative processes")
    print("   • Red/Orange family: Physical actions and life processes")
    print("   • Green family: Spatial and temporal dimensions")
    print("   • Yellow/Amber family: Properties and evaluations")
    print("   • Gray family: Logical and abstract constructs")
    print("   • Cyan/Pink family: Existence and meta-linguistic elements")
    print()

    print("💡 USAGE: These colors will automatically apply in Semanscope visualizations")
    print("   when using NSM Groups as semantic domains for color coding.")

if __name__ == "__main__":
    generate_color_preview()