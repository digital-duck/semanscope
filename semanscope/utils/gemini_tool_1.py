from typing import List, Dict, Tuple, Optional
import json

# --- START: DATA FROM radical_extractor.py (Simplified for demonstration) ---
# The master list of feature radicals used for creating the one-hot vector.
FEATURE_RADICALS = ['人', '口', '土', '木', '火', '艹', '氵', '金', '手', '日']
VECTOR_DIMENSION = len(FEATURE_RADICALS)

# Mapping of character to its *simplified* structural components (radicals)
# NOTE: This data is currently mock data. The user's custom 6,000-character 
# hierarchical decomposition data should replace this.
CHARACTER_DECOMPOSITION: Dict[str, List[str]] = {
    '江': ['氵', '工'],      # River (Should match: ['氵', '工'])
    '打': ['手', '丁'],      # To hit (Should match: ['扌', '丁'] if '扌' is used instead of '手')
    '烧': ['火', '尧'],      # To burn (Should match: ['火', '尧'])
    '树': ['木', '对'],      # Tree
    '做': ['人', '故'],      # To do
    '曰': ['日'],           # To say
    '草': ['艹', '早'],      # Grass
    '埋': ['土', '里'],      # To bury
    '铜': ['金', '同'],      # Copper
    '叭': ['口', '八'],      # Horn
    '森': ['木', '木', '木'], # Forest (A multi-component character for checking)
}

# Stroke count lookup (subset of your STROKE_COUNTS)
STROKE_COUNTS_SUBSET: Dict[str, int] = {
    '江': 6, '打': 5, '烧': 10, '树': 9, '做': 11, '曰': 4, '草': 9, '埋': 10, '铜': 13, '叭': 5, '森': 12
}
# --- END: DATA FROM radical_extractor.py ---


def create_feature_vector(character: str) -> Tuple[List[int], int]:
    """
    Creates a combined feature vector (Radical Vector + Stroke Count) for a character.
    (This function remains the core feature extraction logic.)
    """
    # ... (function body remains the same as previous version) ...

    if character not in CHARACTER_DECOMPOSITION:
        print(f"Error: Decomposition data missing for '{character}'.")
        return [0] * VECTOR_DIMENSION, 0
    
    if character not in STROKE_COUNTS_SUBSET:
        print(f"Error: Stroke count missing for '{character}'.")
        # Fallback to an estimate or 0, depending on application needs
        stroke_count = 0
    else:
        stroke_count = STROKE_COUNTS_SUBSET[character]
        
    # 1. Initialize the radical vector (all zeros)
    feature_vector = [0] * VECTOR_DIMENSION
    
    # 2. Get the character's components
    components = CHARACTER_DECOMPOSITION[character]
    
    # 3. Populate the vector based on component presence
    for component in components:
        try:
            # Find the index of the component in our master radical list
            index = FEATURE_RADICALS.index(component)
            # Set the corresponding dimension to 1 (One-Hot Encoding)
            feature_vector[index] = 1
        except ValueError:
            # Handle cases where a component is not in our defined feature set
            # For robustness, you might want to log this.
            pass
            
    return feature_vector, stroke_count


def validate_decomposition(character: str, custom_components: List[str], external_decomposition: Dict[str, List[str]]) -> Optional[str]:
    """
    Compares the user's custom (first-layer) components against components 
    retrieved from an external, standard structural dictionary (e.g., cjklib).

    Args:
        character: The character to validate.
        custom_components: The list of components from the user's data.
        external_decomposition: A dictionary mapping character to its externally sourced components.

    Returns:
        None if the components match (allowing for order differences), 
        or an error string describing the mismatch.
    """
    # --- NOTE ON cjklib: ---
    # The 'cjklib' library (or similar CJK data packages) would be used here.
    # To use it, you would need to: 
    # 1. pip install cjklib
    # 2. Add: from cjklib import character
    # 3. Replace the 'external_components' lookup with the actual library call:
    #    external_components = character.getComponents(character, 'All')
    # -----------------------

    external_components = external_decomposition.get(character, [])

    # The order often differs between sources, so we compare sets for a match
    set_custom = set(custom_components)
    set_external = set(external_components)

    if not external_components:
        return f"Warning: External data source has no decomposition for '{character}'."
        
    if set_custom == set_external:
        return None # Match!
    else:
        # Check for partial matches or different conventions (e.g., '手' vs '扌')
        missing_in_custom = set_external - set_custom
        extra_in_custom = set_custom - set_external
        
        # This is a strict mismatch
        return f"Mismatch for '{character}'. Custom: {custom_components} (Extra: {extra_in_custom}, Missing: {missing_in_custom}). External: {external_components}"


# --- Example Usage ---
if __name__ == '__main__':
    target_characters = ['江', '打', '烧', '树', '铜', '森']
    
    # --- MOCK EXTERNAL DATA FOR VALIDATION (Simulating cjklib output) ---
    # We intentionally include a known mismatch for '打' to show the validation working.
    MOCK_EXTERNAL_DATA: Dict[str, List[str]] = {
        '江': ['氵', '工'],      # Match
        '打': ['扌', '丁'],      # Mismatch: User data has ['手', '丁'], external has ['扌', '丁']
        '烧': ['火', '尧'],      # Match
        '树': ['木', '对'],      # Match
        '铜': ['钅', '同'],      # Mismatch: User data has ['金', '同'], external has ['钅', '同'] (different radical forms)
        '森': ['木', '木', '木'], # Match
        '曰': ['日']             # Match
    }
    # -------------------------------------------------------------------

    print(f"Master Radical Feature Dimension: {VECTOR_DIMENSION} elements.\n")
    print("--- Feature Vector Generation ---")
    
    validation_issues: List[str] = []
    
    for char in target_characters:
        vector, strokes = create_feature_vector(char)
        combined_features = vector + [strokes]
        
        print(f"Character: {char}")
        print(f"  Custom Components: {CHARACTER_DECOMPOSITION.get(char, 'N/A')}")
        print(f"  Total Feature Vector Length: {len(combined_features)}\n")
        
        # Perform Validation
        custom_comps = CHARACTER_DECOMPOSITION.get(char, [])
        issue = validate_decomposition(char, custom_comps, MOCK_EXTERNAL_DATA)
        if issue:
            validation_issues.append(issue)

    print("\n--- Level-1 Validation Report (vs. MOCK EXTERNAL DATA) ---")
    if validation_issues:
        print("ISSUES FOUND:")
        for issue in validation_issues:
            print(f"- {issue}")
    else:
        print("All target characters match the external data at the first level of decomposition.")
