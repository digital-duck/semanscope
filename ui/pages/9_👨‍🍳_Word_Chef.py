import streamlit as st
import pandas as pd
import random
from itertools import product
import re
from semanscope.config import (
    check_login,
    DEFAULT_TTL
)
from semanscope.components.embedding_viz import EmbeddingVisualizer
from semanscope.components.dimension_reduction import DimensionReducer
from semanscope.components.plotting_echarts import EChartsPlotManager
from semanscope.utils.global_settings import get_global_viz_settings

# Page config
st.set_page_config(
    page_title="Word Chef - Semantic Cooking Lab",
    page_icon="👨‍🍳",
    layout="wide"
)

class WordChef:
    """Semantic cooking laboratory for creating and analyzing word/phrase/sentence combinations"""

    def __init__(self):
        self.physics_groups = {
            'entity': [],
            'identity': [],
            'action': [],
            'relation': [],
            'spatial': [],
            'temporal': [],
            'quantifier': [],
            'qualifier': [],
            'causality': [],
            'misc': []
        }
        self.recipe_pattern = "<entity|identity> + <action|relation> + <entity|identity> + <spatial> + <temporal> + <quantifier|qualifier> + [<causality>]"

        # Semantic granularity layers
        self.parsed_input = {
            'original_sentence': '',
            'words': [],
            'phrases': [],
            'physics_components': {}
        }

    def load_ingredients(self):
        """Load semantic ingredients from various sources"""
        # Default ingredients for each physics group
        default_ingredients = {
            'entity': ['I', 'you', 'apple', 'apples', 'fruit', 'fruits', 'health', 'people', 'person', 'food'],
            'identity': ['this', 'that', 'the same', 'other', 'kind', 'type', 'category'],
            'action': ['eat', 'drink', 'see', 'think', 'know', 'want', 'feel', 'move', 'live', 'say'],
            'relation': ['are', 'is', 'be', 'have', 'like', 'touch', 'belong'],
            'spatial': ['at home', 'here', 'there', 'above', 'below', 'inside', 'outside', 'near', 'far'],
            'temporal': ['daily', 'now', 'today', 'yesterday', 'tomorrow', 'always', 'never', 'sometimes'],
            'quantifier': ['1', '2', '3', 'one', 'two', 'many', 'few', 'some', 'all', 'much'],
            'qualifier': ['good', 'bad', 'big', 'small', 'very', 'nice', 'beautiful', 'healthy', 'fresh'],
            'causality': ['because', 'if', 'maybe', 'can', 'not'],
            'misc': ['and', 'or', 'but', 'also', 'then']
        }

        # Initialize with default ingredients
        for group, ingredients in default_ingredients.items():
            self.physics_groups[group] = ingredients.copy()

    def parse_input_sentence(self, sentence):
        """Parse input sentence into words, phrases, and physics components"""
        self.parsed_input['original_sentence'] = sentence.strip()

        # Extract words
        words = sentence.strip().split()
        self.parsed_input['words'] = words

        # Extract phrases (simple heuristics for now)
        phrases = self.extract_phrases(sentence)
        self.parsed_input['phrases'] = phrases

        # Classify into physics groups (simplified classification)
        physics_components = self.classify_physics_components(sentence)
        self.parsed_input['physics_components'] = physics_components

        return self.parsed_input

    def extract_phrases(self, sentence):
        """Extract meaningful phrases from sentence"""
        # Simple phrase extraction (can be enhanced with NLP)
        phrases = []

        # Common phrase patterns
        phrase_patterns = [
            r'\b\d+\s+\w+',  # "2 apples", "3 times"
            r'\bat\s+\w+',   # "at home", "at work"
            r'\bbecause\s+[\w\s]+',  # "because fruits are good"
            r'\bare\s+\w+',  # "are good", "are healthy"
            r'\bgood\s+for\s+\w+',  # "good for health"
        ]

        import re
        for pattern in phrase_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            phrases.extend(matches)

        # Add individual content words as single-word phrases
        content_words = [w for w in sentence.split() if len(w) > 2 and w.lower() not in ['the', 'and', 'but', 'for']]
        phrases.extend(content_words)

        return list(set(phrases))  # Remove duplicates

    def classify_physics_components(self, sentence):
        """Classify words/phrases into physics groups (simplified)"""
        words = sentence.lower().split()
        components = {}

        # Simple classification rules (can be enhanced)
        entity_words = ['i', 'you', 'apples', 'fruits', 'health', 'people', 'person']
        action_words = ['eat', 'drink', 'think', 'see', 'move', 'live']
        relation_words = ['are', 'is', 'be', 'have']
        spatial_words = ['home', 'here', 'there', 'above', 'below', 'at']
        temporal_words = ['daily', 'now', 'today', 'always', 'never']
        quantifier_words = ['1', '2', '3', 'one', 'two', 'many', 'few', 'some', 'all']
        qualifier_words = ['good', 'bad', 'big', 'small', 'healthy', 'fresh']
        causality_words = ['because', 'if', 'since', 'when']

        for word in words:
            if word in entity_words:
                components.setdefault('entity', []).append(word)
            elif word in action_words:
                components.setdefault('action', []).append(word)
            elif word in relation_words:
                components.setdefault('relation', []).append(word)
            elif word in spatial_words or 'at' in sentence and word == 'home':
                components.setdefault('spatial', []).append(word)
            elif word in temporal_words:
                components.setdefault('temporal', []).append(word)
            elif word in quantifier_words or word.isdigit():
                components.setdefault('quantifier', []).append(word)
            elif word in qualifier_words:
                components.setdefault('qualifier', []).append(word)
            elif word in causality_words:
                components.setdefault('causality', []).append(word)
            else:
                components.setdefault('misc', []).append(word)

        return components

    def add_ingredient(self, group, ingredient):
        """Add a new ingredient to a physics group"""
        if group in self.physics_groups and ingredient not in self.physics_groups[group]:
            self.physics_groups[group].append(ingredient)

    def remove_ingredient(self, group, ingredient):
        """Remove an ingredient from a physics group"""
        if group in self.physics_groups and ingredient in self.physics_groups[group]:
            self.physics_groups[group].remove(ingredient)

    def generate_sentence(self, pattern_config):
        """Generate a sentence based on pattern configuration"""
        try:
            components = []

            # Entity/Identity 1
            if pattern_config.get('entity_identity_1'):
                if random.choice([True, False]):  # Randomly choose entity or identity
                    components.append(random.choice(self.physics_groups['entity']))
                else:
                    components.append(random.choice(self.physics_groups['identity']))

            # Action/Relation
            if pattern_config.get('action_relation'):
                if random.choice([True, False]):  # Randomly choose action or relation
                    components.append(random.choice(self.physics_groups['action']))
                else:
                    components.append(random.choice(self.physics_groups['relation']))

            # Entity/Identity 2
            if pattern_config.get('entity_identity_2'):
                if random.choice([True, False]):
                    components.append(random.choice(self.physics_groups['entity']))
                else:
                    components.append(random.choice(self.physics_groups['identity']))

            # Spatial
            if pattern_config.get('spatial'):
                components.append(random.choice(self.physics_groups['spatial']))

            # Temporal
            if pattern_config.get('temporal'):
                components.append(random.choice(self.physics_groups['temporal']))

            # Quantifier/Qualifier
            if pattern_config.get('quantifier_qualifier'):
                if random.choice([True, False]):
                    components.append(random.choice(self.physics_groups['quantifier']))
                else:
                    components.append(random.choice(self.physics_groups['qualifier']))

            # Causality (optional)
            if pattern_config.get('causality') and random.choice([True, False]):
                components.append(random.choice(self.physics_groups['causality']))
                # Add another clause after causality
                components.append(random.choice(self.physics_groups['entity']))
                components.append(random.choice(self.physics_groups['relation']))
                components.append(random.choice(self.physics_groups['qualifier']))

            return ' '.join(components)

        except Exception as e:
            return f"Error generating sentence: {str(e)}"

    def generate_multi_granularity_batch(self, num_items, granularity_level="mixed"):
        """Generate batch with word/phrase/sentence granularities based on parsed input"""
        if not self.parsed_input['original_sentence']:
            return pd.DataFrame()

        items = []

        # Extract source materials
        words = self.parsed_input['words']
        phrases = self.parsed_input['phrases']
        physics_components = self.parsed_input['physics_components']

        # Add each decomposed word as a separate item
        for word in words:
            items.append({
                'word': word,
                'domain': 'word',
                'validity': 'valid',
                'semantic_coherence': 'high'
            })

        # Add each decomposed phrase as a separate item
        for phrase in phrases:
            items.append({
                'word': phrase,
                'domain': 'phrase',
                'validity': 'valid',
                'semantic_coherence': 'high'
            })

        for i in range(num_items):
            if granularity_level == "word" or (granularity_level == "mixed" and i % 3 == 0):
                # Word-level recombination
                if len(words) >= 3:
                    shuffled_words = random.sample(words, min(len(words), random.randint(3, 8)))
                    random.shuffle(shuffled_words)
                    item_text = ' '.join(shuffled_words)
                else:
                    continue

            elif granularity_level == "phrase" or (granularity_level == "mixed" and i % 3 == 1):
                # Phrase-level recombination
                if len(phrases) >= 2:
                    selected_phrases = random.sample(phrases, min(len(phrases), random.randint(2, 5)))
                    random.shuffle(selected_phrases)
                    item_text = ' '.join(selected_phrases)
                else:
                    # Fallback to word level
                    shuffled_words = random.sample(words, min(len(words), random.randint(3, 6)))
                    item_text = ' '.join(shuffled_words)

            else:
                # Sentence-level recombination using physics components
                sentence_parts = []

                for group, components in physics_components.items():
                    if components and random.choice([True, False]):  # Random inclusion
                        sentence_parts.append(random.choice(components))

                if len(sentence_parts) >= 3:
                    random.shuffle(sentence_parts)
                    item_text = ' '.join(sentence_parts)
                else:
                    # Fallback to phrase level
                    selected_phrases = random.sample(phrases, min(len(phrases), 3))
                    item_text = ' '.join(selected_phrases)

            # Assess validity and domain based on actual content
            validity = self.assess_validity(item_text)
            semantic_coherence = self.assess_semantic_coherence(item_text)

            # Set domain to sentence for all generated items
            domain = "sentence"

            items.append({
                'word': item_text,
                'domain': domain,
                'validity': validity,
                'semantic_coherence': semantic_coherence
            })


        df = pd.DataFrame(items)
        # Deduplicate based on 'word' column (keeping first occurrence)
        df = df.drop_duplicates(subset=['word'], keep='first')
        return df

    def assess_semantic_coherence(self, text):
        """Assess semantic coherence beyond basic validity"""
        words = text.split()

        # Coherence heuristics
        score = 0

        # Length coherence
        if 3 <= len(words) <= 15:
            score += 1

        # Physics group diversity
        components = self.classify_physics_components(text)
        unique_groups = len([group for group, items in components.items() if items])
        if unique_groups >= 3:
            score += 1

        # Causality bonus
        if any(word in text.lower() for word in ['because', 'if', 'since']):
            score += 1

        # Repetition penalty
        if len(set(words)) < len(words) * 0.7:  # Too much repetition
            score -= 1

        # Return coherence level
        if score >= 3:
            return "high"
        elif score >= 1:
            return "medium"
        else:
            return "low"

    def generate_batch(self, num_sentences, pattern_config, include_invalid=True):
        """Generate a batch of sentences for analysis (legacy method)"""
        sentences = []

        for i in range(num_sentences):
            sentence = self.generate_sentence(pattern_config)
            validity = self.assess_validity(sentence)

            sentences.append({
                'sentence_id': i + 1,
                'sentence': sentence,
                'validity': validity,
                'pattern': self.recipe_pattern,
                'length': len(sentence.split())
            })

        return pd.DataFrame(sentences)

    def assess_validity(self, sentence):
        """Rough assessment of sentence validity (placeholder for more sophisticated analysis)"""
        # Simple heuristics for now
        words = sentence.split()

        # Very short or very long sentences are likely problematic
        if len(words) < 3 or len(words) > 20:
            return 'questionable'

        # Contains causality patterns
        if any(causal in sentence.lower() for causal in ['because', 'if', 'since']):
            return 'valid'

        # Random assessment for demonstration
        return random.choice(['valid', 'questionable', 'invalid'])

def main():
    check_login()

    st.subheader("👨‍🍳 Word Chef - Semantic Cooking Laboratory")
    st.markdown("**Cook up semantic combinations and explore the physics of meaning!**")

    # Initialize Word Chef
    if 'word_chef' not in st.session_state:
        st.session_state.word_chef = WordChef()
        st.session_state.word_chef.load_ingredients()

    chef = st.session_state.word_chef

    # Sidebar: Ingredient Management
    with st.sidebar:
        st.subheader("🥘 Ingredient Pantry")
        st.markdown("**Manage your semantic ingredients by physics group**")

        # Recipe Pattern Display
        with st.expander("📜 Recipe Pattern", expanded=False):
            st.code(chef.recipe_pattern, language="text")
            st.markdown("""
            **Physics Groups:**
            - **Entity**: Subjects/objects (I, apple, health)
            - **Identity**: Classifiers (this, kind, type)
            - **Action**: State transitions (eat, think, move)
            - **Relation**: Connections (are, is, like)
            - **Spatial**: Location (at home, here, above)
            - **Temporal**: Time (daily, now, always)
            - **Quantifier**: Numbers (2, many, all)
            - **Qualifier**: Properties (good, big, fresh)
            - **Causality**: Logic (because, if, maybe)
            """)

        # Ingredient Editor
        st.markdown("### 🔧 Edit Ingredients")
        selected_group = st.selectbox(
            "Select Physics Group",
            options=list(chef.physics_groups.keys()),
            key="ingredient_group_selector"
        )

        # Display current ingredients
        current_ingredients = chef.physics_groups[selected_group]
        st.markdown(f"**Current {selected_group} ingredients:**")
        st.write(", ".join(current_ingredients) if current_ingredients else "No ingredients")

        # Add new ingredient
        new_ingredient = st.text_input(
            f"Add new {selected_group}",
            placeholder="Enter new ingredient...",
            key="new_ingredient_input"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add", key="add_ingredient"):
                if new_ingredient.strip():
                    chef.add_ingredient(selected_group, new_ingredient.strip())
                    st.success(f"Added '{new_ingredient}' to {selected_group}")
                    st.rerun()

        with col2:
            # Remove ingredient
            if current_ingredients:
                ingredient_to_remove = st.selectbox(
                    "Remove",
                    options=current_ingredients,
                    key="remove_ingredient_selector"
                )
                if st.button("🗑️ Remove", key="remove_ingredient"):
                    chef.remove_ingredient(selected_group, ingredient_to_remove)
                    st.success(f"Removed '{ingredient_to_remove}' from {selected_group}")
                    st.rerun()

        st.markdown("---")

        # Batch Generation Settings
        st.subheader("⚙️ Cooking Settings")

        num_sentences = st.slider(
            "Number of sentences to cook",
            min_value=5,
            max_value=100,
            value=20,
            help="How many sentence variations to generate"
        )

        # Pattern Configuration
        st.markdown("**Pattern Elements:**")
        pattern_config = {
            'entity_identity_1': st.checkbox("Entity/Identity 1", value=True),
            'action_relation': st.checkbox("Action/Relation", value=True),
            'entity_identity_2': st.checkbox("Entity/Identity 2", value=True),
            'spatial': st.checkbox("Spatial", value=True),
            'temporal': st.checkbox("Temporal", value=True),
            'quantifier_qualifier': st.checkbox("Quantifier/Qualifier", value=True),
            'causality': st.checkbox("Causality (optional)", value=False)
        }

    # Main Content: Cooking Interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🍳 Sentence Kitchen")

        # Sentence Input & Decomposition
        with st.expander("🧬 Sentence Decomposition Lab", expanded=True):
            st.markdown("**Enter a sentence to decompose and recombine:**")

            input_sentence = st.text_area(
                "Input Sentence",
                value="I eat 2 apples daily at home because fruits are good for health",
                help="Enter a sentence to break down into words, phrases, and physics components",
                height=100,
                key="input_sentence"
            )

            col_parse1, col_parse2 = st.columns(2)

            with col_parse1:
                if st.button("🧬 Parse Sentence", type="primary"):
                    if input_sentence.strip():
                        # Parse the sentence
                        parsed_data = chef.parse_input_sentence(input_sentence.strip())
                        st.session_state.parsed_sentence = parsed_data
                        st.success("✅ Sentence decomposed successfully!")

            with col_parse2:
                if st.button("🔬 Analyze Original", type="secondary"):
                    if input_sentence.strip():
                        # Store original sentence for visualization
                        st.session_state.manual_sentence_analysis = {
                            'sentence': input_sentence.strip(),
                            'words': input_sentence.strip().split(),
                            'validity': chef.assess_validity(input_sentence.strip())
                        }
                        st.success("✅ Original sentence prepared for analysis!")

        # Display Parsed Components
        if 'parsed_sentence' in st.session_state:
            parsed = st.session_state.parsed_sentence

            with st.expander("📊 Decomposition Results", expanded=True):
                st.markdown("**Original Sentence:**")
                st.markdown(f"*{parsed['original_sentence']}*")

                col_comp1, col_comp2, col_comp3 = st.columns([2,4,4])

                with col_comp1:
                    st.markdown("**🔤 Words:**")
                    words_text = '\n'.join(parsed['words'])
                    st.text_area(
                        "Words List",
                        value=words_text,
                        height=300,
                        key="words_display",
                        label_visibility="collapsed"
                    )

                with col_comp2:
                    st.markdown("**🔗 Phrases:**")
                    phrases_text = '\n'.join(parsed['phrases']) if parsed['phrases'] else "None detected"
                    st.text_area(
                        "Phrases List",
                        value=phrases_text,
                        height=300,
                        key="phrases_display",
                        label_visibility="collapsed"
                    )

                with col_comp3:
                    st.markdown("**⚛️ Physics Groups:**")
                    physics_lines = []
                    for group, items in parsed['physics_components'].items():
                        if items:
                            physics_lines.append(f"{group}: {', '.join(items)}")
                    physics_text = '\n'.join(physics_lines) if physics_lines else "No groups detected"
                    st.text_area(
                        "Physics Groups",
                        value=physics_text,
                        height=300,
                        key="physics_display",
                        label_visibility="collapsed"
                    )

        # Manual Creation (simplified)
        with st.expander("✍️ Manual Creation", expanded=False):
            manual_text = st.text_input(
                "Quick manual entry",
                placeholder="Type a custom sentence or phrase...",
                help="Create your own text for analysis"
            )

            if st.button("🔬 Analyze Manual Entry") and manual_text.strip():
                st.session_state.manual_sentence_analysis = {
                    'sentence': manual_text.strip(),
                    'words': manual_text.strip().split(),
                    'validity': chef.assess_validity(manual_text.strip())
                }
                st.success("✅ Manual entry prepared for analysis!")

        # Quick Single Generation
        st.markdown("### 🎲 Quick Generate")
        col_gen1, col_gen2 = st.columns(2)

        with col_gen1:
            if st.button("🎯 Generate Valid Recipe", width='stretch'):
                sentence = chef.generate_sentence(pattern_config)
                st.markdown(f"**Generated:** {sentence}")

        with col_gen2:
            if st.button("🎲 Random Recipe", width='stretch'):
                # Random pattern configuration
                random_config = {key: random.choice([True, False]) for key in pattern_config.keys()}
                sentence = chef.generate_sentence(random_config)
                st.markdown(f"**Random:** {sentence}")

    with col2:
        st.subheader("📊 Multi-Granularity Batch Cooking")

        # Fixed to sentence level only
        granularity_mode = "sentence"

        # Batch Generation
        if st.button("🚀 Cook Multi-Granularity Batch", type="primary", width='stretch'):
            if 'parsed_sentence' in st.session_state:
                with st.spinner("👨‍🍳 Cooking semantic combinations across granularities..."):
                    # Set the parsed input from session state to the chef object
                    chef.parsed_input = st.session_state.parsed_sentence
                    batch_df = chef.generate_multi_granularity_batch(num_sentences, granularity_mode)
                    st.session_state.cooked_batch = batch_df
                    st.success(f"🍽️ Cooked {len(batch_df)} items across granularity levels!")
            else:
                st.warning("⚠️ Please parse a sentence first using the Decomposition Lab!")

        # Legacy Batch Generation (fallback)
        if st.button("🎲 Cook Legacy Batch", type="secondary", width='stretch'):
            with st.spinner("👨‍🍳 Cooking traditional sentence combinations..."):
                batch_df = chef.generate_batch(num_sentences, pattern_config)
                st.session_state.cooked_batch = batch_df
                st.success(f"🍽️ Cooked {len(batch_df)} traditional sentences!")

        # Display Cooked Batch
        if 'cooked_batch' in st.session_state:
            st.markdown("### 🍽️ Fresh Batch")
            df = st.session_state.cooked_batch

            # Summary stats
            if 'domain' in df.columns:
                # Multi-granularity batch
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                validity_counts = df['validity'].value_counts()
                granularity_counts = df['domain'].value_counts()
                coherence_counts = df['semantic_coherence'].value_counts()

                with col_stat1:
                    st.metric("Total Items", len(df))
                with col_stat2:
                    st.metric("High Coherence", coherence_counts.get('high', 0))
                with col_stat3:
                    st.metric("Valid", validity_counts.get('valid', 0))
                with col_stat4:
                    st.metric("Word Level", granularity_counts.get('word', 0))

                # Granularity breakdown
                st.markdown("**📊 Granularity Distribution:**")
                for granularity in ['word', 'phrase', 'sentence']:
                    count = granularity_counts.get(granularity, 0)
                    if count > 0:
                        st.write(f"• **{granularity.capitalize()}**: {count} items")

            else:
                # Legacy batch
                validity_counts = df['validity'].value_counts()
                col_stat1, col_stat2, col_stat3 = st.columns(3)

                with col_stat1:
                    st.metric("Valid", validity_counts.get('valid', 0))
                with col_stat2:
                    st.metric("Questionable", validity_counts.get('questionable', 0))
                with col_stat3:
                    st.metric("Invalid", validity_counts.get('invalid', 0))

            # Data table
            st.dataframe(
                df,
                width='stretch',
                height=300
            )

            # Download option
            csv = df.to_csv(index=False)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"word_chef_batch_{timestamp}.csv"

            st.download_button(
                label="📥 Download Batch CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )

    # Visualization Section
    st.markdown("---")
    st.subheader("🗺️ Semantic Space Exploration")

    # Get global settings
    viz_settings = get_global_viz_settings()

    # Visualization controls
    col_viz1, col_viz2, col_viz3 = st.columns(3)

    with col_viz1:
        viz_mode = st.radio(
            "Visualization Mode",
            ["Manual Recipe", "Batch Analysis", "Word-level Analysis"],
            help="Choose what to visualize in semantic space"
        )

    with col_viz2:
        dimensions = st.radio(
            "Dimensions",
            ["2D", "3D"],
            horizontal=True
        )

    with col_viz3:
        if st.button("🧭 Launch Semanscope", type="secondary"):
            if viz_mode == "Manual Recipe" and 'manual_sentence_analysis' in st.session_state:
                # Visualize manual sentence
                manual_data = st.session_state.manual_sentence_analysis
                words = manual_data['words']

                # Initialize visualization components
                visualizer = EmbeddingVisualizer()
                reducer = DimensionReducer()
                echarts_manager = EChartsPlotManager()

                with st.spinner("🔄 Analyzing semantic patterns..."):
                    try:
                        # Generate embeddings
                        embeddings = visualizer.get_embeddings(words, viz_settings['model_name'])

                        if embeddings is not None:
                            # Reduce dimensions
                            dims = 3 if dimensions == "3D" else 2
                            reduced_embeddings = reducer.reduce_dimensions_with_cache(
                                embeddings, viz_settings['method_name'], dims,
                                dataset="word_chef_manual", lang="enu", model=viz_settings['model_name']
                            )

                            # Create colors (simple: valid=green, others=red)
                            validity = manual_data['validity']
                            color = '#16A34A' if validity == 'valid' else '#DC2626'
                            colors = [color] * len(words)

                            # Plot
                            title = f"Word Chef Analysis: {manual_data['sentence'][:50]}..."
                            if dimensions == "3D":
                                echarts_manager.plot_3d(
                                    reduced_embeddings, words, colors, title,
                                    method_name=viz_settings['method_name'],
                                    model_name=viz_settings['model_name'],
                                    dataset_name="Manual Recipe",
                                    display_chart=True
                                )
                            else:
                                echarts_manager.plot_2d(
                                    reduced_embeddings, words, colors, title,
                                    method_name=viz_settings['method_name'],
                                    model_name=viz_settings['model_name'],
                                    dataset_name="Manual Recipe",
                                    display_chart=True
                                )

                            st.success("🎯 Semantic analysis complete!")

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")

            elif viz_mode == "Batch Analysis" and 'cooked_batch' in st.session_state:
                st.info("🚧 Batch analysis coming soon! For now, use manual recipe mode.")

            elif viz_mode == "Word-level Analysis":
                st.info("🚧 Word-level analysis coming soon!")

            else:
                st.warning("⚠️ Please generate some recipes first!")

    # Footer
    st.markdown("---")
    with st.expander("ℹ️ About Word Chef", expanded=False):
        st.markdown("""
        **Word Chef** is your semantic cooking laboratory for exploring the physics of meaning!

        🎯 **Purpose**: Systematically generate and analyze word/sentence combinations to understand semantic structure

        🧪 **Experiments**:
        - Test semantic validity boundaries
        - Explore meaningful vs. meaningless combinations
        - Discover geometric patterns in language

        ⚛️ **Physics Inspiration**: Just like atomic combinations, not all word combinations are stable.
        Explore the semantic forces that govern meaningful language!

        🔬 **Analysis**: Use Semanscope to visualize how your semantic creations organize in embedding space.
        """)

if __name__ == "__main__":
    main()