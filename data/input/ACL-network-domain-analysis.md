# ACL Semantic Network Domain Analysis

**Date**: 2025-10-19
**Purpose**: Domain analysis for semantic visualization and color-coding
**Datasets**: 3 multilingual semantic networks

---

## Overview Statistics

| Language | Root Words | Total Words | Fine-Grained Domains | Coarse-Grained Domains |
|----------|-----------|-------------|---------------------|----------------------|
| **Chinese** | 子 (zi) | 124 | 7 | 7 |
| **German** | Haus/Arbeit | 91 | 65 | 8 |
| **English** | work/light | 63 | 43 | 8 |

---

## Coarse-Grained Domain Mapping for Color-Coding

### Recommended Color Palette (8 domains)

| Coarse Domain | Color Code | RGB | Description |
|---------------|------------|-----|-------------|
| **People & Social** | `#E74C3C` | (231, 76, 60) | Red - Relationships, roles, social structures |
| **Physical Objects** | `#3498DB` | (52, 152, 219) | Blue - Tangible items, tools, furniture |
| **Nature & Biology** | `#27AE60` | (39, 174, 96) | Green - Natural phenomena, animals, plants |
| **Abstract Concepts** | `#9B59B6` | (155, 89, 182) | Purple - Ideas, qualities, philosophical |
| **Science & Technology** | `#F39C12` | (243, 156, 18) | Orange - Physics, computing, modern tech |
| **Built Environment** | `#95A5A6` | (149, 165, 166) | Gray - Architecture, buildings, infrastructure |
| **Cognition & Learning** | `#1ABC9C` | (26, 188, 156) | Teal - Education, mental processes, enlightenment |
| **Time & Process** | `#E67E22` | (230, 126, 34) | Dark Orange - Temporal, workflow, procedures |

---

## Dataset 1: Chinese 子 (zi) Network

**File**: `ACL-network-子-chn.txt`
**Total Words**: 124
**Fine-Grained Domains**: 7

### Domain Distribution

| Fine-Grained Domain | Word Count | Coarse Domain | Color |
|---------------------|-----------|---------------|-------|
| historical_figures | 9 | People & Social | Red |
| social_relationships | 28 | People & Social | Red |
| physics_domain | 18 | Science & Technology | Orange |
| body_parts_slang | 8 | Nature & Biology | Green |
| everyday_objects | 35 | Physical Objects | Blue |
| abstract_concepts | 20 | Abstract Concepts | Purple |
| food_plants | 6 | Nature & Biology | Green |

### Coarse-Grained Mapping

```json
{
  "people_social": {
    "color": "#E74C3C",
    "fine_domains": ["historical_figures", "social_relationships"],
    "count": 37
  },
  "physical_objects": {
    "color": "#3498DB",
    "fine_domains": ["everyday_objects"],
    "count": 35
  },
  "nature_biology": {
    "color": "#27AE60",
    "fine_domains": ["body_parts_slang", "food_plants"],
    "count": 14
  },
  "abstract_concepts": {
    "color": "#9B59B6",
    "fine_domains": ["abstract_concepts"],
    "count": 20
  },
  "science_technology": {
    "color": "#F39C12",
    "fine_domains": ["physics_domain"],
    "count": 18
  }
}
```

### Sample Words by Coarse Domain

**People & Social** (37 words)
- Historical: 孔子 (Confucius), 老子 (Laozi), 孟子 (Mencius)
- Relationships: 弟子 (disciple), 君子 (gentleman), 孩子 (child)

**Physical Objects** (35 words)
- Furniture: 桌子 (table), 椅子 (chair), 凳子 (stool)
- Containers: 箱子 (box), 盒子 (case), 瓶子 (bottle)
- Tools: 刷子 (brush), 刀子 (knife), 梯子 (ladder)

**Nature & Biology** (14 words)
- Animals: 狮子 (lion), 兔子 (rabbit), 猴子 (monkey)
- Body: 鼻子 (nose), 脑子 (brain), 肚子 (belly)
- Food: 瓜子 (melon seeds), 橘子 (orange), 茄子 (eggplant)

**Abstract Concepts** (20 words)
- Temporal: 日子 (days), 子时 (midnight hour), 甲子 (60-year cycle)
- Social: 面子 (face/dignity), 圈子 (circle/social group)
- Mathematical: 子集 (subset), 子群 (subgroup), 子空间 (subspace)

**Science & Technology** (18 words)
- Fundamental: 原子 (atom), 电子 (electron), 质子 (proton)
- Quantum: 量子 (quantum), 光子 (photon), 玻色子 (boson)
- Biological: 种子 (seed), 精子 (sperm), 卵子 (egg)

---

## Dataset 2: German Haus/Arbeit Network

**File**: `ACL-network-haus-arbeit-deu.txt`
**Total Words**: 91
**Fine-Grained Domains**: 65

### Coarse-Grained Domain Mapping

| Coarse Domain | Fine-Grained Domains (count) | Total Words |
|---------------|----------------------------|-------------|
| **People & Social** | 11 domains | 23 |
| **Physical Objects** | 8 domains | 9 |
| **Built Environment** | 12 domains | 25 |
| **Abstract Concepts** | 5 domains | 5 |
| **Cognition & Learning** | 4 domains | 4 |
| **Time & Process** | 9 domains | 14 |
| **Science & Technology** | 3 domains | 3 |
| **Nature & Biology** | 2 domains | 8 |

### Detailed Coarse Domain Breakdown

#### 1. People & Social (23 words)
**Fine domains**: social_roles, occupational_roles, administrative_roles, ownership_roles, economic_roles, collaborative_roles, social_organizations, employment_status, employment_positions, behavioral_qualities, social_activities

**Color**: `#E74C3C` (Red)

Examples:
- Roles: Hausfrau, Hausmann, Arbeiter, Mitarbeiter
- Organizations: Hausgemeinschaft, Arbeitsgruppe
- Employment: arbeitslos, Arbeitsstelle

#### 2. Physical Objects (9 words)
**Fine domains**: clothing_items, material_possessions, occupational_clothing, occupational_equipment, educational_materials, documentary_records, legal_documents, administrative_documents

**Color**: `#3498DB` (Blue)

Examples:
- Clothing: Hausschuh, Arbeitsanzug, Arbeitskleidung
- Objects: Hausrat, Arbeitsbuch, Arbeitsheft

#### 3. Built Environment (25 words)
**Fine domains**: architectural_components, architectural_types, residential_architecture, urban_architecture, civic_buildings, commercial_establishments, educational_buildings, industrial_buildings, recreational_buildings, recreational_structures, agricultural_structures, construction_activities

**Color**: `#95A5A6` (Gray)

Examples:
- Components: Hausdach, Hausfassade, Haustür, Hauswand
- Buildings: Hochhaus, Rathaus, Kaufhaus, Wohnhaus
- Types: Blockhaus, Fachwerkhaus, Glashaus

#### 4. Abstract Concepts (5 words)
**Fine domains**: core_morphology, grammatical_forms, size_modification, abstract_qualities, quality_assessment

**Color**: `#9B59B6` (Purple)

Examples:
- Core: Haus, Arbeit
- Forms: Häuser, Häuschen
- Qualities: arbeitsam

#### 5. Cognition & Learning (4 words)
**Fine domains**: educational_tasks, educational_materials, educational_buildings, educational_spaces

**Color**: `#1ABC9C` (Teal)

Examples:
- Tasks: Hausaufgabe
- Materials: Arbeitsheft
- Spaces: Schulhaus

#### 6. Time & Process (14 words)
**Fine domains**: temporal_aspects, temporal_organization, temporal_work_patterns, process_management, process_activities, action_verbs, prefixed_verb, employment_arrangements, work_arrangements

**Color**: `#E67E22` (Dark Orange)

Examples:
- Temporal: arbeitsam, gearbeitet, Arbeitszeit
- Work patterns: Nachtarbeit, Schichtarbeit, Teilzeitarbeit
- Process: Arbeitsgang, Bearbeitung, bearbeiten

#### 7. Science & Technology (3 words)
**Fine domains**: governmental_institutions, human_resources, economic_systems

**Color**: `#F39C12` (Orange)

Examples:
- Systems: Arbeitsamt, Arbeitsmarkt, Arbeitskraft

#### 8. Nature & Biology (8 words)
**Fine domains**: domestic_animals, agricultural_activities, agricultural_structures

**Color**: `#27AE60` (Green)

Examples:
- Animals: Haushund, Hauskatze, Hausmaus, Haustier
- Agriculture: Feldarbeit, Treibhaus

### Complete Fine-to-Coarse Mapping (German)

```json
{
  "people_social": [
    "social_roles", "occupational_roles", "administrative_roles",
    "ownership_roles", "economic_roles", "collaborative_roles",
    "social_organizations", "employment_status", "employment_positions",
    "behavioral_qualities", "social_activities"
  ],
  "physical_objects": [
    "clothing_items", "material_possessions", "occupational_clothing",
    "occupational_equipment", "educational_materials", "documentary_records",
    "legal_documents", "administrative_documents"
  ],
  "built_environment": [
    "architectural_components", "architectural_types", "residential_architecture",
    "urban_architecture", "civic_buildings", "commercial_establishments",
    "educational_buildings", "industrial_buildings", "recreational_buildings",
    "recreational_structures", "agricultural_structures", "construction_activities"
  ],
  "abstract_concepts": [
    "core_morphology", "grammatical_forms", "size_modification",
    "abstract_qualities", "quality_assessment"
  ],
  "cognition_learning": [
    "educational_tasks", "educational_materials", "educational_buildings",
    "educational_spaces"
  ],
  "time_process": [
    "temporal_aspects", "temporal_organization", "temporal_work_patterns",
    "process_management", "process_activities", "action_verbs",
    "prefixed_verb", "employment_arrangements", "work_arrangements"
  ],
  "science_technology": [
    "governmental_institutions", "human_resources", "economic_systems"
  ],
  "nature_biology": [
    "domestic_animals", "agricultural_activities", "agricultural_structures"
  ]
}
```

---

## Dataset 3: English work/light Network

**File**: `ACL-network-work-light-enu.txt`
**Total Words**: 63
**Fine-Grained Domains**: 43

### Coarse-Grained Domain Mapping

| Coarse Domain | Fine-Grained Domains (count) | Total Words |
|---------------|----------------------------|-------------|
| **People & Social** | 6 domains | 10 |
| **Physical Objects** | 8 domains | 12 |
| **Built Environment** | 3 domains | 4 |
| **Abstract Concepts** | 7 domains | 9 |
| **Science & Technology** | 10 domains | 17 |
| **Cognition & Learning** | 4 domains | 7 |
| **Time & Process** | 4 domains | 7 |
| **Nature & Biology** | 1 domain | 4 |

### Detailed Coarse Domain Breakdown

#### 1. People & Social (10 words)
**Fine domains**: occupational_roles, social_categories, social_relationships, social_systems, collective_groups, organizational_units

**Color**: `#E74C3C` (Red)

Examples:
- Roles: worker, workman
- Groups: workforce, workgroup
- Relationships: workmate
- Categories: working-class

#### 2. Physical Objects (12 words)
**Fine domains**: household_furniture, clothing_categories, educational_tools, portable_equipment, automotive_equipment, household_equipment, technological_equipment, technological_features

**Color**: `#3498DB` (Blue)

Examples:
- Furniture: worktop
- Equipment: flashlight, headlight, nightlight
- Tools: workbook, highlighter
- Tech: workstation, lightbulb, spotlight

#### 3. Built Environment (4 words)
**Fine domains**: architectural_structures, urban_infrastructure, historical_institutions

**Color**: `#95A5A6` (Gray)

Examples:
- Structures: lighthouse
- Infrastructure: streetlight
- Historical: workhouse

#### 4. Abstract Concepts (9 words)
**Fine domains**: abstract_qualities, quality_assessment, comparative_forms, superlative_forms, measurement_concepts, physical_properties, emotional_states

**Color**: `#9B59B6` (Purple)

Examples:
- Qualities: workmanship, lightness
- Assessment: workable, delightful
- Forms: lighter, lightest
- Concepts: workload, lightweight
- Emotions: lighthearted, delighted

#### 5. Science & Technology (17 words)
**Fine domains**: physics_concepts, astronomical_units, illumination_systems, artificial_illumination, natural_illumination, natural_phenomena, artistic_practices, entertainment_events, process_management, technological_equipment

**Color**: `#F39C12` (Orange)

Examples:
- Physics: lightwave
- Astronomy: lightyear
- Illumination: lighting, candlelight, lamplight
- Natural: sunlight, moonlight, lightning
- Tech: workflow, lightbulb

#### 6. Cognition & Learning (7 words)
**Fine domains**: cognitive_processes, cognitive_states, philosophical_concepts, educational_spaces

**Color**: `#1ABC9C` (Teal)

Examples:
- Processes: enlighten, enlightening, highlight
- States: enlightened
- Concepts: enlightenment
- Spaces: workshop

#### 7. Time & Process (7 words)
**Fine domains**: temporal_aspects, temporal_units, temporal_illumination, process_management

**Color**: `#E67E22` (Dark Orange)

Examples:
- Aspects: working, worked, lighted
- Units: workday
- Illumination: daylight
- Management: workflow

#### 8. Nature & Biology (4 words)
**Fine domains**: natural_phenomena

**Color**: `#27AE60` (Green)

Examples:
- Natural: sunlight, moonlight, starlight, lightning

### Complete Fine-to-Coarse Mapping (English)

```json
{
  "people_social": [
    "occupational_roles", "social_categories", "social_relationships",
    "social_systems", "collective_groups", "organizational_units"
  ],
  "physical_objects": [
    "household_furniture", "clothing_categories", "educational_tools",
    "portable_equipment", "automotive_equipment", "household_equipment",
    "technological_equipment", "technological_features"
  ],
  "built_environment": [
    "architectural_structures", "urban_infrastructure", "historical_institutions"
  ],
  "abstract_concepts": [
    "abstract_qualities", "quality_assessment", "comparative_forms",
    "superlative_forms", "measurement_concepts", "physical_properties",
    "emotional_states"
  ],
  "science_technology": [
    "physics_concepts", "astronomical_units", "illumination_systems",
    "artificial_illumination", "natural_illumination", "natural_phenomena",
    "artistic_practices", "entertainment_events", "process_management",
    "technological_equipment"
  ],
  "cognition_learning": [
    "cognitive_processes", "cognitive_states", "philosophical_concepts",
    "educational_spaces"
  ],
  "time_process": [
    "temporal_aspects", "temporal_units", "temporal_illumination",
    "process_management"
  ],
  "nature_biology": [
    "natural_phenomena"
  ]
}
```

---

## Cross-Language Coarse Domain Summary

### Domain Distribution Across Languages

| Coarse Domain | Chinese (子) | German (Haus/Arbeit) | English (work/light) |
|---------------|-------------|---------------------|---------------------|
| **People & Social** | 37 (30%) | 23 (25%) | 10 (16%) |
| **Physical Objects** | 35 (28%) | 9 (10%) | 12 (19%) |
| **Nature & Biology** | 14 (11%) | 8 (9%) | 4 (6%) |
| **Abstract Concepts** | 20 (16%) | 5 (5%) | 9 (14%) |
| **Science & Technology** | 18 (15%) | 3 (3%) | 17 (27%) |
| **Built Environment** | 0 (0%) | 25 (27%) | 4 (6%) |
| **Cognition & Learning** | 0 (0%) | 4 (4%) | 7 (11%) |
| **Time & Process** | 0 (0%) | 14 (15%) | 7 (11%) |

### Key Insights

1. **Chinese 子 Network**:
   - Dominated by People/Social (30%) and Physical Objects (28%)
   - Strong presence of Science & Technology (15%)
   - No Built Environment or Time/Process domains
   - Reflects diverse semantic extensions of a single character

2. **German Haus/Arbeit Network**:
   - Most balanced distribution across all 8 domains
   - Strong emphasis on Built Environment (27%)
   - Reflects German compound morphology
   - Work-related concepts dominate time/process categories

3. **English work/light Network**:
   - Highest Science & Technology proportion (27%)
   - Strong abstract concepts (14%)
   - "Light" drives science/natural phenomena domains
   - "Work" drives social/organizational domains

---

## Implementation Notes for Color-Coding

### Recommended Approach

1. **Load domain mappings** from this document
2. **Map fine-grained → coarse-grained** domains using JSON mappings above
3. **Apply consistent colors** across all three languages
4. **Maintain color consistency** in visualizations

### Python Color Mapping Dictionary

```python
COARSE_DOMAIN_COLORS = {
    "people_social": "#E74C3C",      # Red
    "physical_objects": "#3498DB",   # Blue
    "nature_biology": "#27AE60",     # Green
    "abstract_concepts": "#9B59B6",  # Purple
    "science_technology": "#F39C12", # Orange
    "built_environment": "#95A5A6",  # Gray
    "cognition_learning": "#1ABC9C", # Teal
    "time_process": "#E67E22"        # Dark Orange
}

# Fine-to-coarse mappings
CHINESE_DOMAIN_MAP = {
    "historical_figures": "people_social",
    "social_relationships": "people_social",
    "physics_domain": "science_technology",
    "body_parts_slang": "nature_biology",
    "everyday_objects": "physical_objects",
    "abstract_concepts": "abstract_concepts",
    "food_plants": "nature_biology"
}

# (German and English mappings would be defined similarly)
```

### Usage Example

```python
def get_word_color(word, domain, language):
    """Get color for a word based on its domain"""

    # Get appropriate mapping for language
    if language == "chn":
        domain_map = CHINESE_DOMAIN_MAP
    elif language == "deu":
        domain_map = GERMAN_DOMAIN_MAP
    elif language == "enu":
        domain_map = ENGLISH_DOMAIN_MAP

    # Map fine domain to coarse domain
    coarse_domain = domain_map.get(domain, "abstract_concepts")

    # Get color
    return COARSE_DOMAIN_COLORS.get(coarse_domain, "#95A5A6")
```

---

## Visualization Recommendations

### For 2D/3D Scatter Plots

1. **Use consistent colors** across all three languages
2. **Add legend** showing coarse domain categories
3. **Enable filtering** by coarse domain
4. **Cluster labels** by coarse domain for clarity

### For Network Diagrams

1. **Node colors** = coarse domain
2. **Edge colors** = semantic relationship strength
3. **Node size** = word frequency or centrality
4. **Layout algorithm** = force-directed with domain clustering

### For Heatmaps

1. **Rows/Columns** = words
2. **Color intensity** = semantic similarity
3. **Border colors** = coarse domain grouping
4. **Hierarchical clustering** by domain first, then similarity

---

## Future Extensions

### Potential Additions

1. **Cross-linguistic domain alignment** studies
2. **Semantic similarity within domains**
3. **Domain transition patterns** (polysemy analysis)
4. **Frequency-weighted domain distributions**
5. **Temporal evolution** of domain distributions

### Additional Datasets

Consider adding:
- French semantic networks
- Spanish semantic networks
- Arabic semantic networks
- Cross-cultural domain comparisons

---

## References

- Dataset source: ACL (Ancient Chinese Learning) project
- Semantic domain framework: Based on cognitive linguistics principles
- Color palette: Flat UI color scheme for accessibility

**Document Version**: 1.0
**Last Updated**: 2025-10-19
