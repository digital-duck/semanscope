# ICML Challenge Dataset: Untranslatable & Difficult Concepts

**File**: `ICML-challenge-untranslatable-SA.csv`
**Created**: 2025-12-15
**Created by**: Claude Sonnet 4.5 (Anthropic)
**Purpose**: Stress-test embedding models on linguistically challenging concepts for ICML-2026 Semantic Affinity paper
**Dataset Rationale**: Proposed by Claude as Dataset #4 (challenge set) to test model limits

---

## Dataset Overview

### Statistics
- **Total Concepts**: 86 challenging linguistic phenomena
- **Languages**: 8 (English, Chinese, Spanish, French, German, Russian, Korean, Arabic)
- **Categories**: 3 main types of linguistic challenges
  - **Untranslatable words** (37 words): Culture-specific concepts with no direct equivalents
  - **Polysemous words** (24 words): Multiple unrelated meanings (homonyms)
  - **False friends** (25 words): Similar forms, different meanings across languages
- **Difficulty Levels**: Low, Medium, High (tagged per word)

### Purpose in ICML Paper

This dataset serves as a **challenge/stress-test set** to identify model failure modes:

**Expected Behavior**:
- **Untranslatable words**: SA should be **low** (approximations ≠ true equivalents)
- **Polysemous words**: SA should reveal if models conflate different meanings
- **False friends**: SA should expose models that rely on surface form over meaning

**Key Insight**: If models achieve high SA on this challenge set, they're likely overfitting to surface-level translation rather than capturing deep semantics.

---

## Category 1: Untranslatable Words (37 concepts)

### Definition
Words representing culture-specific concepts that have **no direct translation equivalents** in other languages. Translations are approximations or loanwords.

### Subcategories

#### German Philosophical Concepts
**Examples**: schadenfreude, wanderlust, zeitgeist, gestalt, angst

**Why These Challenge Models**:
- German philosophical tradition creates concepts absent in other languages
- "Schadenfreude" (幸灾乐祸) requires multi-word explanations in Chinese/English
- Tests whether models conflate approximation with equivalence

**Expected SA**: **0.35-0.45** (low, because translations are approximations)

---

#### Chinese Cultural Concepts
**Examples**: yuanfen (缘分), mianzi (面子), guanxi (关系), wu-wei (无为), yin-yang (阴阳)

**Why These Challenge Models**:
- Deep Chinese cultural/philosophical concepts
- "Yuanfen" (缘分) = "fated relationship" but no true English equivalent
- "Mianzi" (面子) = "face/social prestige" but culturally specific

**Expected SA**: **0.30-0.40** (very low, culturally untranslatable)

---

#### Romance Language Specific
**Examples**: saudade (Portuguese), sobremesa (Spanish), duende (Spanish)

**Why These Challenge Models**:
- "Saudade" (乡愁) = deep nostalgic longing, no English equivalent
- "Sobremesa" = after-meal conversation culture (餐后闲聊)
- Tests Western-language-centric model bias

**Expected SA**: **0.40-0.50** (moderate, Romance languages may align better)

---

#### Nordic Concepts
**Examples**: hygge (Danish), lagom (Swedish)

**Why These Challenge Models**:
- "Hygge" (温馨) = cozy contentment, culturally specific
- "Lagom" (适度) = "just the right amount", Swedish cultural value

**Expected SA**: **0.35-0.45** (low, Nordic cultural specificity)

---

#### Japanese Aesthetic Concepts
**Examples**: wabi-sabi (侘寂), komorebi (木漏れ日), ikigai (生きがい), mono-no-aware (物の哀れ)

**Why These Challenge Models**:
- Japanese aesthetic philosophy with no direct equivalents
- "Wabi-sabi" = beauty in imperfection
- "Komorebi" = sunlight filtering through leaves (highly specific sensory concept)

**Expected SA**: **0.30-0.40** (very low, Japanese cultural/aesthetic specificity)

---

#### Korean Emotional Concepts
**Examples**: han (한), jeong (정), nunchi (눈치)

**Why These Challenge Models**:
- "Han" (恨) = deep collective sorrow/resentment, Korean cultural concept
- "Nunchi" (눈치) = social awareness/reading the room
- Tests Korean-specific emotional semantics

**Expected SA**: **0.35-0.45** (low, Korean cultural specificity)

---

#### Middle Eastern/African Concepts
**Examples**: tarof (Persian), inshallah (Arabic), ubuntu (African)

**Why These Challenge Models**:
- "Tarof" (客套) = Persian ritual courtesy
- "Inshallah" (إن شاء الله) = "if God wills", cultural/religious concept
- "Ubuntu" (人道) = African philosophy of interconnected humanity

**Expected SA**: **0.35-0.45** (low, cultural/religious specificity)

---

#### Extreme Challenge: Rare Untranslatables
**Examples**: mamihlapinatapai (Yaghan), toska (Russian), goya (Urdu)

**Why These Challenge Models**:
- "Mamihlapinatapai" = wordless mutual understanding (心照不宣)
- "Toska" (тоска) = profound spiritual anguish (Russian)
- Tests model behavior on extremely rare concepts

**Expected SA**: **0.25-0.35** (very low, extreme untranslatability)

---

## Category 2: Polysemous Words (24 concepts)

### Definition
Words with **multiple unrelated meanings** (homonyms). Tests whether models conflate different senses.

### Examples with Distinct Meanings

#### bank
- **bank_financial**: 银行 (financial institution)
- **bank_river**: 河岸 (riverbank)

**Why This Challenges Models**:
- Completely unrelated meanings
- Tests if embeddings distinguish senses or conflate them

**Expected SA**: **Should differ across senses**
- If SA(bank_financial) ≈ SA(bank_river), model conflates meanings
- If SA(bank_financial) ≠ SA(bank_river), model distinguishes senses ✅

---

#### spring
- **spring_season**: 春天 (season)
- **spring_water**: 泉水 (water source)
- **spring_coil**: 弹簧 (mechanical coil)

**Why This Challenges Models**:
- Three completely unrelated concepts
- Tests fine-grained sense disambiguation

**Expected SA**: **Should vary by sense**

---

#### light
- **light_illumination**: 光 (brightness)
- **light_weight**: 轻的 (not heavy)

**Why This Challenges Models**:
- Different parts of speech (noun vs adjective)
- Tests whether models capture syntactic differences

---

#### fair
- **fair_just**: 公平 (justice)
- **fair_festival**: 集市 (market/event)
- **fair_pale**: 白皙 (pale skin)

**Expected Behavior**: Good models should show **low SA between different senses** of the same word.

---

## Category 3: False Friends (25 word pairs)

### Definition
Words that **look/sound similar across languages but have different meanings**. Tests whether models rely on surface form vs semantics.

### Examples

#### gift (English) vs Gift (German)
- **gift_english**: 礼物 (present)
- **gift_german**: 毒药 (poison)

**Why This Challenges Models**:
- Identical spelling, opposite meanings
- Tests if models use surface orthography vs semantics

**Expected SA**: **Should be LOW** (completely different meanings)
- If SA is high, model is using surface form ❌
- If SA is low, model captures semantic difference ✅

---

#### actual (English) vs actual (Spanish)
- **actual_english**: 实际 (real, factual)
- **actual_spanish**: 当前 (current, present)

**Why This Challenges Models**:
- Partial overlap but different primary meanings
- Common translation error

---

#### embarrassed (English) vs embarazada (Spanish)
- **embarrassed_english**: 尴尬 (feeling awkward)
- **embarazada_spanish**: 怀孕 (pregnant)

**Why This Challenges Models**:
- Extremely common false friend
- Famously causes translation errors

**Expected SA**: **Should be VERY LOW** (unrelated meanings)

---

#### preservative (English) vs preservativo (Spanish)
- **preservative_english**: 防腐剂 (food additive)
- **preservativo_spanish**: 避孕套 (condom)

**Why This Challenges Models**:
- Completely different semantic domains
- Tests pharmaceutical/food vocabulary alignment

---

#### sensible (English) vs sensible (French)
- **sensible_english**: 明智 (rational, wise)
- **sensible_french**: 敏感 (sensitive)

**Why This Challenges Models**:
- Similar forms, related but distinct meanings

---

### Analysis Strategy

**For False Friends**:
```python
# Expected pattern for good models
SA(false_friend_pair) < SA(true_translation_pair)

# Example:
SA(gift_english, gift_german) < 0.40  # Should be low (different meanings)
SA(gift_english, 礼物) > 0.60  # Should be high (true translation)
```

Models that show **high SA for false friends** are relying on **surface form**, not semantics.

---

## Expected Results Across Categories

### Hypothesis Table

| Category | Expected SA (Good Model) | Expected SA (Poor Model) | Interpretation |
|----------|--------------------------|--------------------------|----------------|
| **Untranslatable** | 0.30-0.45 | 0.50-0.55 | Poor models treat approximations as equivalents |
| **Polysemous (same sense)** | 0.55-0.65 | 0.50-0.60 | Good models align same meanings |
| **Polysemous (different sense)** | 0.30-0.45 | 0.50-0.60 | Good models distinguish senses; poor models conflate |
| **False Friends** | 0.25-0.35 | 0.50-0.65 | Poor models use surface form; good models use semantics |

### Diagnostic Power

**If SA is uniformly high (>0.55) across all categories**:
- Model is **NOT capturing deep semantics**
- Model relies on surface-level translation or orthography
- Model fails linguistic nuance test

**If SA varies appropriately**:
- Untranslatable: LOW ✅
- Polysemous (different senses): LOW ✅
- False friends: LOW ✅
- Model demonstrates semantic understanding

---

## Usage in ICML Paper

### As Challenge Set

**Primary Role**: Identify model failure modes that standard benchmarks miss

**Key Questions**:
1. Do models achieve low SA on untranslatable words? (If not, they conflate approximation with equivalence)
2. Do models distinguish polysemous senses? (If not, they lack sense disambiguation)
3. Do models resist false friend traps? (If not, they rely on surface form)

### Comparison Across Models

**Prediction**:
- **BERT-based models**: Will struggle on untranslatables but distinguish polysemes/false friends
- **LLM-based models**: May conflate all categories (uniformly high SA = failure)
- **Excellent models**: Show graded SA reflecting true linguistic difficulty

---

## File Format

### CSV Structure
```csv
english,chinese,spanish,french,german,russian,korean,arabic,category,difficulty
schadenfreude,幸灾乐祸,schadenfreude,schadenfreude,Schadenfreude,злорадство,고소함,شماتة,untranslatable,high
bank_financial,银行,banco,banque,Bank,банк,은행,بنك,polysemous,medium
gift_english,礼物,regalo,cadeau,Geschenk,подарок,선물,هدية,false_friends,low
...
```

### Difficulty Levels
- **Low** (33 words, 38%): Relatively common concepts, moderate challenge
- **Medium** (27 words, 31%): Significant linguistic challenge
- **High** (26 words, 30%): Extreme untranslatability or false friend traps

---

## Translation Methodology

### For Untranslatable Words
- **Method**: Best approximation in target language
- **Note**: Translations are inherently imperfect (that's the point!)
- **Loanwords**: Some languages adopt loanwords (e.g., schadenfreude in English)

### For Polysemous Words
- **Method**: Disambiguate by appending sense (e.g., bank_financial vs bank_river)
- **English source**: Includes sense label to clarify meaning

### For False Friends
- **Method**: Translate each "friend" to its actual meaning
- **Comparison**: Shows divergence despite surface similarity

---

## Sample Translations

### Untranslatable Examples
```
schadenfreude → 幸灾乐祸 (Chinese: "happy disaster happy joy", approximation)
yuanfen (缘分) → destino_compartido (Spanish: "shared destiny", approximation)
hygge → 温馨 (Chinese: "warm/cozy", partial match)
```

### Polysemous Examples
```
bank_financial → 银行 (bank), banco (bank), banque (bank)
bank_river → 河岸 (riverbank), orilla (riverbank), rive (riverbank)
[Different Chinese/Spanish/French words for different meanings!]
```

### False Friends Examples
```
gift_english → 礼物 (present)
gift_german → 毒药 (poison)
[Identical English/German spelling, completely different translations!]
```

---

## Research Questions

### Primary Questions
1. Do models achieve appropriately **low SA** on untranslatable words?
2. Do models **distinguish** different senses of polysemous words?
3. Do models **resist** false friend traps (low SA for similar-form-different-meaning)?

### Diagnostic Questions
1. Which model shows largest SA difference between polysemous senses? (Best sense disambiguation)
2. Which model shows lowest SA on false friends? (Best semantic grounding)
3. Do BERT-based models outperform LLMs on this challenge set?

### Failure Mode Analysis
1. If SA(untranslatable) > 0.55, model treats approximations as equivalents
2. If SA(polyseme_A) ≈ SA(polyseme_B), model conflates senses
3. If SA(false_friends) > 0.50, model uses surface form over semantics

---

## Citation

```bibtex
@dataset{challenge_untranslatable_icml2026,
  title={Challenge Dataset: Untranslatable Words, Polysemes, and False Friends for Multilingual Embedding Stress Testing},
  author={Claude Sonnet 4.5 (Anthropic) and Yuan, Jian},
  year={2025},
  note={Created by Claude Sonnet 4.5 as challenge dataset for ICML-2026 Semantic Affinity benchmark},
  howpublished={ICML-2026 Semantic Affinity Paper - Dataset \#4 (Challenge Set)}
}
```

---

## Acknowledgment

This challenge dataset was proposed and generated by **Claude Sonnet 4.5** (Anthropic) in collaboration with **Jian Yuan** (Digital Duck / ZiNets Project) to stress-test multilingual embedding models on linguistically difficult phenomena.

---

**Last Updated**: 2025-12-15
**Purpose**: Challenge/stress-test dataset to identify model failure modes
**Version**: 1.0
**Status**: ✅ Production Ready
