#!/usr/bin/env python3
"""
Script to filter out pure radicals from elemental characters for better visualization.
Creates a filtered version of the character dataset excluding abstract radicals.
"""

def get_radical_chars():
    """Return set of characters that are pure radicals (1-3 strokes mostly)"""
    # Based on 元字-chn.txt structure, most 1-3 stroke characters are pure radicals
    radical_chars = set()
    
    # 1-stroke radicals (lines 3-12)
    radicals_1_stroke = [
        # '一', '乙',
        '丨', '乛', '丶', '丿', '乀', '乁',  '乚', '亅'
    ]
    radical_chars.update(radicals_1_stroke)
    
    # 2-stroke radicals that are purely structural
    radicals_2_stroke = [
        # '儿', '八', '几','刁','厂','匕', 
        '亠', '亻',  '丷', '冂', '讠', '冖', '冫',  '凵', 
        '刂',  '勹', '匚', '匸', '卜', '卩', '㔾',  '厶', 
        '⺀', '⺁', '龴', '乂'
    ]
    radical_chars.update(radicals_2_stroke)
    
    # 3-stroke radicals that are purely structural  
    radicals_3_stroke = [
        # '尸','巾','广','幺',
        '氵', '兀', '犭', '纟', '艹', '辶', '阝', '门', '⻏', '饣', '夂', '夊',
        '宀', '⺌', '⺍', '尢',  '屮', '巛',    '廴','亍',
        '廾', '彐', '彑', '彡', '彳', '忄', '扌', '丬'
    ]
    radical_chars.update(radicals_3_stroke)
    
    # 4-stroke pure radicals
    radicals_4_stroke = [
        '灬', '爫', '爻', '爿', '牜', '⺧', '礻', '禸', '罓', '耂',
        '⺩', '攴', '攵', '朩', '殳', '犭','旡', '曰','殳','亢'
    ]
    radical_chars.update(radicals_4_stroke)
    
    # 5+ stroke structural components
    radicals_higher = [
        '氺', '疋','𤴔',  '⺪','疒', '癶', '罒', '衤', '钅',  '刍','戋','黾','㕻',
        '⺮', '⺶', '⺷', '耒', '臼','覀', '幷', '屰', '坙', '舛',
        '豸', '佥','肙','孚','呙','夆','奂','㐬','幷','龺','畐','隹',
        '豕','囟','虍','聿'
    ]
    radical_chars.update(radicals_higher)
    
    return radical_chars

def filter_characters(input_file, output_file, output_pure_radicals):
    """Filter out radical characters from input file"""
    radical_chars = get_radical_chars()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    filtered_lines = []
    for line in lines:
        char = line.strip()
        if char and char not in radical_chars and not char.startswith('#'):
            filtered_lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)
    
    with open(output_pure_radicals, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(list(radical_chars)))

    print(f"Filtered {len(lines) - len(filtered_lines)} radical characters")
    print(f"Remaining characters: {len(filtered_lines)}")

if __name__ == "__main__":
    input_file = "src/data/input/元字-chn.txt"
    output_file = "src/data/input/元字-filtered-chn.txt"
    output_pure_radicals = "src/data/input/元字-pure-radicals-chn.txt"
    
    filter_characters(input_file, output_file, output_pure_radicals)
    print(f"Created filtered file: {output_file}")
    print(f"Pure Radicals saved: {output_pure_radicals}")
