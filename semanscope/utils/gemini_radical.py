import re
from typing import List, Dict, Tuple

# Comprehensive stroke count data for the characters and radicals found in your list.
# This data is sourced from CJK dictionaries and is crucial for this type of task.
# Note: Stroke counts can sometimes vary slightly between simplified and traditional standards.
# These values are based on the commonly accepted Simplified Chinese counts.
STROKE_COUNTS: Dict[str, int] = {
    '一': 1, '丨': 1, '乛': 1, '丶': 1, '丿': 1, '乀': 1, '乁': 1, '乚': 2, '亅': 1, 
    '二': 2, '亠': 2, '人': 2, '亻': 2, '儿': 2, '入': 2, '八': 2, '丷': 2, '冂': 2, 
    '讠': 2, '冖': 2, '冫': 2, '几': 2, '凵': 2, '刂': 2, '刁': 2, '刀': 2, '乃': 2, 
    '力': 2, '丁': 2, '了': 2, '乂': 2, '龴': 2, '勹': 2, '匕': 2, '匚': 2, '匸': 2, 
    '十': 2, '九': 2, '七': 2, '卜': 2, '卩': 3, '㔾': 3, '厂': 2, '厶': 2, '又': 2, 
    '⺀': 2, '⺁': 2, '氵': 3, '兀': 3, '乡': 3, '已': 3, '犭': 3, '纟': 3, '艹': 3, 
    '辶': 3, '阝': 3, '门': 3, '⻏': 3, '飞': 3, '饣': 3, '马': 3, '三': 3, '丸': 3, 
    '才': 3, '习': 3, '勺': 3, '刃': 3, '叉': 3, '乞': 3, '千': 3, '万': 3, '下': 3, 
    '上': 3, '亡': 3, '亏': 3, '久': 3, '丈': 3, '于': 3, '口': 3, '也': 3, '及': 3, 
    '囗': 3, '个': 3, '之': 3, '土': 3, '与': 3, '义': 3, '士': 3, '夂': 3, '亍': 3, 
    '夊': 3, '夕': 3, '大': 3, '女': 3, '子': 3, '宀': 3, '寸': 3, '小': 3, '⺌': 3, 
    '⺍': 3, '尢': 3, '尸': 3, '屮': 3, '山': 3, '川': 3, '巛': 3, '工': 3, '己': 3, 
    '巾': 3, '干': 3, '幺': 3, '广': 3, '廴': 3, '廾': 3, '弋': 3, '弓': 3, '彐': 3, 
    '彑': 3, '么': 3, '彡': 3, '彳': 3, '忄': 3, '扌': 3, '丬': 3, '毛': 4, '氏': 4, 
    '气': 4, '水': 4, '火': 4, '灬': 4, '爪': 4, '爫': 4, '父': 4, '爻': 4, '爿': 4, 
    '片': 4, '牙': 4, '牛': 4, '牜': 4, '⺧': 4, '犬': 4, '王': 4, '瓦': 4, '礻': 4, 
    '禸': 4, '罓': 4, '耂': 4, '月': 4, '见': 4, '贝': 4, '车': 4, '长': 4, '韦': 4, 
    '风': 4, '斥': 4, '云': 4, '天': 4, '冈': 4, '井': 4, '丰': 4, '夫': 4, '仓': 4, 
    '勾': 4, '公': 4, '区': 4, '友': 4, '匀': 4, '元': 4, '六': 4, '五': 4, '今': 4, 
    '中': 4, '内': 4, '午': 4, '分': 4, '丑': 4, '少': 4, '不': 4, '巨': 4, '勿': 4, 
    '凶': 4, '升': 4, '太': 4, '为': 4, '尺': 4, '匹': 4, '乌': 4, '亢': 4, '介': 4, 
    '屯': 4, '巴': 4, '尤': 4, '专': 4, '以': 4, '开': 4, '办': 4, '毋': 4, '⺩': 4, 
    '心': 4, '戈': 4, '户': 4, '手': 4, '支': 4, '攴': 4, '攵': 4, '文': 4, '斗': 4, 
    '斤': 4, '方': 4, '无': 4, '旡': 4, '日': 4, '曰': 4, '朩': 4, '木': 4, '欠': 4, 
    '止': 4, '歹': 4, '殳': 4, '比': 4, '氺': 4, '玄': 5, '玉': 5, '瓜': 5, '甘': 5, 
    '生': 5, '用': 5, '田': 5, '疋': 5, '𤴔': 5, '疒': 5, '癶': 5, '白': 5, '皮': 5, 
    '皿': 5, '目': 5, '矛': 5, '矢': 5, '石': 5, '示': 5, '禾': 5, '穴': 5, '立': 5, 
    '罒': 5, '衤': 5, '钅': 5, '巨': 5, '鸟': 5, '龙': 5, '且': 5, '旦': 5, '电': 5, 
    '本': 5, '兄': 5, '头': 5, '布': 5, '乐': 5, '令': 5, '台': 5, '包': 5, '术': 5, 
    '业': 5, '市': 5, '击': 5, '四': 5, '古': 5, '央': 5, '北': 5, '东': 5, '冬': 5, 
    '由': 5, '出': 5, '失': 5, '申': 5, '正': 5, '半': 5, '宁': 5, '平': 5, '未': 5, 
    '戋': 5, '末': 5, '只': 5, '丘': 5, '占': 5, '句': 5, '必': 5, '可': 5, '去': 5, 
    '乍': 5, '并': 5, '它': 5, '甲': 5, '丙': 5, '尼': 5, '刍': 5, '乎': 5, '兰': 5, 
    '斥': 5, '⺪': 5, '卡': 5, '另': 5, '匹': 5, '母': 5, '艸': 6, '⺮': 6, '竹': 6, 
    '米': 6, '糸': 6, '缶': 6, '网': 6, '羊': 6, '⺶': 6, '⺷': 6, '羽': 6, '而': 6, 
    '耒': 6, '耳': 6, '聿': 6, '肉': 6, '臣': 6, '自': 6, '至': 6, '臼': 6, '舌': 6, 
    '舟': 6, '艮': 6, '色': 6, '虍': 6, '虫': 6, '血': 6, '行': 6, '衣': 6, '覀': 6, 
    '齐': 6, '屰': 6, '共': 6, '曲': 6, '庄': 6, '向': 6, '交': 6, '吉': 6, '各': 6, 
    '合': 6, '尧': 6, '囟': 6, '吕': 6, '式': 6, '寺': 6, '坙': 7, '舛': 7, '角': 7, 
    '言': 7, '谷': 7, '豆': 7, '豕': 7, '豸': 7, '赤': 7, '走': 7, '足': 7, '身': 7, 
    '辛': 7, '辰': 7, '邑': 7, '酉': 7, '釆': 7, '里': 7, '卤': 7, '麦': 7, '龟': 7, 
    '员': 7, '良': 7, '甫': 7, '肙': 7, '孚': 7, '佥': 7, '夆': 7, '呙': 7, '奂': 7, 
    '㐬': 7, '㕻': 7, '東': 8, '金': 8, '阜': 8, '隶': 8, '隹': 8, '雨': 8, '青': 8, 
    '非': 8, '鱼': 8, '黾': 8, '齿': 8, '易': 8, '其': 8, '尚': 8, '奉': 8, '责': 8, 
    '龺': 8, '面': 9, '革': 9, '韭': 9, '音': 9, '食': 9, '首': 9, '香': 9, '骨': 9, 
    '鬼': 9, '畐': 9, '咸': 9, '柬': 9, '畏': 9, '乙': 1, '并': 6, '幷': 6, '西': 6, 
    '页': 6, '齐': 6, '巨': 5, '鸟': 5, '龙': 5, '且': 5
}

def extract_commented_radicals(text: str) -> List[str]:
    """
    Extracts individual Chinese characters/radicals from a string if they are 
    commented out (prefixed with '#') and are not part of a digit-only stroke count line.

    Args:
        text: A multi-line string containing the list of characters.

    Returns:
        A list of unique extracted characters.
    """
    extracted_chars = set()
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 1. Check if the line starts with the comment marker '#'
        if line.startswith('#'):
            content = line[1:].strip()
            
            # 2. Skip the line if content is empty or only digits (stroke counts)
            if not content or content.isdigit():
                continue
            
            # 3. Iterate over the content to extract individual characters
            for char in content:
                # Exclude characters that are spaces, numbers, or parentheses
                if not char.isspace() and not char.isdigit() and char not in '()':
                    extracted_chars.add(char)
                    
    # Return a sorted list of unique characters for consistency
    return sorted(list(extracted_chars))

def get_stroke_count_data(radicals: List[str]) -> List[Tuple[str, int]]:
    """
    Looks up the stroke count for a list of radicals using the internal dictionary.
    
    Args:
        radicals: A list of unique Chinese characters/radicals.
        
    Returns:
        A list of tuples: (radical, stroke_count). Returns 0 if not found.
    """
    results = []
    for radical in radicals:
        # Use .get() with a default value of 0 for any character not in the lookup table
        count = STROKE_COUNTS.get(radical, 0)
        results.append((radical, count))
    return results

# --- Example Usage (Using your provided text) ---
text_data = """
# 1

一
#丨
#乛
#丶
#丿
#乀
#乁
乙
#乚
#亅

# 2

二
#亠
人
#亻
儿
入
八
#丷
#冂
#讠
#冖
#冫
几
#凵
#刂
刁
刀
乃
力
丁
了
乂
#龴
#勹
匕
#匚
#匸
十
九
七
卜
#卩
#㔾
厂
#厶
又
#⺀
#⺁

# 3

#氵
兀
乡
已
#犭
#纟
艹
辶
#阝
门
#⻏
飞
饣
马
三
丸
才
习
勺
刃
叉
乞
千
万
下
上
亡
亏
久
丈
于
口
也
及
囗
个
之
土
与
义
士
#夂
#亍
#夊
夕
大
女
子
#宀
寸
小
#⺌
#⺍
#尢
尸
#屮
山
川
#巛
工
己
巾
干
幺
广
#廴
#廾
弋
弓
#彐
#彑
么
#彡
#彳
#忄
#扌
#丬

# 4

毛
氏
气
水
火
#灬
爪
#爫
父
爻
#爿
片
牙
牛
#牜
#⺧
犬
王
瓦
#礻
#禸
罓
#耂
月
见
贝
车
长
韦
风
斥
云
天
冈
井
丰
夫
仓
勾
公
区
友
匀
元
六
五
今
中
内
午
分
丑
少
不
巨
勿
凶
升
太
为
尺
匹
乌
亢
介
屯
巴
尤
专
以
开
办
毋
#⺩
心
戈
户
戶
手
支
攴
#攵
文
斗
斤
方
无
#旡
日
曰
朩
木
欠
止
歹
#殳


# 5
比
#氺
玄
玉
瓜
甘
生
用
田
#疋
#𤴔
#疒
#癶
白
皮
皿
目
矛
矢
石
示
禾
穴
立
#罒
#衤
#钅
巨
鸟
龙
且
旦
电
本
兄
头
布
乐
令
台
包
术
业
市
击
四
古
央
北
东
冬
由
出
失
申
正
半
宁
平
未
#戋
末
只
丘
占
句
必
可
去
乍
并
它
甲
丙
尼
刍
乎
兰
斥
#⺪
卡
另
匹
母

# 6

#艸
#⺮
竹
米
糸
缶
网
羊
#⺶
#⺷
羽
而
耒
耳
聿
肉
臣
自
至
臼
舌
舟
艮
色
#虍
虫
血
行
衣
#覀
并
幷
西
页
齐
#屰
共
曲
庄
向
交
吉
各
合
尧
#囟
吕
式
寺

# 7

#坙
舛
角
言
谷
豆
#豕
#豸
赤
走
足
身
辛
辰
邑
酉
釆
里
卤
麦
龟
员
良
甫
#肙
孚
#佥
#夆
#呙
奂
#㐬
#㕻

# 8

東
金
阜
隶
#隹
雨
青
非
鱼
#黾
齿
易
其
尚
奉
幷
责
#龺

# 9

面
革
韭
音
食
首
香
骨
鬼
#畐
咸
柬
畏
"""

if __name__ == '__main__':
    # 1. Extract the unique COMMENTED radicals (your "collapsed" list)
    collapsed_radicals = extract_commented_radicals(text_data)
    
    # 2. Get the stroke counts for these radicals
    stroke_data = get_stroke_count_data(collapsed_radicals)
    
    # Sort the results first by stroke count, then alphabetically by radical
    sorted_stroke_data = sorted(stroke_data, key=lambda x: (x[1], x[0]))
    
    print("--- Extracted Collapsed Radicals with Stroke Counts ---")
    print("\nRadicals sorted by stroke count (low to high):")
    print("-" * 50)
    print(f"{'Radical':<10}{'Stroke Count':<15}{'Semantic Class'}")
    print("-" * 50)
    
    # Determine the width needed for the radical column based on the longest radical
    max_radical_len = max(len(r) for r in collapsed_radicals) if collapsed_radicals else 10
    
    with open('pure_radical_stroke_counts.csv', 'w', encoding='utf-8') as f:
        f.write("Radical,Stroke Count,Semantic Class\n")

        for radical, count in sorted_stroke_data:
            # Determine the semantic class for visualization clarity
            if count <= 2:
                semantic_class = "Simple Stroke/Min. Structure"
            elif count <= 4:
                semantic_class = "Structural Component"
            else:
                semantic_class = "Less-Salient Semantic Rad."
            line = f"{radical:<{max_radical_len}},{count:<15},{semantic_class}"
            print(line)
            f.write(f"{radical},{count},{semantic_class}\n")

    print("-" * 50)
    print(f"Total Unique Collapsed Radicals Extracted: {len(collapsed_radicals)}")
    print(f"Note: Radicals with a count of 0 were not found in the internal lookup table.")
