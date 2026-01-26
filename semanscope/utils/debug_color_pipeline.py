#!/usr/bin/env python3
"""
Debug color pipeline - trace colors from CSV parsing through to plotting
"""
import streamlit as st
import json
from pathlib import Path

def log_color_pipeline_step(step_name: str, data: dict, colors: list = None):
    """Log each step of the color pipeline for debugging"""
    log_dir = Path("../data/logs/color-coding")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "step": step_name,
        "timestamp": str(pd.Timestamp.now()),
        "data": data
    }

    if colors:
        log_entry["colors_sample"] = colors[:10]  # First 10 colors
        log_entry["colors_count"] = len(colors)

        # Count unique colors
        unique_colors = {}
        for color in colors:
            unique_colors[color] = unique_colors.get(color, 0) + 1
        log_entry["unique_colors"] = unique_colors

    # Append to pipeline log
    pipeline_log_file = log_dir / "color_pipeline_debug.jsonl"
    with open(pipeline_log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def analyze_numbers_colors():
    """Analyze specifically why numbers have mixed colors"""
    print("🔍 Analyzing numbers color assignment...")

    # Check if we have a recent log file
    log_file = Path("../data/logs/color-coding/ACL-word-v2-enu.json")
    if not log_file.exists():
        print("❌ No color mapping log found. Please load ACL-word-v2 first.")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    # Filter numbers domain entries
    numbers_entries = [entry for entry in log_data['word_colors'] if entry['domain'] == 'numbers']

    print(f"📊 Found {len(numbers_entries)} number words:")

    # Group by color to see the distribution
    color_groups = {}
    for entry in numbers_entries:
        color = entry['color']
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(entry)

    print(f"🎨 Numbers are using {len(color_groups)} different colors:")
    for color, entries in color_groups.items():
        words = [e['word'] for e in entries]
        print(f"   {color}: {words}")

    # Check if all numbers should be the same color
    expected_color = "#44FF44"  # Green for numbers
    if len(color_groups) > 1:
        print(f"❌ PROBLEM: Numbers should all be {expected_color} but found {len(color_groups)} different colors!")

        # Check if there's an issue in the CSV data
        print("\n🔍 Checking CSV data consistency:")
        types = set(entry['type'] for entry in numbers_entries)
        print(f"   Number types found: {types}")

        domains = set(entry['domain'] for entry in numbers_entries)
        print(f"   Domains found: {domains}")

    else:
        actual_color = list(color_groups.keys())[0]
        if actual_color == expected_color:
            print(f"✅ All numbers correctly assigned {expected_color}")
        else:
            print(f"❌ All numbers assigned {actual_color} instead of expected {expected_color}")

if __name__ == "__main__":
    analyze_numbers_colors()