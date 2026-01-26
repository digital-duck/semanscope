#!/usr/bin/env python3
"""
Parse v2.5 batch benchmark RA results from log file with multi-language support.

Enhanced parser for v2.5 datasets with 5 languages (EN, ZH, ES, DE, TR).
Handles variable number of languages, all cross-language pairs, and N-way RA.

Usage:
    python parse_batch_results_ra_v2.5.py --log-file LOG_FILE --output CSV_FILE [OPTIONS]

Examples:
    # Basic usage
    python parse_batch_results_ra_v2.5.py \
        --log-file cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.txt \
        --output results-v2.5.csv

    # With pivoted output for N-way RA
    python parse_batch_results_ra_v2.5.py \
        --log-file cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.txt \
        --output results-v2.5.csv \
        --pivot \
        --pivot-nway nway-pivot.csv
"""

import re
import csv
import json
import click
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def parse_ra_block(lines: List[str], start_idx: int) -> Tuple[Dict[str, Any], int]:
    """
    Parse a single RA result block from log lines (multi-language support).

    Args:
        lines: List of log file lines
        start_idx: Starting index to parse from

    Returns:
        Tuple of (parsed_data dict, next_index)
    """
    data = {
        'model': None,
        'dataset': None,
        'languages': None,
        'lang_codes': None,
        'n_pairs': None,
        'valid_pairs': None,
        'json_file': None,
    }

    # Store all within and cross RA scores dynamically
    within_scores = {}  # {lang_code: {'mean': x, 'std': y}}
    cross_scores = {}   # {lang_pair: {'mean': x, 'std': y}}
    nway_score = None

    lang_codes = []
    i = start_idx

    # Parse until we hit the next separator or end of file
    while i < len(lines):
        line = lines[i]

        # Check for next block separator
        if '=' * 40 in line and i > start_idx + 5:
            break

        # Extract model name
        if line.strip().startswith('Model:'):
            data['model'] = line.split('Model:')[1].strip()

        # Extract dataset name
        elif line.strip().startswith('Dataset:'):
            data['dataset'] = line.split('Dataset:')[1].strip()

        # Extract languages
        elif 'Languages:' in line:
            # Format: "Languages: english, chinese, spanish, german, turkish (EN, ZH, ES, DE, TR)"
            match = re.search(r'Languages:\s+([^(]+)\(([^)]+)\)', line)
            if match:
                lang_names = match.group(1).strip()
                lang_codes_str = match.group(2).strip()
                data['languages'] = lang_names
                lang_codes = [code.strip() for code in lang_codes_str.split(',')]
                data['lang_codes'] = lang_codes

        # Extract number of valid pairs
        elif '✓ Valid pairs:' in line:
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                data['valid_pairs'] = int(match.group(1))
                data['n_pairs'] = int(match.group(2))

        # Extract Cosine metrics
        elif line.strip() == 'Cosine:':
            i += 1
            # Parse all within-language and cross-language scores
            while i < len(lines):
                curr_line = lines[i].strip()

                # Stop at next metric type or separator
                if curr_line in ['Euclidean:', ''] or '=' * 40 in curr_line:
                    i -= 1  # Back up one line
                    break

                if '💾 Results saved:' in curr_line:
                    i -= 1
                    break

                # Within-language scores: "EN (within):  0.5133 ± 0.2061"
                within_match = re.match(r'([A-Z]{2})\s+\(within\):\s+([\d.-]+)\s*±\s*([\d.]+)', curr_line)
                if within_match:
                    lang = within_match.group(1)
                    mean = float(within_match.group(2))
                    std = float(within_match.group(3))
                    within_scores[lang] = {'mean': mean, 'std': std}
                    i += 1
                    continue

                # Cross-language scores: "EN-ZH (cross):   0.4083 ± 0.2030"
                cross_match = re.match(r'([A-Z]{2})-([A-Z]{2})\s+\(cross\):\s+([\d.-]+)\s*±\s*([\d.]+)', curr_line)
                if cross_match:
                    lang1 = cross_match.group(1)
                    lang2 = cross_match.group(2)
                    mean = float(cross_match.group(3))
                    std = float(cross_match.group(4))
                    lang_pair = f"{lang1}-{lang2}"
                    cross_scores[lang_pair] = {'mean': mean, 'std': std}
                    i += 1
                    continue

                # N-way score: "EN-ZH-ES-DE-TR (N-way):   0.2963 ± 0.1360"
                nway_match = re.match(r'([A-Z-]+)\s+\(N-way\):\s+([\d.-]+)\s*±\s*([\d.]+)', curr_line)
                if nway_match:
                    mean = float(nway_match.group(2))
                    std = float(nway_match.group(3))
                    nway_score = {'mean': mean, 'std': std}
                    i += 1
                    continue

                i += 1

        # Extract JSON file path
        elif '💾 Results saved:' in line:
            match = re.search(r'💾 Results saved:\s+(.+\.json)', line)
            if match:
                data['json_file'] = match.group(1).strip()

        i += 1

    # Add parsed scores to data
    data['within_scores'] = within_scores
    data['cross_scores'] = cross_scores
    data['nway_score'] = nway_score

    return data, i


def parse_log_file(log_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Parse RA batch benchmark log file to extract all results.

    Args:
        log_path: Path to the log file
        verbose: Print progress messages

    Returns:
        List of dictionaries containing parsed metrics
    """
    if verbose:
        click.echo(f"📖 Parsing log file: {log_path}")

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for result blocks with Dataset: followed by Model: and Languages:
        if ('Dataset:' in line and i > 0 and '=' * 40 in lines[i-1]):
            # Check if this is a valid result block (has Model and Languages nearby)
            has_model = False
            has_languages = False
            for j in range(i+1, min(i+5, len(lines))):
                if 'Model:' in lines[j]:
                    has_model = True
                if 'Languages:' in lines[j]:
                    has_languages = True

            if has_model and has_languages:
                data, next_i = parse_ra_block(lines, i)

                # Only add if we have valid data with metrics
                if (data['model'] and data['dataset'] and
                    (data['within_scores'] or data['cross_scores'] or data['nway_score'])):
                    results.append(data)
                    if verbose:
                        n_within = len(data.get('within_scores', {}))
                        n_cross = len(data.get('cross_scores', {}))
                        has_nway = '✓' if data.get('nway_score') else '✗'
                        click.echo(f"   ✓ {data['model']:<45} {data['dataset']:<45} "
                                 f"(W:{n_within}, C:{n_cross}, N:{has_nway})")

                i = next_i
            else:
                i += 1
        else:
            i += 1

    if verbose:
        click.echo(f"   Found {len(results)} result blocks")

    return results


def write_csv_wide(output_path: str, data_rows: List[Dict[str, Any]], verbose: bool = True) -> None:
    """
    Write extracted metrics to CSV file with all language combinations as columns.

    Creates a wide-format CSV with columns for each within/cross language pair.

    Args:
        output_path: Path to output CSV file
        data_rows: List of dictionaries containing metrics
        verbose: Print progress messages
    """
    if not data_rows:
        click.echo("❌ No data to write!", err=True)
        return

    # Collect all unique language codes and pairs from all results
    all_within_langs = set()
    all_cross_pairs = set()

    for row in data_rows:
        all_within_langs.update(row.get('within_scores', {}).keys())
        all_cross_pairs.update(row.get('cross_scores', {}).keys())

    # Sort for consistent column ordering
    within_langs = sorted(all_within_langs)
    cross_pairs = sorted(all_cross_pairs)

    # Build column headers
    base_cols = ['model', 'dataset', 'languages', 'n_pairs', 'valid_pairs']

    within_cols = []
    for lang in within_langs:
        within_cols.extend([f'cos_{lang}', f'cos_{lang}_std'])

    cross_cols = []
    for pair in cross_pairs:
        cross_cols.extend([f'cos_{pair}', f'cos_{pair}_std'])

    nway_cols = ['cos_nway', 'cos_nway_std']
    meta_cols = ['json_file']

    fieldnames = base_cols + within_cols + cross_cols + nway_cols + meta_cols

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in data_rows:
            csv_row = {
                'model': row.get('model'),
                'dataset': row.get('dataset'),
                'languages': row.get('languages'),
                'n_pairs': row.get('n_pairs'),
                'valid_pairs': row.get('valid_pairs'),
                'json_file': row.get('json_file'),
            }

            # Add within-language scores
            for lang in within_langs:
                if lang in row.get('within_scores', {}):
                    csv_row[f'cos_{lang}'] = row['within_scores'][lang]['mean']
                    csv_row[f'cos_{lang}_std'] = row['within_scores'][lang]['std']
                else:
                    csv_row[f'cos_{lang}'] = None
                    csv_row[f'cos_{lang}_std'] = None

            # Add cross-language scores
            for pair in cross_pairs:
                if pair in row.get('cross_scores', {}):
                    csv_row[f'cos_{pair}'] = row['cross_scores'][pair]['mean']
                    csv_row[f'cos_{pair}_std'] = row['cross_scores'][pair]['std']
                else:
                    csv_row[f'cos_{pair}'] = None
                    csv_row[f'cos_{pair}_std'] = None

            # Add N-way score
            if row.get('nway_score'):
                csv_row['cos_nway'] = row['nway_score']['mean']
                csv_row['cos_nway_std'] = row['nway_score']['std']
            else:
                csv_row['cos_nway'] = None
                csv_row['cos_nway_std'] = None

            writer.writerow(csv_row)

    if verbose:
        click.echo(f"✅ CSV file written: {output_path}")
        click.echo(f"   Total rows: {len(data_rows)}")
        click.echo(f"   Within-language scores: {len(within_langs)} languages")
        click.echo(f"   Cross-language scores: {len(cross_pairs)} pairs")


def create_pivoted_csv(
    data_rows: List[Dict[str, Any]],
    output_path: str,
    score_type: str = 'nway',
    verbose: bool = True
) -> None:
    """
    Create pivoted CSV with datasets as columns and models as rows.

    Args:
        data_rows: List of dictionaries containing metrics
        output_path: Path to output CSV file
        score_type: Type of score to pivot - 'nway', 'within_EN', 'cross_EN-ZH', etc.
        verbose: Print progress messages
    """
    if not data_rows:
        click.echo("❌ No data to write!", err=True)
        return

    # Get unique models and datasets
    models = []
    datasets = []
    dataset_n_pairs = {}

    for row in data_rows:
        if row['model'] not in models:
            models.append(row['model'])
        if row['dataset'] not in datasets:
            datasets.append(row['dataset'])
            dataset_n_pairs[row['dataset']] = row.get('valid_pairs', row.get('n_pairs', 'N/A'))

    # Build pivoted data structure
    pivoted_data = {model: {} for model in models}

    for row in data_rows:
        model = row['model']
        dataset = row['dataset']

        # Extract the appropriate score based on score_type
        if score_type == 'nway':
            score = row.get('nway_score')
        elif score_type.startswith('within_'):
            lang = score_type.replace('within_', '')
            score = row.get('within_scores', {}).get(lang)
        elif score_type.startswith('cross_'):
            pair = score_type.replace('cross_', '')
            score = row.get('cross_scores', {}).get(pair)
        else:
            score = None

        # Format: "mean ± std"
        if score and 'mean' in score and 'std' in score:
            value = f"{score['mean']:.4f} ± {score['std']:.4f}"
        else:
            value = "N/A"

        if model in pivoted_data:
            pivoted_data[model][dataset] = value

    # Create simplified column names with n_pairs
    dataset_short_names = {}
    for i, dataset in enumerate(datasets, 1):
        n_pairs = dataset_n_pairs.get(dataset, 'N/A')
        # Extract just the category name (e.g., "family-relations" from "NeurIPS-01-family-relations-v2.5")
        parts = dataset.split('-')
        if len(parts) >= 3:
            category = '-'.join(parts[2:]).replace('-v2.5', '').replace('-v2', '')
        else:
            category = dataset
        dataset_short_names[dataset] = f"{category} [n={n_pairs}]"

    # Write to CSV
    fieldnames = ['model'] + [dataset_short_names[ds] for ds in datasets]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model in models:
            row_data = {'model': model}

            for dataset in datasets:
                col_name = dataset_short_names[dataset]
                if dataset in pivoted_data[model]:
                    row_data[col_name] = pivoted_data[model][dataset]
                else:
                    row_data[col_name] = "N/A"

            writer.writerow(row_data)

    if verbose:
        click.echo(f"✅ Pivoted CSV file written: {output_path}")
        click.echo(f"   Score type: {score_type}")
        click.echo(f"   Models: {len(models)}")
        click.echo(f"   Datasets: {len(datasets)}")


@click.command()
@click.option(
    '--log-file',
    '-l',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to the batch benchmark log file'
)
@click.option(
    '--output',
    '-o',
    required=True,
    type=click.Path(dir_okay=False),
    help='Path to output CSV file with full metrics (wide format)'
)
@click.option(
    '--pivot/--no-pivot',
    default=False,
    help='Create pivoted CSV files with datasets as columns'
)
@click.option(
    '--pivot-nway',
    type=click.Path(dir_okay=False),
    help='Path to output pivoted CSV for N-way RA (requires --pivot)'
)
@click.option(
    '--pivot-score',
    type=str,
    default='nway',
    help='Score type for pivot: "nway", "within_EN", "cross_EN-ZH", etc. (default: nway)'
)
@click.option(
    '--verbose/--quiet',
    '-v/-q',
    default=True,
    help='Print progress messages'
)
def main(
    log_file: str,
    output: str,
    pivot: bool,
    pivot_nway: Optional[str],
    pivot_score: str,
    verbose: bool
):
    """
    Parse v2.5 RA batch benchmark log file with multi-language support.

    This enhanced parser handles variable number of languages and extracts:
    - All within-language RA scores
    - All cross-language RA pairs
    - N-way RA score

    Examples:

        # Basic usage - extract all metrics to wide CSV
        python parse_batch_results_ra_v2.5.py \\
            --log-file cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.txt \\
            --output results-v2.5.csv

        # With pivoted output for N-way RA
        python parse_batch_results_ra_v2.5.py \\
            --log-file cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.txt \\
            --output results-v2.5.csv \\
            --pivot \\
            --pivot-nway nway-pivot.csv

        # Pivot on specific language pair
        python parse_batch_results_ra_v2.5.py \\
            --log-file ... --output results.csv \\
            --pivot --pivot-score "cross_EN-ZH" \\
            --pivot-nway en-zh-pivot.csv
    """
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("📊 RA Batch Benchmark Results Parser (v2.5 Multi-Language)")
        click.echo("=" * 80 + "\n")

    # Parse log file
    data_rows = parse_log_file(log_file, verbose=verbose)

    if not data_rows:
        click.echo("❌ No results found in log file!", err=True)
        return

    # Write full metrics CSV (wide format)
    if verbose:
        click.echo(f"\n💾 Writing full metrics CSV (wide format)...")
    write_csv_wide(output, data_rows, verbose=verbose)

    # Create pivoted CSV if requested
    if pivot and pivot_nway:
        if verbose:
            click.echo(f"\n📊 Creating pivoted CSV...")
        create_pivoted_csv(
            data_rows,
            pivot_nway,
            score_type=pivot_score,
            verbose=verbose
        )
    elif pivot and not pivot_nway:
        click.echo("⚠️  Warning: --pivot specified but no --pivot-nway output file provided", err=True)

    # Summary
    if verbose:
        click.echo(f"\n" + "=" * 80)
        click.echo(f"Summary:")
        click.echo(f"  Log file:       {log_file}")
        click.echo(f"  Results found:  {len(data_rows)}")
        click.echo(f"\n  Output files:")
        click.echo(f"    Full metrics:  {output}")
        if pivot and pivot_nway:
            click.echo(f"    Pivoted ({pivot_score}): {pivot_nway}")
        click.echo("=" * 80 + "\n")


if __name__ == '__main__':
    main()
