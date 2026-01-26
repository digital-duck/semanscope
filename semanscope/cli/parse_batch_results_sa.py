#!/usr/bin/env python3
"""
Parse batch benchmark SA results from log file and create CSV with all metrics.

This script parses the SA batch benchmark log file to extract:
1. Model name, dataset name, languages
2. Cosine and Euclidean SA metrics
3. Inter/intra cluster spreads with SEM values
4. Reads JSON files for complete metadata

Usage:
    python parse_batch_results_sa.py --log-file LOG_FILE --output CSV_FILE [OPTIONS]

Examples:
    # Basic usage
python parse_batch_results_sa.py \
    --log-file cli_batch_benchmark_sa-2025-12-29.txt \
    --output results-sa-2025-12-29.csv

    # With pivoted output
    cd /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/semanscope/cli
python parse_batch_results_sa.py \
    --log-file cli_batch_benchmark_sa-2025-12-29-all-models-all-datasets-EN-ZH.txt \
    --output 2025-12-29-all-models-all-datasets-EN-ZH-sa.csv \
    --pivot \
    --pivot-cosine 2025-12-29-all-models-all-datasets-EN-ZH-sa-pivot-cos.csv \
    --pivot-euclidean 2025-12-29-all-models-all-datasets-EN-ZH-sa-pivot-euc.csv
"""

import json
import csv
import re
import click
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_log_file(log_path: str, verbose: bool = True) -> List[str]:
    """
    Parse log file to extract JSON file paths.

    Args:
        log_path: Path to the log file
        verbose: Print progress messages

    Returns:
        List of JSON file paths
    """
    if verbose:
        click.echo(f"📖 Parsing log file: {log_path}")

    json_files = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '💾 Results saved:' in line:
                # Extract the file path after "💾 Results saved: "
                match = re.search(r'💾 Results saved:\s+(.+\.json)', line)
                if match:
                    json_files.append(match.group(1).strip())

    if verbose:
        click.echo(f"   Found {len(json_files)} JSON files")

    return json_files


def extract_metrics_from_json(json_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract all required metrics from a JSON result file.

    Args:
        json_path: Path to the JSON file
        verbose: Print progress messages

    Returns:
        Dictionary with extracted metrics
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    results = data.get('results', {})
    euclidean = results.get('euclidean', {})
    cosine = results.get('cosine', {})

    # Join languages as comma-separated string
    languages = ','.join(metadata.get('languages', []))

    return {
        'model': metadata.get('model', ''),
        'dataset': metadata.get('dataset', ''),
        'n_words_expanded': metadata.get('n_words_expanded', 0),
        'languages': languages,
        # Euclidean metrics (shortened names)
        'eucl_sa': euclidean.get('sa_score', None),
        'eucl_sa_sem': euclidean.get('sem', None),
        'eucl_inter': euclidean.get('inter_spread', None),
        'eucl_inter_sem': euclidean.get('inter_spread_sem', None),
        'eucl_intra': euclidean.get('intra_spread', None),
        'eucl_intra_sem': euclidean.get('intra_spread_sem', None),
        # Cosine metrics (shortened names)
        'cos_sa': cosine.get('sa_score', None),
        'cos_sa_sem': cosine.get('sem', None),
        'cos_inter': cosine.get('inter_spread', None),
        'cos_inter_sem': cosine.get('inter_spread_sem', None),
        'cos_intra': cosine.get('intra_spread', None),
        'cos_intra_sem': cosine.get('intra_spread_sem', None),
    }


def write_csv(output_path: str, data_rows: List[Dict[str, Any]], verbose: bool = True) -> None:
    """
    Write extracted metrics to CSV file.

    Args:
        output_path: Path to output CSV file
        data_rows: List of dictionaries containing metrics
        verbose: Print progress messages
    """
    if not data_rows:
        click.echo("❌ No data to write!", err=True)
        return

    # Define column order (shortened names)
    fieldnames = [
        'model',
        'dataset',
        'n_words_expanded',
        'languages',
        'eucl_sa',
        'eucl_sa_sem',
        'eucl_inter',
        'eucl_inter_sem',
        'eucl_intra',
        'eucl_intra_sem',
        'cos_sa',
        'cos_sa_sem',
        'cos_inter',
        'cos_inter_sem',
        'cos_intra',
        'cos_intra_sem',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)

    if verbose:
        click.echo(f"✅ CSV file written: {output_path}")
        click.echo(f"   Total rows: {len(data_rows)}")


def create_pivoted_csv(
    data_rows: List[Dict[str, Any]],
    output_path: str,
    metric_type: str,
    verbose: bool = True
) -> None:
    """
    Create pivoted CSV with datasets as columns and models as rows.

    Args:
        data_rows: List of dictionaries containing metrics
        output_path: Path to output CSV file
        metric_type: Either 'euclidean' or 'cosine'
        verbose: Print progress messages
    """
    if not data_rows:
        click.echo("❌ No data to write!", err=True)
        return

    # Get unique models and datasets (preserve order from data)
    models = []
    datasets = []
    dataset_word_counts = {}

    for row in data_rows:
        if row['model'] not in models:
            models.append(row['model'])
        if row['dataset'] not in datasets:
            datasets.append(row['dataset'])
            dataset_word_counts[row['dataset']] = row['n_words_expanded']

    # Build pivoted data structure
    # Key: model, Value: dict of {dataset: value}
    pivoted_data = {model: {} for model in models}

    # Map metric_type to shortened column names
    metric_prefix = 'eucl' if metric_type == 'euclidean' else 'cos'

    for row in data_rows:
        model = row['model']
        dataset = row['dataset']
        sa_score = row.get(f'{metric_prefix}_sa')
        sem = row.get(f'{metric_prefix}_sa_sem')

        # Format: "sa_score ± sem"
        if sa_score is not None and sem is not None:
            value = f"{sa_score:.4f} ± {sem:.4f}"
        else:
            value = "N/A"

        if model in pivoted_data:
            pivoted_data[model][dataset] = value

    # Create simplified column names: DS1 [n_words], DS2 [n_words], etc.
    dataset_short_names = {}
    for i, dataset in enumerate(datasets, 1):
        n_words = dataset_word_counts[dataset]
        dataset_short_names[dataset] = f"DS{i} [n={n_words}]"

    # Write to CSV
    # Column order: model | DS1 | DS2 | DS3 | DS4 | ...
    fieldnames = ['model'] + [dataset_short_names[ds] for ds in datasets]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model in models:
            row_data = {'model': model}

            # Add SA score values for each dataset
            for dataset in datasets:
                col_name = dataset_short_names[dataset]
                if dataset in pivoted_data[model]:
                    row_data[col_name] = pivoted_data[model][dataset]
                else:
                    row_data[col_name] = "N/A"

            writer.writerow(row_data)

    if verbose:
        click.echo(f"✅ Pivoted CSV file written: {output_path}")
        click.echo(f"   Metric type: {metric_type}")
        click.echo(f"   Models: {len(models)}")
        click.echo(f"   Datasets: {len(datasets)}")


@click.command()
@click.option(
    '--log-file',
    '-l',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to the batch benchmark log file (e.g., cli_batch_benchmark_sa-2025-12-29.txt)'
)
@click.option(
    '--output',
    '-o',
    required=True,
    type=click.Path(dir_okay=False),
    help='Path to output CSV file with full metrics'
)
@click.option(
    '--pivot/--no-pivot',
    default=False,
    help='Create pivoted CSV files with datasets as columns'
)
@click.option(
    '--pivot-cosine',
    type=click.Path(dir_okay=False),
    help='Path to output pivoted CSV for Cosine SA scores (requires --pivot)'
)
@click.option(
    '--pivot-euclidean',
    type=click.Path(dir_okay=False),
    help='Path to output pivoted CSV for Euclidean SA scores (requires --pivot)'
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
    pivot_cosine: Optional[str],
    pivot_euclidean: Optional[str],
    verbose: bool
):
    """
    Parse SA batch benchmark log file and create CSV output.

    This script extracts model performance metrics from the batch benchmark
    log file and generates CSV files for analysis. It supports both full
    metrics output and pivoted tables for easier comparison.

    Examples:

        # Basic usage - extract all metrics to CSV
        python parse_batch_results_sa.py \\
            --log-file cli_batch_benchmark_sa-2025-12-29.txt \\
            --output results-sa-2025-12-29.csv

        # With pivoted output for SA scores
        python parse_batch_results_sa.py \\
            --log-file cli_batch_benchmark_sa-2025-12-29.txt \\
            --output results-sa.csv \\
            --pivot \\
            --pivot-cosine cosine-sa-pivot.csv \\
            --pivot-euclidean euclidean-sa-pivot.csv
    """
    if verbose:
        click.echo("\n" + "=" * 70)
        click.echo("📊 SA Batch Benchmark Results Parser")
        click.echo("=" * 70 + "\n")

    # Parse log file to get JSON file paths
    json_files = parse_log_file(log_file, verbose=verbose)

    if not json_files:
        click.echo("❌ No JSON files found in log file!", err=True)
        return

    # Extract metrics from all JSON files
    if verbose:
        click.echo(f"\n📊 Extracting metrics from JSON files...")

    data_rows = []
    for i, json_path in enumerate(json_files, 1):
        try:
            metrics = extract_metrics_from_json(json_path, verbose=False)
            data_rows.append(metrics)
            if verbose:
                click.echo(f"   [{i}/{len(json_files)}] {metrics['model']:<45} {metrics['dataset']}")
        except Exception as e:
            click.echo(f"   ❌ Error processing {json_path}: {e}", err=True)

    if not data_rows:
        click.echo("❌ No metrics extracted!", err=True)
        return

    # Write full metrics CSV
    if verbose:
        click.echo(f"\n💾 Writing full metrics CSV...")
    write_csv(output, data_rows, verbose=verbose)

    # Create pivoted CSVs if requested
    if pivot:
        if verbose:
            click.echo(f"\n📊 Creating pivoted CSV files...")

        if pivot_cosine:
            create_pivoted_csv(
                data_rows,
                pivot_cosine,
                'cosine',
                verbose=verbose
            )

        if pivot_euclidean:
            create_pivoted_csv(
                data_rows,
                pivot_euclidean,
                'euclidean',
                verbose=verbose
            )

        if not pivot_cosine and not pivot_euclidean:
            click.echo("⚠️  Warning: --pivot specified but no pivot output files provided", err=True)
            click.echo("   Use --pivot-cosine and/or --pivot-euclidean to specify output paths", err=True)

    # Summary
    if verbose:
        click.echo(f"\n" + "=" * 70)
        click.echo(f"Summary:")
        click.echo(f"  Log file:       {log_file}")
        click.echo(f"  Results found:  {len(data_rows)}")
        click.echo(f"\n  Output files:")
        click.echo(f"    Full metrics:  {output}")
        if pivot and pivot_cosine:
            click.echo(f"    Cosine pivot:  {pivot_cosine}")
        if pivot and pivot_euclidean:
            click.echo(f"    Euclidean pivot: {pivot_euclidean}")
        click.echo("=" * 70 + "\n")


if __name__ == '__main__':
    main()
