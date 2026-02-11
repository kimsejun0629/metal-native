#!/usr/bin/env python3
"""Visualization for MPS benchmark results.

Generates publication-quality charts comparing PyTorch MPS vs MetalNative-style
optimized operations on Apple Silicon.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ============================================================================
# Style Configuration
# ============================================================================

# Color palette
COLORS = {
    'cpu': '#6C757D',         # Gray
    'mps': '#0D6EFD',        # Blue
    'optimized': '#DC3545',   # Red
    'memory_naive': '#FFC107', # Yellow
    'memory_flash': '#198754', # Green
    'bg': '#FAFAFA',
    'grid': '#E0E0E0',
    'text': '#212529',
}

def setup_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#CCCCCC',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': COLORS['grid'],
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'figure.dpi': 150,
    })


# ============================================================================
# Chart Functions
# ============================================================================

def plot_matmul(ax, data):
    """Plot matrix multiplication performance."""
    configs = [r['config'] for r in data]
    cpu_times = [r['cpu_time_ms'] for r in data]
    mps_times = [r['mps_time_ms'] for r in data]
    opt_times = [r['optimized_mps_time_ms'] for r in data]

    x = np.arange(len(configs))
    width = 0.25

    bars1 = ax.bar(x - width, cpu_times, width, label='CPU', color=COLORS['cpu'], alpha=0.85)
    bars2 = ax.bar(x, mps_times, width, label='PyTorch MPS', color=COLORS['mps'], alpha=0.85)
    bars3 = ax.bar(x + width, opt_times, width, label='Optimized MPS (FP16)', color=COLORS['optimized'], alpha=0.85)

    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Time (ms) - lower is better')
    ax.set_title('Matrix Multiplication Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')

    # Add speedup annotations on optimized bars
    for i, (bar, r) in enumerate(zip(bars3, data)):
        speedup = r.get('optimized_speedup_vs_cpu')
        if speedup:
            ax.annotate(f'{speedup:.1f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, fontweight='bold',
                        color=COLORS['optimized'])


def plot_matmul_gflops(ax, data):
    """Plot GFLOPS comparison for matmul."""
    configs = [r['config'] for r in data]
    cpu_gf = [r.get('cpu_gflops', 0) or 0 for r in data]
    mps_gf = [r.get('mps_gflops', 0) or 0 for r in data]
    opt_gf = [r.get('optimized_gflops', 0) or 0 for r in data]

    x = np.arange(len(configs))
    width = 0.25

    ax.bar(x - width, cpu_gf, width, label='CPU', color=COLORS['cpu'], alpha=0.85)
    ax.bar(x, mps_gf, width, label='PyTorch MPS', color=COLORS['mps'], alpha=0.85)
    ax.bar(x + width, opt_gf, width, label='Optimized MPS', color=COLORS['optimized'], alpha=0.85)

    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('GFLOPS - higher is better')
    ax.set_title('Matrix Multiplication Throughput (GFLOPS)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()


def plot_attention(ax, data):
    """Plot attention performance comparison."""
    configs = [r['config'] for r in data]
    # Shorten labels
    labels = []
    for c in configs:
        parts = c.split('_')
        labels.append(f"S{parts[2][1:]}")  # Just seq_len

    cpu_times = [r['cpu_time_ms'] for r in data]
    mps_times = [r['mps_time_ms'] for r in data]
    opt_times = [r['optimized_mps_time_ms'] for r in data]

    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, cpu_times, width, label='CPU (Naive)', color=COLORS['cpu'], alpha=0.85)
    ax.bar(x, mps_times, width, label='MPS (Naive)', color=COLORS['mps'], alpha=0.85)
    ax.bar(x + width, opt_times, width, label='MPS (SDPA/Flash)', color=COLORS['optimized'], alpha=0.85)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms) - lower is better')
    ax.set_title('Attention: Naive vs Flash Attention')
    ax.set_xticks(x)
    ax.set_xticklabels([r['config'] for r in data], rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_yscale('log')


def plot_attention_speedup(ax, data):
    """Plot attention speedup line chart."""
    seq_lens = []
    speedups_mps_vs_cpu = []
    speedups_opt_vs_mps = []
    speedups_opt_vs_cpu = []

    for r in data:
        parts = r['config'].split('_')
        seq_len = int(parts[2][1:])
        seq_lens.append(seq_len)
        speedups_mps_vs_cpu.append(r.get('mps_speedup_vs_cpu', 0) or 0)
        speedups_opt_vs_mps.append(r.get('optimized_speedup_vs_mps', 0) or 0)
        speedups_opt_vs_cpu.append(r.get('optimized_speedup_vs_cpu', 0) or 0)

    ax.plot(range(len(data)), speedups_mps_vs_cpu, 'o-', color=COLORS['mps'],
            label='MPS vs CPU', linewidth=2, markersize=6)
    ax.plot(range(len(data)), speedups_opt_vs_cpu, 's-', color=COLORS['optimized'],
            label='Optimized vs CPU', linewidth=2, markersize=6)
    ax.plot(range(len(data)), speedups_opt_vs_mps, '^--', color='#9B59B6',
            label='Optimized vs MPS', linewidth=1.5, markersize=5)

    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Attention Speedup Analysis')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([r['config'] for r in data], rotation=45, ha='right', fontsize=7)
    ax.legend()


def plot_memory_efficiency(ax, data):
    """Plot memory efficiency comparison."""
    names = [r['name'] for r in data]
    naive_mem = [r['naive_peak_mb'] for r in data]
    flash_mem = [r['flash_peak_mb'] for r in data]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, naive_mem, width, label='Naive Attention',
                    color=COLORS['memory_naive'], alpha=0.85, edgecolor='#CC9900')
    bars2 = ax.bar(x + width/2, flash_mem, width, label='Flash Attention (MetalNative)',
                    color=COLORS['memory_flash'], alpha=0.85, edgecolor='#146C43')

    # Add savings annotations
    for i, r in enumerate(data):
        ax.annotate(f'-{r["savings_pct"]:.0f}%',
                    xy=(x[i] + width/2, flash_mem[i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=COLORS['memory_flash'])

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Efficiency: Naive vs Flash Attention')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()


def plot_operation_comparison(ax, results):
    """Plot overall speedup comparison across all operations."""
    categories = []
    mps_speedups = []
    opt_speedups = []

    # Aggregate by operation type
    for key in ['matmul', 'attention', 'softmax', 'layernorm', 'elementwise', 'reduction', 'conv2d', 'transformer']:
        if key not in results or not results[key]:
            continue

        data = results[key]
        avg_mps = np.mean([r.get('mps_speedup_vs_cpu', 0) or 0 for r in data])
        avg_opt = np.mean([r.get('optimized_speedup_vs_cpu', 0) or 0 for r in data])

        label_map = {
            'matmul': 'MatMul',
            'attention': 'Attention',
            'softmax': 'Softmax',
            'layernorm': 'LayerNorm',
            'elementwise': 'Elementwise',
            'reduction': 'Reduction',
            'conv2d': 'Conv2D',
            'transformer': 'Transformer',
        }
        categories.append(label_map.get(key, key))
        mps_speedups.append(avg_mps)
        opt_speedups.append(avg_opt)

    x = np.arange(len(categories))
    width = 0.35

    ax.barh(x - width/2, mps_speedups, width, label='PyTorch MPS',
            color=COLORS['mps'], alpha=0.85)
    ax.barh(x + width/2, opt_speedups, width, label='Optimized MPS (MetalNative)',
            color=COLORS['optimized'], alpha=0.85)

    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5, label='CPU baseline')
    ax.set_xlabel('Speedup vs CPU (x) - higher is better')
    ax.set_title('Overall Speedup by Operation Type')
    ax.set_yticks(x)
    ax.set_yticklabels(categories)
    ax.legend(loc='lower right')

    # Add value labels
    for i, (mps_s, opt_s) in enumerate(zip(mps_speedups, opt_speedups)):
        ax.annotate(f'{mps_s:.1f}x', xy=(mps_s, i - width/2),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8, color=COLORS['mps'])
        ax.annotate(f'{opt_s:.1f}x', xy=(opt_s, i + width/2),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8, color=COLORS['optimized'])


def plot_transformer_blocks(ax, data):
    """Plot transformer block performance."""
    configs = [r['config'].split('_')[0] for r in data]
    cpu_times = [r['cpu_time_ms'] for r in data]
    mps_times = [r['mps_time_ms'] for r in data]
    opt_times = [r['optimized_mps_time_ms'] for r in data]

    x = np.arange(len(configs))
    width = 0.25

    ax.bar(x - width, cpu_times, width, label='CPU (Standard)', color=COLORS['cpu'], alpha=0.85)
    ax.bar(x, mps_times, width, label='MPS (Standard)', color=COLORS['mps'], alpha=0.85)
    ax.bar(x + width, opt_times, width, label='MPS (Optimized)', color=COLORS['optimized'], alpha=0.85)

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Time (ms) - lower is better')
    ax.set_title('Transformer Block: Standard vs Optimized')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')

    # Add speedup annotations
    for i, (bar_x, r) in enumerate(zip(x, data)):
        speedup = r.get('optimized_speedup_vs_mps', 0) or 0
        ax.annotate(f'{speedup:.2f}x',
                    xy=(bar_x + width, opt_times[i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=COLORS['optimized'])


def plot_norm_and_softmax(ax, results):
    """Plot LayerNorm and Softmax comparison."""
    categories = []
    mps_times = []
    opt_times = []
    speedups = []

    for key in ['softmax', 'layernorm']:
        if key not in results:
            continue
        for r in results[key]:
            label = f"{key.capitalize()}\n{r['config'][:20]}"
            categories.append(label)
            mps_times.append(r['mps_time_ms'])
            opt_times.append(r['optimized_mps_time_ms'])
            speedups.append(r.get('optimized_speedup_vs_mps', 0) or 0)

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, mps_times, width, label='MPS Standard', color=COLORS['mps'], alpha=0.85)
    ax.bar(x + width/2, opt_times, width, label='MPS Optimized', color=COLORS['optimized'], alpha=0.85)

    ax.set_xlabel('Operation')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Memory-Bound Operations: MPS vs Optimized')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7, rotation=45, ha='right')
    ax.legend()


# ============================================================================
# Main Visualization
# ============================================================================

def generate_visualizations(results: dict, output_dir: str):
    """Generate all visualization charts.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save charts
    """
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    system_info = results.get('system_info', {})
    chip = system_info.get('chip', 'Apple Silicon')
    memory = system_info.get('memory', '')
    torch_ver = system_info.get('torch_version', '')
    timestamp = results.get('timestamp', '')

    suptitle_suffix = f"{chip} | {memory} | PyTorch {torch_ver}"

    # ---- Chart 1: Overview Dashboard (2x2) ----
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'MetalNative vs PyTorch MPS - Performance Overview\n{suptitle_suffix}',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08)

    if results.get('matmul'):
        ax1 = fig.add_subplot(gs[0, 0])
        plot_matmul(ax1, results['matmul'])

    if results.get('attention'):
        ax2 = fig.add_subplot(gs[0, 1])
        plot_attention(ax2, results['attention'])

    if results.get('transformer'):
        ax3 = fig.add_subplot(gs[1, 0])
        plot_transformer_blocks(ax3, results['transformer'])

    ax4 = fig.add_subplot(gs[1, 1])
    plot_operation_comparison(ax4, results)

    path1 = os.path.join(output_dir, '01_overview_dashboard.png')
    fig.savefig(path1, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ---- Chart 2: Attention Deep Dive (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Attention Performance Deep Dive\n{suptitle_suffix}',
                 fontsize=15, fontweight='bold')
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.08)

    if results.get('attention'):
        plot_attention(axes[0, 0], results['attention'])
        plot_attention_speedup(axes[0, 1], results['attention'])

    if results.get('memory'):
        plot_memory_efficiency(axes[1, 0], results['memory'])

    # Attention time by seq_len (line chart)
    if results.get('attention'):
        ax = axes[1, 1]
        data = results['attention']
        configs = [r['config'] for r in data]
        mps_t = [r['mps_time_ms'] for r in data]
        opt_t = [r['optimized_mps_time_ms'] for r in data]

        ax.plot(range(len(data)), mps_t, 'o-', color=COLORS['mps'],
                label='MPS Naive', linewidth=2, markersize=6)
        ax.plot(range(len(data)), opt_t, 's-', color=COLORS['optimized'],
                label='MPS Flash (SDPA)', linewidth=2, markersize=6)
        ax.fill_between(range(len(data)), mps_t, opt_t, alpha=0.15, color=COLORS['optimized'])
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Attention Latency: Naive vs Flash')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=7)
        ax.legend()

    path2 = os.path.join(output_dir, '02_attention_deep_dive.png')
    fig.savefig(path2, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path2}")

    # ---- Chart 3: Compute Operations (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Compute & Memory-Bound Operations\n{suptitle_suffix}',
                 fontsize=15, fontweight='bold')
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.08)

    if results.get('matmul'):
        plot_matmul_gflops(axes[0, 0], results['matmul'])

    if results.get('conv2d'):
        data = results['conv2d']
        configs = [r['config'] for r in data]
        x = np.arange(len(configs))
        width = 0.25
        axes[0, 1].bar(x - width, [r['cpu_time_ms'] for r in data], width,
                        label='CPU', color=COLORS['cpu'], alpha=0.85)
        axes[0, 1].bar(x, [r['mps_time_ms'] for r in data], width,
                        label='MPS', color=COLORS['mps'], alpha=0.85)
        axes[0, 1].bar(x + width, [r['optimized_mps_time_ms'] for r in data], width,
                        label='Optimized', color=COLORS['optimized'], alpha=0.85)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].set_title('Conv2D Performance')
        axes[0, 1].legend()

    plot_norm_and_softmax(axes[1, 0], results)

    # Reduction benchmark
    if results.get('reduction'):
        data = results['reduction']
        configs = [r['config'] for r in data]
        x = np.arange(len(configs))
        width = 0.25
        axes[1, 1].bar(x - width, [r['cpu_time_ms'] for r in data], width,
                        label='CPU', color=COLORS['cpu'], alpha=0.85)
        axes[1, 1].bar(x, [r['mps_time_ms'] for r in data], width,
                        label='MPS', color=COLORS['mps'], alpha=0.85)
        axes[1, 1].bar(x + width, [r['optimized_mps_time_ms'] for r in data], width,
                        label='Optimized', color=COLORS['optimized'], alpha=0.85)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(configs, rotation=45, ha='right', fontsize=7)
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].set_title('Reduction Operations')
        axes[1, 1].legend()

    path3 = os.path.join(output_dir, '03_compute_operations.png')
    fig.savefig(path3, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path3}")

    # ---- Chart 4: Summary Radar / Heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Performance Summary\n{suptitle_suffix}',
                 fontsize=15, fontweight='bold')
    plt.subplots_adjust(wspace=0.35, top=0.85, bottom=0.1)

    # Speedup heatmap
    op_names = []
    mps_speedups = []
    opt_speedups = []
    opt_vs_mps = []

    for key in ['matmul', 'attention', 'softmax', 'layernorm', 'elementwise', 'reduction', 'conv2d', 'transformer']:
        if key not in results or not results[key]:
            continue
        data = results[key]
        label_map = {
            'matmul': 'MatMul', 'attention': 'Attention', 'softmax': 'Softmax',
            'layernorm': 'LayerNorm', 'elementwise': 'Elementwise', 'reduction': 'Reduction',
            'conv2d': 'Conv2D', 'transformer': 'Transformer',
        }
        op_names.append(label_map.get(key, key))
        mps_speedups.append(np.mean([r.get('mps_speedup_vs_cpu', 1) or 1 for r in data]))
        opt_speedups.append(np.mean([r.get('optimized_speedup_vs_cpu', 1) or 1 for r in data]))
        opt_vs_mps.append(np.mean([r.get('optimized_speedup_vs_mps', 1) or 1 for r in data]))

    # Grouped bar chart - Speedup Summary
    ax = axes[0]
    x = np.arange(len(op_names))
    width = 0.3
    bars1 = ax.bar(x - width/2, mps_speedups, width, label='MPS vs CPU', color=COLORS['mps'], alpha=0.85)
    bars2 = ax.bar(x + width/2, opt_speedups, width, label='Optimized vs CPU', color=COLORS['optimized'], alpha=0.85)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(op_names, rotation=45, ha='right')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Speedup vs CPU Baseline')
    ax.legend()

    # Optimization gain (Optimized vs MPS)
    ax2 = axes[1]
    colors_bar = [COLORS['optimized'] if s > 1 else COLORS['mps'] for s in opt_vs_mps]
    bars = ax2.barh(op_names, opt_vs_mps, color=colors_bar, alpha=0.85, edgecolor='white')
    ax2.axvline(x=1, color='gray', linestyle=':', alpha=0.5, label='No improvement')
    ax2.set_xlabel('Speedup (x)')
    ax2.set_title('MetalNative Optimization Gain vs Standard MPS')

    for i, (bar, val) in enumerate(zip(bars, opt_vs_mps)):
        label = f'{val:.2f}x'
        ax2.annotate(label, xy=(val, bar.get_y() + bar.get_height()/2),
                     xytext=(5, 0), textcoords='offset points',
                     ha='left', va='center', fontsize=10, fontweight='bold')

    path4 = os.path.join(output_dir, '04_performance_summary.png')
    fig.savefig(path4, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path4}")

    print(f"\nAll visualizations saved to: {output_dir}/")
    return [path1, path2, path3, path4]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize MPS benchmark results')
    parser.add_argument('--input', type=str, required=True, help='Input JSON results file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for charts')
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.input), 'benchmark_charts'
    )

    print("Generating benchmark visualizations...")
    paths = generate_visualizations(results, output_dir)
    print(f"\nDone! Generated {len(paths)} charts.")


if __name__ == "__main__":
    main()
