#!/usr/bin/env python3
"""Generate performance charts for README.md"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# Output directory
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# Color palette
COLORS = {
    'primary': '#3B82F6',     # Blue
    'secondary': '#10B981',   # Green
    'accent': '#F59E0B',      # Amber
    'danger': '#EF4444',      # Red
    'purple': '#8B5CF6',      # Purple
    'dark': '#1E293B',        # Slate
    'gray': '#94A3B8',        # Gray
    'bg': '#0F172A',          # Dark background
    'card': '#1E293B',        # Card background
    'text': '#E2E8F0',        # Light text
    'grid': '#334155',        # Grid lines
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': COLORS['card'],
    'axes.edgecolor': COLORS['grid'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 12,
})


def chart_1_forward_latency():
    """Bar chart: Forward pass latency by model size."""
    models = ['Qwen2.5\n0.5B', 'Qwen2.5\n1.5B', 'SmolLM2\n1.7B', 'Qwen2.5\n3B', 'Qwen2.5\n7B']
    latency = [18.15, 39.39, 36.68, 67.79, 122.69]
    params = [0.5, 1.5, 1.7, 3.0, 7.0]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(models, latency, color=[COLORS['primary']] * 4 + [COLORS['purple']],
                  edgecolor='white', linewidth=0.5, width=0.6, zorder=3)

    # Add value labels
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold',
                fontsize=13, color=COLORS['text'])

    ax.set_ylabel('Forward Pass Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Forward Pass Latency — Modern LLMs on Apple M4 Max MPS',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 155)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.tick_params(axis='x', labelsize=11)

    # Add subtitle
    fig.text(0.5, 0.01, 'FP16 | seq_len=64 | PyTorch 2.10.0 | Real HuggingFace Pretrained Weights',
             ha='center', fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, 'forward_latency.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: forward_latency.png")


def chart_2_throughput():
    """Horizontal bar chart: Generation throughput."""
    models = ['Qwen2.5-7B', 'Qwen2.5-3B', 'SmolLM2-1.7B', 'Qwen2.5-1.5B', 'Qwen2.5-0.5B']
    tok_s = [19.1, 35.8, 56.2, 56.1, 72.6]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [COLORS['purple'], COLORS['accent'], COLORS['secondary'],
              COLORS['secondary'], COLORS['primary']]
    bars = ax.barh(models, tok_s, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.55, zorder=3)

    # Value labels
    for bar, val in zip(bars, tok_s):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} tok/s', ha='left', va='center', fontweight='bold',
                fontsize=13, color=COLORS['text'])

    ax.set_xlabel('Tokens per Second', fontsize=13, fontweight='bold')
    ax.set_title('Generation Throughput — 50 Tokens (FP16)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, 90)
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    ax.tick_params(axis='y', labelsize=12)

    fig.text(0.5, 0.01, 'Apple M4 Max 36GB | greedy decoding | min_new_tokens=50',
             ha='center', fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, 'throughput.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: throughput.png")


def chart_3_memory():
    """Bar chart: GPU memory usage."""
    models = ['Qwen2.5\n0.5B', 'Qwen2.5\n1.5B', 'SmolLM2\n1.7B', 'Qwen2.5\n3B', 'Qwen2.5\n7B']
    memory_gb = [0.92, 2.87, 3.19, 5.75, 14.19]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Color gradient based on memory
    bar_colors = [COLORS['secondary'], COLORS['secondary'], COLORS['primary'],
                  COLORS['accent'], COLORS['danger']]
    bars = ax.bar(models, memory_gb, color=bar_colors,
                  edgecolor='white', linewidth=0.5, width=0.6, zorder=3)

    # Value labels
    for bar, val in zip(bars, memory_gb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f} GB', ha='center', va='bottom', fontweight='bold',
                fontsize=13, color=COLORS['text'])

    # 36GB line
    ax.axhline(y=36, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
    ax.text(4.4, 36.5, 'M4 Max 36GB', fontsize=10, color=COLORS['danger'],
            ha='right', fontweight='bold')

    ax.set_ylabel('GPU Memory (GB)', fontsize=13, fontweight='bold')
    ax.set_title('GPU Memory Usage — Real Pretrained Models (FP16)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 40)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    fig.text(0.5, 0.01, 'Measured with torch.mps.current_allocated_memory() during inference',
             ha='center', fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, 'memory_usage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: memory_usage.png")


def chart_4_speedup():
    """Bar chart: MPS speedup vs CPU."""
    models = ['Qwen2.5\n0.5B', 'Qwen2.5\n1.5B', 'SmolLM2\n1.7B', 'Qwen2.5\n3B']
    speedup = [4.01, 4.08, 4.94, 4.72]
    cpu_ms = [72.83, 160.75, 181.31, 320.00]
    mps_ms = [18.15, 39.39, 36.68, 67.79]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(models))
    width = 0.35

    bars_cpu = ax.bar(x - width/2, cpu_ms, width, label='CPU (Float32)',
                      color=COLORS['danger'], edgecolor='white', linewidth=0.5, zorder=3, alpha=0.85)
    bars_mps = ax.bar(x + width/2, mps_ms, width, label='MPS (FP16)',
                      color=COLORS['primary'], edgecolor='white', linewidth=0.5, zorder=3)

    # Speedup annotations
    for i, (cpu, mps, sp) in enumerate(zip(cpu_ms, mps_ms, speedup)):
        ax.annotate(f'{sp:.1f}x faster',
                    xy=(i, max(cpu, mps) + 8), fontsize=12, fontweight='bold',
                    ha='center', color=COLORS['secondary'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'],
                              edgecolor=COLORS['secondary'], alpha=0.9))

    ax.set_ylabel('Forward Pass Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('MPS vs CPU — Forward Pass Performance',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.8,
              facecolor=COLORS['card'], edgecolor=COLORS['grid'])
    ax.set_ylim(0, 380)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    fig.text(0.5, 0.01, 'CPU: Apple M4 Max (Float32) | MPS: Metal GPU (FP16) | seq_len=64',
             ha='center', fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, 'speedup_vs_cpu.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: speedup_vs_cpu.png")


def chart_5_transformer_block():
    """Bar chart: Transformer block speedup (synthetic benchmark)."""
    models = ['GPT2-Med\nd=1024', 'GPT2-Large\nd=1280', 'Llama-7B\nd=4096', 'Kimi-K2.5\nd=7168']
    standard = [8.9, 13.5, 119.4, 363.5]
    optimized = [2.8, 3.9, 23.5, 62.2]
    speedup = [3.20, 3.49, 5.09, 5.84]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(models))
    width = 0.35

    bars_std = ax.bar(x - width/2, standard, width, label='Standard MPS',
                      color=COLORS['gray'], edgecolor='white', linewidth=0.5, zorder=3, alpha=0.7)
    bars_opt = ax.bar(x + width/2, optimized, width, label='MetalNative Optimized',
                      color=COLORS['secondary'], edgecolor='white', linewidth=0.5, zorder=3)

    # Speedup annotations
    for i, sp in enumerate(speedup):
        y_pos = max(standard[i], optimized[i]) + 8
        ax.annotate(f'{sp:.2f}x',
                    xy=(i, y_pos), fontsize=14, fontweight='bold',
                    ha='center', color=COLORS['accent'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'],
                              edgecolor=COLORS['accent'], alpha=0.9))

    ax.set_ylabel('Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Transformer Block — MetalNative vs Standard MPS',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.8,
              facecolor=COLORS['card'], edgecolor=COLORS['grid'])
    ax.set_ylim(0, 430)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    fig.text(0.5, 0.01, 'RMSNorm + Fused QKV + SDPA FlashAttention + SwiGLU FFN | Apple M4 Max',
             ha='center', fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, 'transformer_block.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: transformer_block.png")


def chart_6_flash_attention_memory():
    """Line chart: FlashAttention memory savings by sequence length."""
    seq_lens = [256, 512, 1024, 2048]
    naive_mb = [128.0, 512.0, 2048.0, 8192.0]
    flash_mb = [4.0, 16.0, 64.0, 256.0]
    savings = [96.9, 96.9, 96.9, 96.9]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Memory bars
    x = np.arange(len(seq_lens))
    width = 0.35
    bars_naive = ax1.bar(x - width/2, naive_mb, width, label='Naive Attention',
                         color=COLORS['danger'], edgecolor='white', linewidth=0.5, zorder=3, alpha=0.7)
    bars_flash = ax1.bar(x + width/2, flash_mb, width, label='FlashAttention',
                         color=COLORS['secondary'], edgecolor='white', linewidth=0.5, zorder=3)

    ax1.set_ylabel('Attention Memory (MB)', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'seq={s}' for s in seq_lens], fontsize=11)
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.8,
               facecolor=COLORS['card'], edgecolor=COLORS['grid'])

    # Savings annotation
    for i, (n, f) in enumerate(zip(naive_mb, flash_mb)):
        reduction = (1 - f/n) * 100
        ax1.annotate(f'-{reduction:.0f}%',
                     xy=(i, n), xytext=(i + 0.15, n * 1.3),
                     fontsize=11, fontweight='bold', color=COLORS['accent'],
                     ha='center')

    ax1.set_title('FlashAttention Memory Savings — Kimi-K2.5 (64 heads, d=112)',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    fig.text(0.5, 0.01, 'Memory = n_heads × seq_len² × sizeof(float) for naive | O(seq_len) for Flash',
             ha='center', fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, 'flash_attention_memory.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: flash_attention_memory.png")


def chart_7_scaling():
    """Line chart: Forward latency scaling with model size."""
    params = [0.5, 1.5, 1.7, 3.0, 7.0]
    latency = [18.15, 39.39, 36.68, 67.79, 122.69]
    memory = [0.92, 2.87, 3.19, 5.75, 14.19]
    labels = ['Qwen2.5-0.5B', 'Qwen2.5-1.5B', 'SmolLM2-1.7B', 'Qwen2.5-3B', 'Qwen2.5-7B']

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Latency line
    line1 = ax1.plot(params, latency, 'o-', color=COLORS['primary'], linewidth=2.5,
                     markersize=10, label='Forward Latency (ms)', zorder=4)
    ax1.set_xlabel('Model Parameters (Billions)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Forward Latency (ms)', fontsize=13, fontweight='bold', color=COLORS['primary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])

    # Memory line (secondary axis)
    ax2 = ax1.twinx()
    line2 = ax2.plot(params, memory, 's--', color=COLORS['accent'], linewidth=2.5,
                     markersize=10, label='GPU Memory (GB)', zorder=4)
    ax2.set_ylabel('GPU Memory (GB)', fontsize=13, fontweight='bold', color=COLORS['accent'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax2.spines['right'].set_color(COLORS['accent'])

    # Labels
    for p, l, lab in zip(params, latency, labels):
        ax1.annotate(lab, xy=(p, l), xytext=(0, 12), textcoords='offset points',
                     fontsize=9, ha='center', color=COLORS['text'], alpha=0.8)

    # Combined legend
    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, fontsize=11, loc='upper left', framealpha=0.8,
               facecolor=COLORS['card'], edgecolor=COLORS['grid'])

    ax1.set_title('Performance Scaling — Latency & Memory vs Model Size',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.grid(axis='both', linestyle='--', alpha=0.3, zorder=0)

    fig.text(0.5, 0.01, 'FP16 | seq_len=64 | Apple M4 Max 36GB Unified Memory',
             ha='center', fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, 'scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: scaling.png")


if __name__ == "__main__":
    print("Generating performance charts...")
    chart_1_forward_latency()
    chart_2_throughput()
    chart_3_memory()
    chart_4_speedup()
    chart_5_transformer_block()
    chart_6_flash_attention_memory()
    chart_7_scaling()
    print(f"\nAll charts saved to: {OUT_DIR}/")
