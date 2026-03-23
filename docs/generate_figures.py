#!/usr/bin/env python3
"""Generate figures for the MLX Layer-Streaming paper."""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DOCS_DIR = os.path.dirname(__file__)


def fig1_memory_reduction():
    """Figure 1: Memory reduction with layer-streaming."""
    models = ['0.8B-8bit', '4B-4bit', '9B-8bit', '27B-4bit']
    normal_mem = [0.74, 2.34, 8.86, 14.09]
    stream_mem = [0.29, 0.47, 2.29, 1.70]
    reduction = [1 - s/n for s, n in zip(stream_mem, normal_mem)]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, normal_mem, width, label='Full Resident', color='#4A90D9', edgecolor='white')
    bars2 = ax1.bar(x + width/2, stream_mem, width, label='Layer-Streaming', color='#E8744F', edgecolor='white')

    ax1.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Qwen3.5\n{m}' for m in models], fontsize=10)
    ax1.legend(fontsize=11)
    ax1.axhline(y=8, color='red', linestyle='--', alpha=0.5, label='8GB Device Limit')
    ax1.text(3.5, 8.3, '8GB Device Limit', color='red', alpha=0.7, fontsize=9)

    # Add reduction labels
    for i, r in enumerate(reduction):
        ax1.text(i + width/2, stream_mem[i] + 0.3, f'-{r*100:.0f}%',
                ha='center', fontsize=9, fontweight='bold', color='#E8744F')

    ax1.set_ylim(0, 16)
    plt.title('Memory Reduction with Layer-Streaming Offloading', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'fig1_memory_reduction.png'), dpi=200)
    plt.close()
    print('Generated fig1_memory_reduction.png')


def fig2_tps_vs_residency():
    """Figure 2: TPS vs resident layer percentage."""
    data = json.load(open(os.path.join(RESULTS_DIR, 'streaming_benchmark.json')))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, model_data in enumerate([d for d in data if d['model'] in ['Qwen3.5-9B-8bit', 'Qwen3.5-27B-4bit']]):
        ax = axes[idx]
        model = model_data['model']
        configs = model_data['configs']

        pcts = []
        tps_vals = []
        mem_vals = []

        for c in configs:
            if c['mode'] == 'full_resident':
                pcts.append(100)
            elif 'stream_0pct' in c['mode']:
                pcts.append(0)
            elif '25pct' in c['mode']:
                pcts.append(25)
            elif '50pct' in c['mode']:
                pcts.append(50)
            elif '75pct' in c['mode']:
                pcts.append(75)
            tps_vals.append(c['gen_tps'])
            mem_vals.append(c['peak_mem_gb'])

        color1 = '#4A90D9'
        color2 = '#E8744F'

        ax.plot(pcts, tps_vals, 'o-', color=color1, linewidth=2, markersize=8, label='TPS')
        ax.set_xlabel('Resident Layers (%)', fontsize=11)
        ax.set_ylabel('Decode TPS', color=color1, fontsize=11)
        ax.tick_params(axis='y', labelcolor=color1)

        ax2 = ax.twinx()
        ax2.plot(pcts, mem_vals, 's--', color=color2, linewidth=2, markersize=7, label='Memory')
        ax2.set_ylabel('Peak Memory (GB)', color=color2, fontsize=11)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axhline(y=8, color='red', linestyle=':', alpha=0.4)

        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 25, 50, 75, 100])

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.suptitle('TPS vs Memory Tradeoff with Partial Layer Residency', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'fig2_tps_vs_residency.png'), dpi=200)
    plt.close()
    print('Generated fig2_tps_vs_residency.png')


def fig3_device_bandwidth():
    """Figure 3: Device TPS scaling with bandwidth."""
    # Device data
    models = ['0.8B-8bit', '2B-8bit', '4B-4bit', '4B-6bit']
    ipad_tps = [103.2, 43.8, 35.5, 25.2]
    iphone_tps = [53.5, 23.0, 18.4, 13.2]
    mac_tps = [302.7, 203.0, 148.1, 117.1]
    weights_gb = [0.80, 1.93, 2.34, 3.32]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: TPS comparison across devices
    x = np.arange(len(models))
    width = 0.25
    ax1.bar(x - width, mac_tps, width, label='Mac Studio (800 GB/s)', color='#2D5F8A')
    ax1.bar(x, ipad_tps, width, label='iPad Air M3 (100 GB/s)', color='#4A90D9')
    ax1.bar(x + width, iphone_tps, width, label='iPhone 15 Pro Max (50 GB/s)', color='#93C5FD')

    ax1.set_ylabel('Decode TPS', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Qwen3.5\n{m}' for m in models], fontsize=9)
    ax1.legend(fontsize=8.5, loc='upper right')
    ax1.set_title('Decode TPS Across Devices', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')

    # Right: iPad/iPhone ratio
    ratios = [i/p for i, p in zip(ipad_tps, iphone_tps)]
    ax2.bar(range(len(models)), ratios, color='#4A90D9', edgecolor='white')
    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Theoretical (2.0x)')
    ax2.set_ylabel('iPad / iPhone TPS Ratio', fontsize=11)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([f'Qwen3.5\n{m}' for m in models], fontsize=9)
    ax2.set_ylim(0, 2.5)
    ax2.legend(fontsize=10)
    ax2.set_title('Bandwidth Scaling Verification', fontsize=12, fontweight='bold')

    for i, r in enumerate(ratios):
        ax2.text(i, r + 0.05, f'{r:.2f}x', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'fig3_device_bandwidth.png'), dpi=200)
    plt.close()
    print('Generated fig3_device_bandwidth.png')


def fig4_ppl_memory_frontier():
    """Figure 4: PPL vs Memory efficiency frontier."""
    data = json.load(open(os.path.join(RESULTS_DIR, 'phase0_baseline_all.json')))

    fig, ax = plt.subplots(figsize=(8, 5))

    models_2b = [(d['speed']['avg_mem'], d['ppl'], d['model']) for d in data if '2B' in d['model']]
    models_4b = [(d['speed']['avg_mem'], d['ppl'], d['model']) for d in data if '4B' in d['model']]
    models_9b = [(d['speed']['avg_mem'], d['ppl'], d['model']) for d in data if '9B' in d['model']]
    models_other = [(d['speed']['avg_mem'], d['ppl'], d['model']) for d in data
                     if '2B' not in d['model'] and '4B' not in d['model'] and '9B' not in d['model']]

    for group, color, label in [
        (models_other, '#93C5FD', '0.8B/27B'),
        (models_2b, '#4A90D9', '2B'),
        (models_4b, '#E8744F', '4B'),
        (models_9b, '#2D8A4E', '9B'),
    ]:
        mems, ppls, names = zip(*group) if group else ([], [], [])
        ax.scatter(mems, ppls, c=color, s=100, label=label, zorder=5, edgecolors='white')
        for m, p, n in zip(mems, ppls, names):
            short = n.replace('Qwen3.5-', '')
            ax.annotate(short, (m, p), textcoords="offset points", xytext=(8, 4), fontsize=7.5)

    ax.axvline(x=5, color='red', linestyle='--', alpha=0.4)
    ax.text(5.2, 6.0, 'iOS ~5GB\navailable', color='red', alpha=0.6, fontsize=8)
    ax.axvline(x=8, color='orange', linestyle='--', alpha=0.4)
    ax.text(8.2, 6.0, '8GB\nphysical', color='orange', alpha=0.6, fontsize=8)

    ax.set_xlabel('Peak Memory (GB)', fontsize=11)
    ax.set_ylabel('Perplexity (lower = better)', fontsize=11)
    ax.set_title('PPL vs Memory: Model Selection for 8GB Devices', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'fig4_ppl_memory_frontier.png'), dpi=200)
    plt.close()
    print('Generated fig4_ppl_memory_frontier.png')


if __name__ == '__main__':
    fig1_memory_reduction()
    fig2_tps_vs_residency()
    fig3_device_bandwidth()
    fig4_ppl_memory_frontier()
    print('\nAll figures generated in docs/')
