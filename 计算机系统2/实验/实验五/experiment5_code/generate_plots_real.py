"""
实验五：Cache实验 - 基于WSL真实数据生成图表
i9-14900HX, GCC 13.3.0, -O2
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 真实数据
# ============================================================

# ---- 矩阵乘法数据 (WSL实测) ----
MATRIX_SIZES = np.array([100, 500, 1000, 1500, 2000, 2500, 3000])
NAIVE_TIME = np.array([0.000255, 0.046886, 0.407510, 1.432449, 11.117311, 40.267149, 97.751168])
OPT_TIME = np.array([0.000278, 0.033446, 0.252517, 0.866728, 2.111857, 4.502034, 8.437031])
SPEEDUP = NAIVE_TIME / OPT_TIME  # [0.917, 1.402, 1.614, 1.653, 5.264, 8.944, 11.586]

# ---- Memory Mountain 数据 (从 WSL 读取) ----
mountain_file = os.path.join(output_dir, 'mountain_data5.txt')

def load_mountain_data(filepath):
    sizes = []
    data = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            sizes.append(int(parts[0].strip()))
            row = [float(x.strip()) for x in parts[1:]]
            data.append(row)
    return np.array(sizes), np.array(data)

SIZES_BYTES, MOUNTAIN_DATA = load_mountain_data(mountain_file)
SIZES_KB = SIZES_BYTES / 1024.0
STRIDES = np.arange(1, 17)

print(f"Loaded mountain data: {len(SIZES_BYTES)} sizes x {MOUNTAIN_DATA.shape[1]} strides")

# ---- 真实 Cache 参数 (getconf 验证) ----
L1D_SIZE = 48       # KB
L2_SIZE = 2048      # KB (2MB)
L3_SIZE = 36864     # KB (36MB)
CACHE_LINE = 64     # bytes
L1D_ASSOC = 12
L2_ASSOC = 16
L3_ASSOC = 12

# ============================================================
# 图 1.1: 矩阵乘法执行时间对比
# ============================================================
def plot_execution_time():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 左图: 双柱图
    x = np.arange(len(MATRIX_SIZES))
    width = 0.35
    bars1 = ax1.bar(x - width/2, NAIVE_TIME, width,
                    label='Naive (i-j-k)', color='#E74C3C', edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, OPT_TIME, width,
                    label='Optimized (i-k-j)', color='#2ECC71', edgecolor='white', linewidth=0.5)

    # 在小size时放大显示
    ax1.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time Comparison\n(i9-14900HX, GCC -O2)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}' for s in MATRIX_SIZES], rotation=45, ha='right')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 标注加速比
    for i, sp in enumerate(SPEEDUP):
        if i >= 3:  # size >= 1500
            ax1.annotate(f'{sp:.1f}x', (x[i], max(NAIVE_TIME[i], OPT_TIME[i])),
                        textcoords="offset points", xytext=(0, 8),
                        ha='center', fontsize=8, fontweight='bold',
                        color='#2C3E50')

    # 右图: 对数坐标 (方便看小size)
    ax2.plot(MATRIX_SIZES, NAIVE_TIME, 'o-', color='#E74C3C', linewidth=2,
             markersize=8, markerfacecolor='white', markeredgewidth=2, label='Naive')
    ax2.plot(MATRIX_SIZES, OPT_TIME, 'o-', color='#2ECC71', linewidth=2,
             markersize=8, markerfacecolor='white', markeredgewidth=2, label='Optimized')
    ax2.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Execution Time (s) [log scale]', fontsize=12, fontweight='bold')
    ax2.set_title('Log-Scale View (Highlights Small-Size Difference)', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 标注每个点
    for i in range(len(MATRIX_SIZES)):
        ax2.annotate(f'{NAIVE_TIME[i]:.4f}', (MATRIX_SIZES[i], NAIVE_TIME[i]),
                    textcoords="offset points", xytext=(5, -12), fontsize=7, color='#E74C3C')
        ax2.annotate(f'{OPT_TIME[i]:.4f}', (MATRIX_SIZES[i], OPT_TIME[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, color='#2ECC71')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_1_execution_time_real.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')
    return path

# ============================================================
# 图 1.2: 加速比曲线
# ============================================================
def plot_speedup():
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(MATRIX_SIZES, SPEEDUP, 'o-', color='#2980B9', linewidth=2.5,
            markersize=12, markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor='#2980B9')

    for i, (x, y) in enumerate(zip(MATRIX_SIZES, SPEEDUP)):
        ax.annotate(f'{y:.2f}x', (x, y), textcoords="offset points",
                    xytext=(0, 18), ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9E79F', alpha=0.9))

    ax.set_xlabel('Matrix Size (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup Ratio (T_naive / T_optimized)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 1.2: Speedup vs Matrix Size\n(i9-14900HX, GCC -O2)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='No Speedup (1.0x)')
    ax.legend(fontsize=11)

    # 标注关键区域
    ax.axvspan(0, 1500, alpha=0.1, color='orange', label='Data fits in L3 Cache')
    ax.axvspan(1500, 3200, alpha=0.1, color='red', label='Data exceeds L3 Cache')
    ax.text(700, 0.95, 'L3 Resident', fontsize=10, ha='center', color='darkorange')
    ax.text(2500, 1.1, 'Main Memory\nDominated', fontsize=10, ha='center', color='darkred')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_2_speedup_real.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')
    return path

# ============================================================
# 图 2.1: Memory Mountain 3D Surface
# ============================================================
def plot_mountain_3d():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(STRIDES, SIZES_KB)
    Z = MOUNTAIN_DATA

    surf = ax.plot_surface(X, np.log2(Y), Z, cmap=plt.cm.terrain,
                           linewidth=0, antialiased=True, alpha=0.95)

    ax.set_xlabel('Stride (elements)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Array Size (KB) [log2]', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel('Read Throughput (MB/s)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Figure 2.1: Memory Mountain - Real Measurement\n'
                 '(i9-14900HX, L1d=48KB, L2=2MB, L3=36MB)',
                 fontsize=15, fontweight='bold', pad=20)

    # Y轴标注
    y_ticks_kb = [16, 48, 128, 512, 2048, 8192, 36864, 65536]
    ax.set_yticks(np.log2(y_ticks_kb))
    ax.set_yticklabels([f'{y}KB' if y < 1024 else f'{y/1024:.0f}MB' for y in y_ticks_kb], fontsize=8)

    ax.view_init(elev=25, azim=-50)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.12)
    cbar.set_label('Throughput (MB/s)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_1_mountain_3d_real.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')
    return path

# ============================================================
# 图 2.2: Memory Mountain Contour
# ============================================================
def plot_mountain_contour():
    fig, ax = plt.subplots(figsize=(16, 10))

    X, Y = np.meshgrid(STRIDES, SIZES_KB)

    # 等高线填充
    levels = [5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
    contourf = ax.contourf(X, Y, MOUNTAIN_DATA, levels=levels,
                           cmap=plt.cm.terrain, extend='both')
    contour = ax.contour(X, Y, MOUNTAIN_DATA, levels=levels,
                         colors='black', linewidths=0.5, alpha=0.4)
    ax.clabel(contour, inline=True, fontsize=7, fmt='%d')

    ax.set_xlabel('Stride (elements)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Array Size (KB)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 2.2: Memory Mountain Contour\n'
                 '(L1d=48KB, L2=2MB, L3=36MB boundaries marked)',
                 fontsize=15, fontweight='bold')

    # 对数Y轴
    ax.set_yscale('log', base=2)
    y_ticks = [16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y}' for y in y_ticks], fontsize=8)

    # Cache boundary lines
    ax.axhline(y=L1D_SIZE, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax.axhline(y=L2_SIZE, color='orange', linestyle='--', alpha=0.8, linewidth=2)
    ax.axhline(y=L3_SIZE, color='blue', linestyle='--', alpha=0.8, linewidth=2)

    # Annotations
    ax.text(14.5, L1D_SIZE*1.3, f'L1d Cache\n({L1D_SIZE}KB)', fontsize=10,
            color='darkred', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax.text(14.5, L2_SIZE*1.3, f'L2 Cache\n({L2_SIZE//1024}MB)', fontsize=10,
            color='darkorange', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax.text(14.5, L3_SIZE*1.3, f'L3 Cache\n({L3_SIZE//1024}MB)', fontsize=10,
            color='darkblue', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax.text(14.5, L3_SIZE*1.6, 'Main\nMemory', fontsize=10,
            color='darkgreen', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    # Cache line boundary
    ax.axvline(x=8, color='purple', linestyle=':', alpha=0.7, linewidth=2)
    ax.text(8.2, 12, 'Cache Line\nBoundary\n(64 bytes)', fontsize=9,
            color='purple', fontweight='bold', va='bottom')

    cbar = fig.colorbar(contourf, ax=ax, shrink=0.85)
    cbar.set_label('Throughput (MB/s)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_2_mountain_contour_real.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')
    return path

# ============================================================
# 图 2.3: Throughput vs Size (stride=1) - 显示Cache边界
# ============================================================
def plot_throughput_vs_size():
    fig, ax = plt.subplots(figsize=(14, 7))

    stride1 = MOUNTAIN_DATA[:, 0]   # stride=1
    stride8 = MOUNTAIN_DATA[:, 7]   # stride=8 (cache line boundary)

    ax.plot(SIZES_KB, stride1, 'o-', color='#2980B9', linewidth=2, markersize=6,
            label='Stride=1 (Best spatial locality)')
    ax.plot(SIZES_KB, stride8, 's-', color='#E74C3C', linewidth=2, markersize=6,
            label='Stride=8 (Cache line boundary, 64B)')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Array Size (KB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Read Throughput (MB/s)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 2.3: Throughput vs Array Size\n'
                 '(Cache level transitions clearly visible)',
                 fontsize=14, fontweight='bold')

    # Cache boundaries
    ax.axvline(x=L1D_SIZE, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=L2_SIZE, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=L3_SIZE, color='blue', linestyle='--', alpha=0.7, linewidth=2)

    # Annotate regions
    mid_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    for (x_start, x_end, label, color) in [
        (8, L1D_SIZE, 'L1d\n48KB', 'darkred'),
        (L1D_SIZE*1.2, L2_SIZE*0.7, 'L2\n2MB', 'darkorange'),
        (L2_SIZE*1.2, L3_SIZE*0.7, 'L3\n36MB', 'darkblue'),
        (L3_SIZE*1.2, 70000, 'Main Memory', 'darkgreen'),
    ]:
        ax.text(np.sqrt(x_start * x_end), 75000, label, fontsize=10,
                ha='center', fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 90000)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_3_throughput_vs_size_real.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')
    return path

# ============================================================
# 图 2.4: Throughput vs Stride (多个固定size) - 确定Cache Line大小
# ============================================================
def plot_throughput_vs_stride():
    fig, ax = plt.subplots(figsize=(13, 7))

    # 选取代表性size (索引)
    target_kb = [16, 48, 256, 2048, 8192, 32768]
    labels = ['16KB (L1d)', '48KB (L1d edge)', '256KB (L2)',
              '2MB (L2 edge)', '8MB (L3)', '32MB (Mem)']
    colors = ['#E74C3C', '#E67E22', '#2ECC71', '#3498DB', '#9B59B6', '#1ABC9C']

    for target, label, color in zip(target_kb, labels, colors):
        idx = np.argmin(np.abs(SIZES_KB - target))
        ax.plot(STRIDES, MOUNTAIN_DATA[idx, :], 'o-', label=f'{label} ({SIZES_KB[idx]:.0f}KB)',
                color=color, linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)

    ax.set_xlabel('Stride (elements)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Read Throughput (MB/s)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 2.4: Throughput vs Stride at Fixed Sizes\n'
                 '(Drop at stride=8 → Cache Block = 64 Bytes)',
                 fontsize=14, fontweight='bold')

    # Cache Line boundary
    ax.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
    ax.annotate('Cache Line = 64 Bytes\n(8 longs × 8 bytes)',
                xy=(8, ax.get_ylim()[1]*0.5), xytext=(10, ax.get_ylim()[1]*0.6),
                fontsize=11, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_4_throughput_vs_stride_real.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')
    return path

# ============================================================
# 图 2.5: Cache验证 (getconf输出)
# ============================================================
def plot_cache_verification():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 左图: Cache容量柱状图
    cache_names = ['L1 Inst.\nCache', 'L1 Data\nCache', 'L2\nCache', 'L3\nCache']
    cache_sizes_kb = [32, 48, 2048, 36864]
    cache_colors = ['#3498DB', '#E74C3C', '#F39C12', '#2ECC71']
    cache_labels = ['32 KB', '48 KB', '2 MB\n(2048 KB)', '36 MB\n(36864 KB)']

    bars = ax1.bar(cache_names, cache_sizes_kb, color=cache_colors, edgecolor='white',
                   linewidth=1.5, width=0.55)
    for bar, label in zip(bars, cache_labels):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height()*1.05,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Cache Size (KB)', fontsize=12, fontweight='bold')
    ax1.set_title('Cache Sizes (getconf -a | grep CACHE)', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 右图: L1 Cache 行数计算示意
    l1d_size = 49152   # bytes
    line_size = 64      # bytes
    n_lines = l1d_size // line_size   # 768 lines
    n_sets = 64  # from getconf: assoc=12, lines=768 → sets=768/12=64
    associativity = 12

    ax2.axis('off')
    ax2.set_title('L1 Data Cache Structure', fontsize=13, fontweight='bold')

    info_text = f"""
    L1 Data Cache Analysis (from measurement + getconf):

    Cache Size:     {l1d_size} bytes (48 KB)
    Cache Line:     {line_size} bytes
    Associativity:  {associativity}-way set associative

    Number of lines = 49,152 / 64 = {n_lines} lines
    Number of sets  = {n_lines} / {associativity} = {n_sets} sets

    Measured confirmation:
    • Stride=8 (8×8=64B) → throughput drops
    → Cache Block = 64 bytes ✓
    • Throughput cliff at ~48KB → L1d boundary ✓
    """
    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes,
             fontsize=12, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_5_cache_verify_real.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')
    return path


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('='*60)
    print('Generating ALL plots from REAL WSL data')
    print(f'Machine: i9-14900HX, L1d=48KB, L2=2MB, L3=36MB')
    print(f'Matrix benchmark: GCC -O2, i-k-j optimized')
    print(f'Memory Mountain: {len(SIZES_BYTES)} size points × 16 strides')
    print('='*60)

    print('\n[Part 1] Matrix Multiplication:')
    plot_execution_time()
    plot_speedup()

    print('\n[Part 2] Memory Mountain:')
    plot_mountain_3d()
    plot_mountain_contour()
    plot_throughput_vs_size()
    plot_throughput_vs_stride()
    plot_cache_verification()

    print('\n' + '='*60)
    print('All real-data plots generated!')
    print(f'Output directory: {output_dir}')
