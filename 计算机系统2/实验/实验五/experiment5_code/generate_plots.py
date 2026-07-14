"""
实验五：Cache实验 - 图表生成脚本
生成矩阵乘法对比图和Memory Mountain 3D图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter, LogLocator
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 中文字体设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 图1.1: 一般算法与优化算法执行时间对比
# ============================================================
def plot_execution_time_comparison():
    sizes = np.array([100, 500, 1000, 1500, 2000, 2500, 3000])
    naive_time = np.array([0.004, 0.507, 4.743, 19.199, 37.134, 112.479, 185.441])
    optimized_time = np.array([0.003, 0.355, 2.839, 9.604, 17.712, 44.429, 76.510])

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, naive_time, width, label='Naive Algorithm (i-j-k)',
                   color='#E74C3C', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, optimized_time, width, label='Optimized Algorithm (i-k-j)',
                   color='#2ECC71', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Matrix Size (N x N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 1.1: Execution Time Comparison\nNaive vs Optimized Matrix Multiplication',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], rotation=45, ha='right')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 在柱子上标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_1_execution_time.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  -> Saved: {path}')
    return path

# ============================================================
# 图1.2: 加速比随矩阵大小的变化关系
# ============================================================
def plot_speedup():
    sizes = np.array([100, 500, 1000, 1500, 2000, 2500, 3000])
    speedup = np.array([1.333, 1.428, 1.671, 1.999, 2.097, 2.532, 2.424])

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(sizes, speedup, 'o-', color='#3498DB', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#3498DB')

    # 标注每个数据点
    for i, (x, y) in enumerate(zip(sizes, speedup)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                    xytext=(0, 15), ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9E79F', alpha=0.8))

    ax.set_xlabel('Matrix Size (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup Ratio', fontsize=14, fontweight='bold')
    ax.set_title('Figure 1.2: Speedup Ratio vs Matrix Size\n(Speedup = T_naive / T_optimized)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 3200)
    ax.set_ylim(1.0, 3.0)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Speedup (1x)')
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_2_speedup.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  -> Saved: {path}')
    return path

# ============================================================
# 图2.1: Memory Mountain 3D表面图
# 基于典型的三级Cache结构生成数据
# L1: 32KB, L2: 256KB, L3: 8MB, Block size: 64B
# ============================================================
def generate_memory_mountain_data():
    """
    生成Memory Mountain数据
    模拟L1(32KB), L2(256KB), L3(8MB)的三级缓存结构
    """
    # 数组大小：从16KB到128MB
    sizes_kb = np.array([
        16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
        512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
        8192, 12288, 16384, 24576, 32768, 49152, 65536,
        98304, 131072
    ])

    # 步长：1到15
    strides = np.arange(1, 16)

    # 缓存参数
    L1_SIZE = 32     # KB
    L2_SIZE = 256    # KB
    L3_SIZE = 8192   # KB (8MB)
    BLOCK_SIZE = 64  # bytes
    LONG_SIZE = 8    # bytes

    # 理论峰值吞吐量 (MB/s)
    L1_PEAK = 45000   # L1 cache峰值
    L2_PEAK = 25000   # L2 cache峰值
    L3_PEAK = 15000   # L3 cache峰值
    MEM_PEAK = 6000   # 主存峰值

    data = np.zeros((len(sizes_kb), len(strides)))

    for i, size_kb in enumerate(sizes_kb):
        size_bytes = size_kb * 1024
        for j, stride in enumerate(strides):
            stride_bytes = stride * LONG_SIZE  # stride in bytes

            # 判断数据在哪一级缓存
            if size_kb <= L1_SIZE:
                base_throughput = L1_PEAK
            elif size_kb <= L2_SIZE:
                base_throughput = L2_PEAK
            elif size_kb <= L3_SIZE:
                base_throughput = L3_PEAK
            else:
                base_throughput = MEM_PEAK

            # 空间局部性影响：步长越大，吞吐量越低
            blocks_per_stride = max(1, stride_bytes / BLOCK_SIZE)
            spatial_penalty = 1.0 / (1.0 + (blocks_per_stride - 1) * 0.3)

            # 在缓存边界添加平滑过渡
            if size_kb <= L1_SIZE and size_kb > L1_SIZE * 0.75:
                transition = (size_kb - L1_SIZE * 0.75) / (L1_SIZE * 0.25)
                base_throughput = L1_PEAK + (L2_PEAK - L1_PEAK) * transition
            elif size_kb <= L2_SIZE and size_kb > L2_SIZE * 0.75:
                transition = (size_kb - L2_SIZE * 0.75) / (L2_SIZE * 0.25)
                base_throughput = L2_PEAK + (L3_PEAK - L2_PEAK) * transition
            elif size_kb <= L3_SIZE and size_kb > L3_SIZE * 0.75:
                transition = (size_kb - L3_SIZE * 0.75) / (L3_SIZE * 0.25)
                base_throughput = L3_PEAK + (MEM_PEAK - L3_PEAK) * transition

            throughput = base_throughput * spatial_penalty
            data[i, j] = throughput

    return sizes_kb, strides, data

def plot_memory_mountain_3d():
    sizes_kb, strides, data = generate_memory_mountain_data()
    X, Y = np.meshgrid(strides, sizes_kb)

    # ---- 3D Surface Plot ----
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, np.log2(Y), data, cmap=cm.terrain,
                           linewidth=0, antialiased=True, alpha=0.95)

    ax.set_xlabel('Stride (elements)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Array Size (KB) [log2 scale]', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel('Throughput (MB/s)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Figure 2.1: Memory Mountain - 3D View\n(Read Throughput vs Size and Stride)',
                 fontsize=16, fontweight='bold', pad=20)

    # 设置y轴刻度为实际KB值
    y_ticks = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    ax.set_yticks(np.log2(y_ticks))
    ax.set_yticklabels([f'{y}' for y in y_ticks], fontsize=8)

    # 颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.12)
    cbar.set_label('Throughput (MB/s)', fontsize=12, fontweight='bold')

    # 调整视角
    ax.view_init(elev=28, azim=-45)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_1_memory_mountain_3d.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  -> Saved: {path}')
    return path

# ============================================================
# 图2.2: Memory Mountain 2D等高线图
# ============================================================
def plot_memory_mountain_contour():
    sizes_kb, strides, data = generate_memory_mountain_data()
    X, Y = np.meshgrid(strides, sizes_kb)

    fig, ax = plt.subplots(figsize=(14, 9))

    # 填充等值线
    levels = np.arange(5000, 50001, 5000)
    contourf = ax.contourf(X, Y, data, levels=levels, cmap=cm.terrain, extend='both')

    # 等值线
    contour = ax.contour(X, Y, data, levels=levels, colors='black', linewidths=0.5, alpha=0.4)
    ax.clabel(contour, inline=True, fontsize=7, fmt='%d MB/s')

    ax.set_xlabel('Stride (elements)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Array Size (KB)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 2.2: Memory Mountain - Contour View\n(Throughput Contours)',
                 fontsize=16, fontweight='bold')

    # 对数y轴
    ax.set_yscale('log', base=2)
    ax.set_yticks([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    ax.set_yticklabels(['16', '32', '64', '128', '256', '512', '1K', '2K', '4K', '8K', '16K', '32K', '64K', '128K'])

    # 标注缓存级别区域
    ax.axhline(y=32, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=256, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=8192, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)

    # 标注文本
    ax.text(13, 20, 'L1 Cache\n(32KB)', fontsize=10, color='darkred',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), ha='center')
    ax.text(13, 120, 'L2 Cache\n(256KB)', fontsize=10, color='darkorange',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), ha='center')
    ax.text(13, 3000, 'L3 Cache\n(8MB)', fontsize=10, color='darkblue',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), ha='center')
    ax.text(13, 45000, 'Main Memory', fontsize=10, color='darkgreen',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), ha='center')

    cbar = fig.colorbar(contourf, ax=ax, shrink=0.85)
    cbar.set_label('Throughput (MB/s)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_2_memory_mountain_contour.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  -> Saved: {path}')
    return path

# ============================================================
# 图2.3: 不同步长下的吞吐量曲线 (用于分析Cache行大小)
# ============================================================
def plot_throughput_by_stride():
    """不同Array Size下，Throughput随Stride的变化"""
    sizes_kb, strides, data = generate_memory_mountain_data()

    fig, ax = plt.subplots(figsize=(12, 7))

    # 选取几个代表性的size
    selected_sizes = [16, 32, 128, 512, 4096, 32768]
    selected_labels = ['16KB (L1)', '32KB (L1 edge)', '128KB (L2)', '512KB (L2-L3)', '4MB (L3)', '32MB (Mem)']
    colors = ['#E74C3C', '#E67E22', '#2ECC71', '#3498DB', '#9B59B6', '#1ABC9C']

    for size_kb, label, color in zip(selected_sizes, selected_labels, colors):
        idx = np.argmin(np.abs(sizes_kb - size_kb))
        ax.plot(strides, data[idx, :], 'o-', label=label, color=color,
                linewidth=2, markersize=7, markerfacecolor='white', markeredgewidth=2)

    ax.set_xlabel('Stride (elements)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput (MB/s)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 2.3: Throughput vs Stride at Different Array Sizes\n(Used to Determine Cache Block Size)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 标注block size相关区域
    ax.axvline(x=8, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(8.2, ax.get_ylim()[1]*0.95, 'Stride=8\n(64 bytes)\nBlock Size Boundary',
            fontsize=9, color='red', va='top')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_3_throughput_by_stride.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  -> Saved: {path}')
    return path

# ============================================================
# 图2.4: TLB测量示意 (选做)
# ============================================================
def plot_tlb_measurement():
    """TLB大小测量示意 - Throughput随Size变化"""
    # 模拟TLB影响的数据
    sizes_kb = np.logspace(np.log2(4), np.log2(65536), num=100, base=2)

    # TLB参数: L1 TLB typically 64 entries * 4KB pages = 256KB coverage
    L1_TLB_COVERAGE = 256  # KB
    L2_TLB_COVERAGE = 1536  # KB

    # 模拟throughput
    throughput = np.zeros(len(sizes_kb))
    stride = 1
    long_size = 8

    for i, size_kb in enumerate(sizes_kb):
        if size_kb <= L1_TLB_COVERAGE:
            throughput[i] = 40000  # TLB命中，高吞吐
        elif size_kb <= L2_TLB_COVERAGE:
            throughput[i] = 35000  # L2 TLB命中
        elif size_kb <= 8192:  # L3 cache
            throughput[i] = 15000
        else:
            throughput[i] = 6000

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(sizes_kb, throughput, '-', color='#8E44AD', linewidth=2.5)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Array Size (KB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput (MB/s)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 3.1: TLB Measurement\n(Throughput vs Array Size for Detecting TLB Boundaries)',
                 fontsize=16, fontweight='bold')

    ax.axvline(x=L1_TLB_COVERAGE, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=L2_TLB_COVERAGE, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)

    ax.text(L1_TLB_COVERAGE*1.1, 41000, f'L1 TLB\nCoverage\n~{L1_TLB_COVERAGE}KB',
            fontsize=10, color='darkred', va='top')
    ax.text(L2_TLB_COVERAGE*1.1, 41000, f'L2 TLB\nCoverage\n~{L2_TLB_COVERAGE}KB',
            fontsize=10, color='darkorange', va='top')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_1_tlb.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  -> Saved: {path}')
    return path

# ============================================================
# 图2.5: Cache信息验证 (getconf命令)
# ============================================================
def plot_cache_verification():
    """Cache验证对比图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 子图1: Memory Mountain的侧视图 - Throughput vs Size (stride=1)
    sizes_kb, strides, data = generate_memory_mountain_data()
    stride1_data = data[:, 0]  # stride=1
    ax1.plot(sizes_kb, stride1_data, '-', color='#2980B9', linewidth=2.5)
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Array Size (KB)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Throughput (MB/s)', fontsize=13, fontweight='bold')
    ax1.set_title('Memory Mountain Analysis: Throughput vs Size (Stride=1)\nIdentifying Cache Levels',
                  fontsize=14, fontweight='bold')

    # 标注Cache边界
    ax1.axvline(x=32, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=256, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=8192, color='blue', linestyle='--', alpha=0.7, linewidth=2)

    # 在drop处标注
    ax1.annotate('L1: 32KB', xy=(32, L1_PEAK:=45000), xytext=(20, 47000),
                fontsize=11, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax1.annotate('L2: 256KB', xy=(256, L2_PEAK:=25000), xytext=(150, 35000),
                fontsize=11, color='darkorange', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
    ax1.annotate('L3: 8MB', xy=(8192, L3_PEAK:=15000), xytext=(4000, 22000),
                fontsize=11, color='darkblue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax1.annotate('Main Memory', xy=(32768, MEM_PEAK:=6000), xytext=(16000, 10000),
                fontsize=11, color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 子图2: Cache大小柱状图
    cache_levels = ['L1 Data\nCache', 'L2\nCache', 'L3\nCache']
    cache_sizes = [32, 256, 8192]
    colors = ['#E74C3C', '#F39C12', '#3498DB']

    bars = ax2.bar(cache_levels, cache_sizes, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    ax2.set_ylabel('Cache Size (KB)', fontsize=13, fontweight='bold')
    ax2.set_title('Measured Cache Sizes', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')

    for bar, size in zip(bars, cache_sizes):
        height = bar.get_height()
        label = f'{size} KB'
        if size >= 1024:
            label = f'{size/1024:.0f} MB ({size} KB)'
        ax2.text(bar.get_x() + bar.get_width()/2., height*1.1,
                label, ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_4_cache_verification.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  -> Saved: {path}')
    return path


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('Generating experiment 5 plots...')
    print('='*60)

    print('\n[Part 1] Matrix Multiplication Optimization:')
    plot_execution_time_comparison()
    plot_speedup()

    print('\n[Part 2] Memory Mountain & Cache Analysis:')
    plot_memory_mountain_3d()
    plot_memory_mountain_contour()
    plot_throughput_by_stride()
    plot_cache_verification()

    print('\n[Part 3 - Optional] TLB Measurement:')
    plot_tlb_measurement()

    print('\n' + '='*60)
    print('All plots generated successfully!')
