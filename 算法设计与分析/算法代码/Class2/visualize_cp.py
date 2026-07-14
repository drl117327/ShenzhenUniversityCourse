import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random
import os

def generate_points(n=20, seed=42):
    random.seed(seed)
    return [(random.uniform(5, 95), random.uniform(5, 95)) for _ in range(n)]

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

class CPVisualizer:
    def __init__(self, points):
        self.points = points
        self.px = sorted(points, key=lambda p: (p[0], p[1]))
        self.py = sorted(points, key=lambda p: (p[1], p[0]))
        self.states = []
        
        self.solve(self.px, self.py, 0, 100)
        
        if self.states:
            last_state = self.states[-1].copy()
            last_state['action'] = 'finished'
            for _ in range(10): # Hold final frame
                self.states.append(last_state)

    def solve(self, px, py, x_min, x_max, depth=0):
        n = len(px)
        if n <= 3:
            self.states.append({'action': 'base_case', 'points': px, 'bounds': (x_min, x_max)})
            best_d = float('inf')
            best_pair = None
            for i in range(n-1):
                for j in range(i+1, n):
                    d = dist(px[i], px[j])
                    self.states.append({'action': 'check', 'p1': px[i], 'p2': px[j], 'bounds': (x_min, x_max), 'best_pair': best_pair})
                    if d < best_d:
                        best_d = d
                        best_pair = (px[i], px[j])
                        self.states.append({'action': 'update_local', 'best_pair': best_pair, 'd': best_d, 'bounds': (x_min, x_max)})
            self.states.append({'action': 'return', 'best_pair': best_pair, 'bounds': (x_min, x_max)})
            return best_d, best_pair

        mid = n // 2
        x_mid = px[mid][0]
        
        self.states.append({'action': 'divide', 'x_mid': x_mid, 'bounds': (x_min, x_max)})
        
        lx, rx = px[:mid], px[mid:]
        lx_set = set(lx)
        rx_set = set(rx)
        ly = [p for p in py if p in lx_set]
        ry = [p for p in py if p in rx_set]
        
        dl, pair_l = self.solve(lx, ly, x_min, x_mid, depth+1)
        dr, pair_r = self.solve(rx, ry, x_mid, x_max, depth+1)
        
        d = dl
        best_pair = pair_l
        if dr < dl:
            d = dr
            best_pair = pair_r
            
        self.states.append({'action': 'merge_start', 'x_mid': x_mid, 'd': d, 'best_pair': best_pair, 'bounds': (x_min, x_max)})
        
        strip = [p for p in py if abs(p[0] - x_mid) < d]
        self.states.append({'action': 'strip', 'x_mid': x_mid, 'd': d, 'strip_points': strip, 'best_pair': best_pair, 'bounds': (x_min, x_max)})
        
        for i in range(len(strip)):
            for j in range(i+1, min(i+8, len(strip))):
                if strip[j][1] - strip[i][1] >= d:
                    break
                
                self.states.append({'action': 'check_strip', 'p1': strip[i], 'p2': strip[j], 'x_mid': x_mid, 'd': d, 'strip_points': strip, 'best_pair': best_pair, 'bounds': (x_min, x_max)})
                
                d_ij = dist(strip[i], strip[j])
                if d_ij < d:
                    d = d_ij
                    best_pair = (strip[i], strip[j])
                    self.states.append({'action': 'update_strip', 'x_mid': x_mid, 'd': d, 'strip_points': strip, 'best_pair': best_pair, 'bounds': (x_min, x_max)})

        self.states.append({'action': 'return', 'best_pair': best_pair, 'bounds': (x_min, x_max)})
        return d, best_pair

def animate_algorithm(points, save_frames=True, out_dir="output_frames"):
    viz = CPVisualizer(points)
    states = viz.states
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("Divide & Conquer: Closest Pair Execution", fontsize=16, fontweight='bold')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Draw all points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, color='black', s=30, zorder=2)
        
        state = states[frame]
        action = state['action']
        bounds = state.get('bounds', (0, 100))
        
        # Highlight active region
        ax.axvspan(bounds[0], bounds[1], color='#f0f0f0', zorder=1)
        
        text_info = f"Step {frame+1}/{len(states)}\n"
        text_info += f"Action: {action.replace('_', ' ').title()}\n"
        
        if 'x_mid' in state:
            ax.axvline(state['x_mid'], color='gray', linestyle='--', linewidth=1.5)
            
        if action == 'divide':
            text_info += f"Dividing at X = {state['x_mid']:.2f}"
            
        elif action == 'base_case':
            pts = state['points']
            pxs = [p[0] for p in pts]
            pys = [p[1] for p in pts]
            ax.scatter(pxs, pys, color='blue', s=70, zorder=3)
            text_info += "Base Case: Brute force checking (N<=3)"
            
        elif action in ('check', 'check_strip'):
            p1, p2 = state['p1'], state['p2']
            # Highlight checking points
            ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='red', s=80, zorder=4)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linestyle='--', zorder=4)
            text_info += f"Checking distance: {dist(p1, p2):.2f}"
            if 'strip_points' in state:
                d = state['d']
                x_mid = state['x_mid']
                ax.axvspan(x_mid - d, x_mid + d, color='lightblue', alpha=0.4, zorder=1)
            
        elif action in ('update_local', 'update_strip', 'return', 'merge_start', 'finished'):
            best_pair = state.get('best_pair')
            if best_pair:
                p1, p2 = best_pair
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green', linewidth=3, zorder=5)
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='green', s=80, zorder=6)
                text_info += f"Local shortest so far: {dist(p1, p2):.2f}"
            if action == 'finished':
                text_info = "ALGORITHM FINISHED!\n" + text_info
            elif action == 'merge_start':
                text_info += "\nMerging sub-problems"
            
        if action in ('strip', 'check_strip', 'update_strip'):
            x_mid = state['x_mid']
            d = state['d']
            # Draw the strip boundaries and region
            ax.axvspan(x_mid - d, x_mid + d, color='lightblue', alpha=0.3, zorder=1)
            ax.axvline(x_mid - d, color='blue', linestyle=':', alpha=0.5)
            ax.axvline(x_mid + d, color='blue', linestyle=':', alpha=0.5)
            
            strip_pts = state['strip_points']
            if strip_pts:
                s_xs = [p[0] for p in strip_pts]
                s_ys = [p[1] for p in strip_pts]
                ax.scatter(s_xs, s_ys, color='orange', edgecolors='black', s=60, zorder=3)
            text_info += f"\nChecking Strip (Width: 2*{d:.2f})"

        # Add Text Box
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, text_info, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', bbox=props)

    if save_frames:
        out_path = os.path.join(os.path.dirname(__file__), out_dir)
        print(f"开始保存 {len(states)} 帧图像到目录: {out_path} ...")
        os.makedirs(out_path, exist_ok=True)
        for i in range(len(states)):
            update(i)
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, f"frame_{i:03d}.png"), dpi=150)
            if (i + 1) % 10 == 0 or (i + 1) == len(states):
                print(f"已保存 {i+1}/{len(states)} 帧...")
        print("所有帧图像保存完成！")

    print("\n启动可视化动画窗口...")
    anim = FuncAnimation(fig, update, frames=len(states), interval=600, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting Closest Pair Visualization...")
    pts = generate_points(n=5, seed=42)
    # 调用时默认保存帧到 output_frames 文件夹中
    animate_algorithm(pts, save_frames=True, out_dir="output_frames")
    print("Visualization window closed.")


# /Users/a.16/Anaconda/anaconda3/bin/conda run -p /Users/a.16/Anaconda/anaconda3 python /Users/a.16/code/python/Homework/Class2/visualize_cp.py