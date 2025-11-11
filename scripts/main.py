import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# ==============================
# Configuration
# ==============================
DATA_ROOT = "data_clean"
IMG_ROOT = "img"
FONT_SIZE = 18

os.makedirs(IMG_ROOT, exist_ok=True)

pattern = re.compile(r"Scene_(\d+)_(opengl|vulkan)_.+_(frames|system)\.csv")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.titlesize': FONT_SIZE + 4,
    'axes.titleweight': 'bold',
    'axes.labelsize': FONT_SIZE + 2,
    'axes.labelweight': 'regular',
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE + 6,
    'figure.titleweight': 'bold',
})

# ==============================
# Data Collection (UNCHANGED)
# ==============================
print("=== Loading Data ===")
frame_rows = []
system_rows = []

for pc in sorted(os.listdir(DATA_ROOT)):
    pc_path = os.path.join(DATA_ROOT, pc)
    if not os.path.isdir(pc_path):
        continue

    for file in glob.glob(os.path.join(pc_path, "*.csv")):
        m = pattern.search(os.path.basename(file))
        if not m:
            continue

        if os.path.getsize(file) == 0:
            print(f"Skipping empty file: {file}")
            continue

        scene, api, kind = m.groups()
        scene = int(scene)

        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            print(f"Skipping unreadable CSV: {file}")
            continue

        df["PC"] = pc
        df["Scene"] = scene
        df["API"] = api

        if kind == "frames":
            if "Frame" not in df.columns or "FrameTime(ms)" not in df.columns:
                print(f"Skipping malformed frame file: {file}")
                continue
            frame_rows.append(df)
        elif kind == "system":
            system_rows.append(df)

frames = pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame()
systems = pd.concat(system_rows, ignore_index=True) if system_rows else pd.DataFrame()

print(f"Loaded {len(frames)} frame records from {len(frames['PC'].unique()) if not frames.empty else 0} PCs")

# ==============================
# Summary Statistics (UNCHANGED)
# ==============================
print("\n=== Generating Summary Statistics ===")
summary_data = []

if not frames.empty and "FPS" in frames.columns:
    for pc in frames["PC"].unique():
        for scene in frames["Scene"].unique():
            for api in frames["API"].unique():
                subset = frames[
                    (frames["PC"] == pc) &
                    (frames["Scene"] == scene) &
                    (frames["API"] == api)
                ]
                if subset.empty:
                    continue

                record = {
                    "PC": pc,
                    "Scene": scene,
                    "API": api,
                    "Average FPS": subset["FPS"].mean(),
                    "Min FPS": subset["FPS"].min(),
                    "Max FPS": subset["FPS"].max(),
                    "Median FPS": subset["FPS"].median(),
                    "Std FPS": subset["FPS"].std(),
                    "P1 FPS": subset["FPS"].quantile(0.01),
                    "P99 FPS": subset["FPS"].quantile(0.99),
                    "Average FrameTime": subset["FrameTime(ms)"].mean(),
                    "P99 FrameTime": subset["FrameTime(ms)"].quantile(0.99),
                }

                render_passes = [
                    'GeometryPass(ms)', 'LightingPass(ms)', 'GizmoPass(ms)',
                    'ParticlePass(ms)', 'ImGuiPass(ms)'
                ]
                for pass_name in render_passes:
                    if pass_name in subset.columns:
                        clean_name = pass_name.replace('(ms)', '').replace('Pass', ' Pass')
                        record[clean_name] = subset[pass_name].mean()

                if 'VRAM(MB)' in subset.columns:
                    record['Average VRAM'] = subset['VRAM(MB)'].mean()
                if 'CPUUtil(%)' in subset.columns:
                    record['Average CPU Util'] = subset['CPUUtil(%)'].mean()

                summary_data.append(record)

summary = pd.DataFrame(summary_data)
print(f"Summary generated: {len(summary)} configurations")

# ==============================
# System Info (UNCHANGED)
# ==============================
print("\n=== System Configurations ===")
pc_gpu_map = {}
if not systems.empty:
    for pc in systems["PC"].unique():
        pc_sys = systems[systems["PC"] == pc].iloc[0]
        cpu = pc_sys.get('CPU Model', 'N/A')
        gpu = pc_sys.get('GPU Model', 'N/A')
        pc_gpu_map[pc] = gpu
        print(f"\n{pc}:")
        print(f"  CPU: {cpu}")
        print(f"  GPU: {gpu}")

# ==============================
# Utility Functions (UNCHANGED)
# ==============================
def save_plot(path, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")

# =====================================================
# 1. OVERALL API COMPARISON - Average FPS across all PCs
# =====================================================
print("\n=== Generating Visualizations ===")

if "Average FPS" in summary.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    scenes = sorted(summary["Scene"].unique())
    x = np.arange(len(scenes))
    width = 0.35

    opengl_fps = [summary[(summary["Scene"] == s) & (summary["API"] == "opengl")]["Average FPS"].mean() for s in scenes]
    vulkan_fps = [summary[(summary["Scene"] == s) & (summary["API"] == "vulkan")]["Average FPS"].mean() for s in scenes]

    bars1 = ax.bar(x - width/2, opengl_fps, width, label='OpenGL', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, vulkan_fps, width, label='Vulkan', alpha=0.8, color='#e74c3c')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}', 
                   ha='center', va='bottom', fontsize=FONT_SIZE-2, fontweight='bold')

    ax.set_xlabel('Scena', fontsize=14, fontweight='bold')
    ax.set_ylabel('FPS Medio', fontsize=14, fontweight='bold')
    ax.set_title('OpenGL vs Vulkan: FPS Medio su Tutti i PC', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Scena {s}' for s in scenes])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    save_plot(f"{IMG_ROOT}/01_overall_fps_comparison.png")

# =====================================================
# 2. PERFORMANCE IMPROVEMENT - Percentage gain
# =====================================================
if "Average FPS" in summary.columns:
    fig, ax = plt.subplots(figsize=(12, 7))
    scenes = sorted(summary["Scene"].unique())

    improvements = []
    for scene in scenes:
        opengl_fps = summary[(summary["Scene"] == scene) & (summary["API"] == "opengl")]["Average FPS"].mean()
        vulkan_fps = summary[(summary["Scene"] == scene) & (summary["API"] == "vulkan")]["Average FPS"].mean()
        improvement = ((vulkan_fps - opengl_fps) / opengl_fps) * 100 if opengl_fps > 0 else 0
        improvements.append(improvement)

    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax.bar(range(len(scenes)), improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(i, val + (2 if val > 0 else -2), f'{val:+.1f}%', ha='center', 
               va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Scena', fontsize=14, fontweight='bold')
    ax.set_ylabel('Miglioramento delle Prestazioni (%)', fontsize=14, fontweight='bold')
    ax.set_title('Guadagno di Prestazioni Vulkan Rispetto a OpenGL (Tutti i PC)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels([f'Scena {s}' for s in scenes])
    ax.grid(axis='y', alpha=0.3)

    save_plot(f"{IMG_ROOT}/02_performance_improvement.png")

# =====================================================
# 3. HEATMAP - OpenGL vs Vulkan Performance by PC
# =====================================================
if "Average FPS" in summary.columns and not summary.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    opengl_data = summary[summary["API"] == "opengl"].pivot_table(
        index="PC", columns="Scene", values="Average FPS", aggfunc='mean'
    )
    
    sns.heatmap(opengl_data, annot=True, fmt='.0f', cmap='Blues', 
                ax=ax1, cbar_kws={'label': 'FPS'}, linewidths=0.5)
    ax1.set_title('Prestazioni OpenGL', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Scena', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Configurazione Hardware', fontsize=12, fontweight='bold')
    
    vulkan_data = summary[summary["API"] == "vulkan"].pivot_table(
        index="PC", columns="Scene", values="Average FPS", aggfunc='mean'
    )
    
    sns.heatmap(vulkan_data, annot=True, fmt='.0f', cmap='Reds', 
                ax=ax2, cbar_kws={'label': 'FPS'}, linewidths=0.5)
    ax2.set_title('Prestazioni Vulkan', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Scena', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Configurazione Hardware', fontsize=12, fontweight='bold')
    
    plt.suptitle('Heatmap delle Prestazioni: OpenGL vs Vulkan', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_plot(f"{IMG_ROOT}/03_heatmap_comparison.png")

# =====================================================
# 4. HEATMAP - Percentage Improvement by PC and Scene
# =====================================================
if not summary.empty:
    pivot_data = []
    for pc in summary["PC"].unique():
        for scene in summary["Scene"].unique():
            opengl = summary[(summary["PC"] == pc) & (summary["Scene"] == scene) & (summary["API"] == "opengl")]["Average FPS"]
            vulkan = summary[(summary["PC"] == pc) & (summary["Scene"] == scene) & (summary["API"] == "vulkan")]["Average FPS"]
            if len(opengl) > 0 and len(vulkan) > 0:
                gain = ((vulkan.values[0] - opengl.values[0]) / opengl.values[0]) * 100
                pivot_data.append({"PC": pc, "Scene": scene, "Gain(%)": gain})
    
    df_gain = pd.DataFrame(pivot_data)
    if not df_gain.empty:
        pivot = df_gain.pivot(index="PC", columns="Scene", values="Gain(%)")
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".1f", 
                   linewidths=0.5, cbar_kws={"label": "Guadagno Vulkan (%)"})
        ax.set_title("Miglioramento delle Prestazioni Vulkan per PC e Scena", 
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Scena', fontsize=12, fontweight='bold')
        ax.set_ylabel('Configurazione Hardware', fontsize=12, fontweight='bold')
        save_plot(f"{IMG_ROOT}/04_heatmap_improvement.png")

# =====================================================
# 5. PC RANKING - Average Performance Gain
# =====================================================
if not summary.empty:
    mean_gain = []
    for pc in summary["PC"].unique():
        total_gain = 0
        count = 0
        for scene in summary["Scene"].unique():
            opengl = summary[(summary["PC"] == pc) & (summary["Scene"] == scene) & (summary["API"] == "opengl")]["Average FPS"]
            vulkan = summary[(summary["PC"] == pc) & (summary["Scene"] == scene) & (summary["API"] == "vulkan")]["Average FPS"]
            if len(opengl) > 0 and len(vulkan) > 0:
                gain = ((vulkan.values[0] - opengl.values[0]) / opengl.values[0]) * 100
                total_gain += gain
                count += 1
        avg_gain = total_gain / count if count > 0 else 0
        mean_gain.append({"PC": pc, "Average Gain (%)": avg_gain, "GPU": pc_gpu_map.get(pc, "Unknown")})

    df_rank = pd.DataFrame(mean_gain).sort_values("Average Gain (%)", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in df_rank["Average Gain (%)"]]
    bars = ax.barh(range(len(df_rank)), df_rank["Average Gain (%)"], color=colors, alpha=0.8)
    
    for i, (bar, val) in enumerate(zip(bars, df_rank["Average Gain (%)"])):
        ax.text(val + (1 if val > 0 else -1), i, f'{val:+.1f}%', 
               va='center', ha='left' if val > 0 else 'right', fontsize=10, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_yticks(range(len(df_rank)))
    ax.set_yticklabels(df_rank["PC"])
    ax.set_xlabel("Guadagno Medio delle Prestazioni (%)", fontsize=12, fontweight='bold')
    ax.set_title("Classifica PC: Miglioramento delle Prestazioni Vulkan", fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    
    save_plot(f"{IMG_ROOT}/05_pc_ranking.png")

# =====================================================
# 6. CONSISTENCY ANALYSIS - Frame Time Stability (Per-PC Average)
# =====================================================
if not frames.empty and "FrameTime(ms)" in frames.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenes = sorted(frames["Scene"].unique())
    pcs = sorted(frames["PC"].unique())
    
    opengl_cv_by_scene = []
    vulkan_cv_by_scene = []
    
    # Calculate CV per PC, then average across PCs for each scene
    for scene in scenes:
        opengl_cvs = []
        vulkan_cvs = []
        
        for pc in pcs:
            # OpenGL CV for this PC and scene
            opengl_times = frames[(frames["Scene"] == scene) & 
                                 (frames["API"] == "opengl") & 
                                 (frames["PC"] == pc)]["FrameTime(ms)"]
            if len(opengl_times) > 0 and opengl_times.mean() > 0:
                cv = (opengl_times.std() / opengl_times.mean()) * 100
                opengl_cvs.append(cv)
            
            # Vulkan CV for this PC and scene
            vulkan_times = frames[(frames["Scene"] == scene) & 
                                 (frames["API"] == "vulkan") & 
                                 (frames["PC"] == pc)]["FrameTime(ms)"]
            if len(vulkan_times) > 0 and vulkan_times.mean() > 0:
                cv = (vulkan_times.std() / vulkan_times.mean()) * 100
                vulkan_cvs.append(cv)
        
        # Average CV across all PCs for this scene
        opengl_cv_by_scene.append(np.mean(opengl_cvs) if opengl_cvs else 0)
        vulkan_cv_by_scene.append(np.mean(vulkan_cvs) if vulkan_cvs else 0)
    
    x = np.arange(len(scenes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, opengl_cv_by_scene, width, label='OpenGL', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, vulkan_cv_by_scene, width, label='Vulkan', alpha=0.8, color='#e74c3c')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=FONT_SIZE-2)
    
    ax.set_xlabel('Scena', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coefficiente di Variazione Medio (%)', fontsize=14, fontweight='bold')
    ax.set_title('Coerenza del Frame Time - Media su Tutti i PC (PiÃ¹ Basso = PiÃ¹ Stabile)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Scena {s}' for s in scenes])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    save_plot(f"{IMG_ROOT}/06_frametime_stability.png")

# =====================================================
# 7. 1% and 99% PERCENTILE FPS - Worst Case Performance
# =====================================================
if "P1 FPS" in summary.columns and "P99 FPS" in summary.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    scenes = sorted(summary["Scene"].unique())
    x = np.arange(len(scenes))
    width = 0.35
    
    # 1% Low FPS (worst case)
    opengl_p1 = [summary[(summary["Scene"] == s) & (summary["API"] == "opengl")]["P1 FPS"].mean() for s in scenes]
    vulkan_p1 = [summary[(summary["Scene"] == s) & (summary["API"] == "vulkan")]["P1 FPS"].mean() for s in scenes]
    
    bars1 = ax1.bar(x - width/2, opengl_p1, width, label='OpenGL', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, vulkan_p1, width, label='Vulkan', alpha=0.8, color='#e74c3c')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}',
                    ha='center', va='bottom', fontsize=FONT_SIZE-2)
    
    ax1.set_xlabel('Scena', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FPS 1% PiÃ¹ Bassi', fontsize=12, fontweight='bold')
    ax1.set_title('Prestazioni Caso Peggiore (FPS 1% PiÃ¹ Bassi)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Scena {s}' for s in scenes])
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 99% High FPS (best case)
    opengl_p99 = [summary[(summary["Scene"] == s) & (summary["API"] == "opengl")]["P99 FPS"].mean() for s in scenes]
    vulkan_p99 = [summary[(summary["Scene"] == s) & (summary["API"] == "vulkan")]["P99 FPS"].mean() for s in scenes]
    
    bars1 = ax2.bar(x - width/2, opengl_p99, width, label='OpenGL', alpha=0.8, color='#3498db')
    bars2 = ax2.bar(x + width/2, vulkan_p99, width, label='Vulkan', alpha=0.8, color='#e74c3c')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}',
                    ha='center', va='bottom', fontsize=FONT_SIZE-2)
    
    ax2.set_xlabel('Scena', fontsize=12, fontweight='bold')
    ax2.set_ylabel('FPS 99% PiÃ¹ Alti', fontsize=12, fontweight='bold')
    ax2.set_title('Prestazioni Caso Migliore (FPS 99% PiÃ¹ Alti)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Scena {s}' for s in scenes])
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Percentili delle Prestazioni: OpenGL vs Vulkan', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_plot(f"{IMG_ROOT}/07_percentile_comparison.png")

# =====================================================
# 8. FPS DISTRIBUTION - Box Plot by Scene (Separate by PC)
# =====================================================
if not frames.empty and "FPS" in frames.columns:
    scenes = sorted(frames["Scene"].unique())
    n_scenes = len(scenes)
    n_cols = 3
    n_rows = (n_scenes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, scene in enumerate(scenes):
        ax = axes[idx]
        scene_data = frames[frames["Scene"] == scene]
        
        # Create box plot separated by PC and API
        data_by_pc = []
        labels = []
        colors = []
        
        for pc in sorted(scene_data["PC"].unique()):
            pc_opengl = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "opengl")]["FPS"]
            pc_vulkan = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "vulkan")]["FPS"]
            
            if len(pc_opengl) > 0:
                data_by_pc.append(pc_opengl)
                labels.append(f'{pc}\nOGL')
                colors.append('#3498db')
            
            if len(pc_vulkan) > 0:
                data_by_pc.append(pc_vulkan)
                labels.append(f'{pc}\nVK')
                colors.append('#e74c3c')
        
        if data_by_pc:
            bp = ax.boxplot(data_by_pc, tick_labels=labels, patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='yellow', markersize=6))
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['OpenGL', 'Vulkan'])
        ax.set_title(f'Scena {scene}', fontsize=12, fontweight='bold')
        ax.set_ylabel('FPS', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    for idx in range(n_scenes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Distribuzione FPS per Scena e PC (Box Plot)', 
                 fontsize=16, fontweight='bold')
    
    save_plot(f"{IMG_ROOT}/08_fps_distribution.png")

# =====================================================
# 9. RENDER PASS BREAKDOWN - Stacked Bars
# =====================================================
passes = [p for p in ["Geometry Pass", "Lighting Pass", "Gizmo Pass", "Particle Pass", "ImGui Pass"] 
          if p in summary.columns]

if passes and not summary.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    scenes = sorted(summary["Scene"].unique())
    
    # OpenGL
    opengl_summary = summary[summary["API"] == "opengl"]
    opengl_data = {p: [opengl_summary[opengl_summary["Scene"] == s][p].mean() for s in scenes] 
                   for p in passes}
    
    bottom = np.zeros(len(scenes))
    for i, (pass_name, values) in enumerate(opengl_data.items()):
        ax1.bar(range(len(scenes)), values, bottom=bottom, label=pass_name, alpha=0.8)
        bottom += values
    
    ax1.set_xlabel('Scena', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Tempo (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Suddivisione Render Pass OpenGL', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(scenes)))
    ax1.set_xticklabels([f'Scena {s}' for s in scenes])
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # Vulkan
    vulkan_summary = summary[summary["API"] == "vulkan"]
    vulkan_data = {p: [vulkan_summary[vulkan_summary["Scene"] == s][p].mean() for s in scenes] 
                   for p in passes}
    
    bottom = np.zeros(len(scenes))
    for i, (pass_name, values) in enumerate(vulkan_data.items()):
        ax2.bar(range(len(scenes)), values, bottom=bottom, label=pass_name, alpha=0.8)
        bottom += values
    
    ax2.set_xlabel('Scena', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tempo (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Suddivisione Render Pass Vulkan', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(scenes)))
    ax2.set_xticklabels([f'Scena {s}' for s in scenes])
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Suddivisione Tempo Render Pass', fontsize=16, fontweight='bold')
    
    save_plot(f"{IMG_ROOT}/09_render_pass_breakdown.png")

# =====================================================
# 10. COMPREHENSIVE SUMMARY TABLE
# =====================================================
if "Average FPS" in summary.columns:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    scenes = sorted(summary["Scene"].unique())
    
    for scene in scenes:
        opengl_fps = summary[(summary["Scene"] == scene) & (summary["API"] == "opengl")]["Average FPS"].mean()
        vulkan_fps = summary[(summary["Scene"] == scene) & (summary["API"] == "vulkan")]["Average FPS"].mean()
        improvement = ((vulkan_fps - opengl_fps) / opengl_fps) * 100 if opengl_fps > 0 else 0
        
        opengl_p1 = summary[(summary["Scene"] == scene) & (summary["API"] == "opengl")]["P1 FPS"].mean()
        vulkan_p1 = summary[(summary["Scene"] == scene) & (summary["API"] == "vulkan")]["P1 FPS"].mean()
        
        table_data.append([
            f'Scene {scene}',
            f'{opengl_fps:.1f}',
            f'{vulkan_fps:.1f}',
            f'{improvement:+.1f}%',
            f'{opengl_p1:.1f}',
            f'{vulkan_p1:.1f}'
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Scena', 'Media OGL', 'Media VK', 'Guadagno', 'OGL 1% Bassi', 'VK 1% Bassi'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.18, 0.18, 0.14, 0.18, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, row in enumerate(table_data, 1):
        improvement_val = float(row[3].strip('%+'))
        color = '#27ae60' if improvement_val > 0 else '#e74c3c'
        table[(i, 3)].set_facecolor(color)
        table[(i, 3)].set_alpha(0.3)
    
    plt.title('Riepilogo Prestazioni: OpenGL vs Vulkan (Tutti i PC)', 
              fontsize=18, fontweight='bold', pad=30)
    
    save_plot(f"{IMG_ROOT}/10_summary_table.png")

# =====================================================
# 11-15. PER-PC DETAILED ANALYSIS
# =====================================================
if "Average FPS" in summary.columns and not summary.empty:
    pcs = sorted(summary["PC"].unique())
    
    for pc in pcs:
        pc_safe = pc.replace(' ', '_').replace('/', '_')
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        pc_data = summary[summary["PC"] == pc]
        scenes = sorted(pc_data["Scene"].unique())
        
        # Top left: FPS comparison
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(scenes))
        width = 0.35
        
        opengl_fps = [pc_data[(pc_data["Scene"] == s) & (pc_data["API"] == "opengl")]["Average FPS"].values[0] 
                     if len(pc_data[(pc_data["Scene"] == s) & (pc_data["API"] == "opengl")]) > 0 else 0 
                     for s in scenes]
        vulkan_fps = [pc_data[(pc_data["Scene"] == s) & (pc_data["API"] == "vulkan")]["Average FPS"].values[0] 
                     if len(pc_data[(pc_data["Scene"] == s) & (pc_data["API"] == "vulkan")]) > 0 else 0 
                     for s in scenes]
        
        bars1 = ax1.bar(x - width/2, opengl_fps, width, label='OpenGL', alpha=0.8, color='#3498db')
        bars2 = ax1.bar(x + width/2, vulkan_fps, width, label='Vulkan', alpha=0.8, color='#e74c3c')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height, 
                           f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Scena', fontsize=10, fontweight='bold')
        ax1.set_ylabel('FPS Medio', fontsize=10, fontweight='bold')
        ax1.set_title('FPS Medio per Scena', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'S{s}' for s in scenes])
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Top right: Improvement percentage
        ax2 = fig.add_subplot(gs[0, 1])
        improvements = []
        for scene in scenes:
            opengl = pc_data[(pc_data["Scene"] == scene) & (pc_data["API"] == "opengl")]["Average FPS"]
            vulkan = pc_data[(pc_data["Scene"] == scene) & (pc_data["API"] == "vulkan")]["Average FPS"]
            
            if len(opengl) > 0 and len(vulkan) > 0:
                improvement = ((vulkan.values[0] - opengl.values[0]) / opengl.values[0]) * 100
            else:
                improvement = 0
            improvements.append(improvement)
        
        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in improvements]
        bars = ax2.bar(range(len(scenes)), improvements, color=colors, alpha=0.8)
        
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            ax2.text(i, val + (1 if val > 0 else -1), f'{val:+.1f}%', 
                   ha='center', va='bottom' if val > 0 else 'top', fontsize=8, fontweight='bold')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Scena', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Guadagno Prestazioni (%)', fontsize=10, fontweight='bold')
        ax2.set_title('Miglioramento Prestazioni Vulkan', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(scenes)))
        ax2.set_xticklabels([f'S{s}' for s in scenes])
        ax2.grid(axis='y', alpha=0.3)
        
        # Bottom left: 1% Low FPS
        ax3 = fig.add_subplot(gs[1, 0])
        opengl_p1 = [pc_data[(pc_data["Scene"] == s) & (pc_data["API"] == "opengl")]["P1 FPS"].values[0] 
                     if len(pc_data[(pc_data["Scene"] == s) & (pc_data["API"] == "opengl")]) > 0 else 0 
                     for s in scenes]
        vulkan_p1 = [pc_data[(pc_data["Scene"] == s) & (pc_data["API"] == "vulkan")]["P1 FPS"].values[0] 
                     if len(pc_data[(pc_data["Scene"] == s) & (pc_data["API"] == "vulkan")]) > 0 else 0 
                     for s in scenes]
        
        bars1 = ax3.bar(x - width/2, opengl_p1, width, label='OpenGL', alpha=0.8, color='#3498db')
        bars2 = ax3.bar(x + width/2, vulkan_p1, width, label='Vulkan', alpha=0.8, color='#e74c3c')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height, 
                           f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        ax3.set_xlabel('Scena', fontsize=10, fontweight='bold')
        ax3.set_ylabel('FPS 1% PiÃ¹ Bassi', fontsize=10, fontweight='bold')
        ax3.set_title('Prestazioni Caso Peggiore (1% PiÃ¹ Bassi)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'S{s}' for s in scenes])
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        
        # Bottom right: Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        summary_text = f"Configurazione: {pc}\n"
        if pc in pc_gpu_map:
            summary_text += f"GPU: {pc_gpu_map[pc]}\n\n"
        
        avg_opengl = np.mean([f for f in opengl_fps if f > 0])
        avg_vulkan = np.mean([f for f in vulkan_fps if f > 0])
        avg_improvement = ((avg_vulkan - avg_opengl) / avg_opengl) * 100 if avg_opengl > 0 else 0
        
        summary_text += f"Statistiche Generali:\n"
        summary_text += f"  Media FPS OpenGL: {avg_opengl:.1f}\n"
        summary_text += f"  Media FPS Vulkan: {avg_vulkan:.1f}\n"
        summary_text += f"  Guadagno Medio: {avg_improvement:+.1f}%\n\n"
        
        summary_text += f"Scena Migliore (Vulkan):\n"
        best_scene_idx = np.argmax(vulkan_fps)
        summary_text += f"  Scena {scenes[best_scene_idx]}: {vulkan_fps[best_scene_idx]:.1f} FPS\n\n"
        
        summary_text += f"Scena Peggiore (Vulkan):\n"
        worst_scene_idx = np.argmin([f if f > 0 else float('inf') for f in vulkan_fps])
        summary_text += f"  Scena {scenes[worst_scene_idx]}: {vulkan_fps[worst_scene_idx]:.1f} FPS\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{pc}: Analisi Dettagliata delle Prestazioni', 
                    fontsize=16, fontweight='bold')
        
        save_plot(f"{IMG_ROOT}/11_per_pc_{pc_safe}_analysis.png")

# =====================================================
# 12. CROSS-PC PERFORMANCE COMPARISON
# =====================================================
if "Average FPS" in summary.columns and not summary.empty:
    pcs = sorted(summary["PC"].unique())
    
    # Select a few key scenes for comparison
    scenes = sorted(summary["Scene"].unique())
    key_scenes = scenes[::max(1, len(scenes)//4)][:4]  # Pick up to 4 evenly distributed scenes
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scene in enumerate(key_scenes):
        ax = axes[idx]
        scene_data = summary[summary["Scene"] == scene]
        
        opengl_vals = []
        vulkan_vals = []
        pc_labels = []
        
        for pc in pcs:
            opengl = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "opengl")]["Average FPS"]
            vulkan = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "vulkan")]["Average FPS"]
            
            if len(opengl) > 0 and len(vulkan) > 0:
                opengl_vals.append(opengl.values[0])
                vulkan_vals.append(vulkan.values[0])
                pc_labels.append(pc)
        
        x = np.arange(len(pc_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, opengl_vals, width, label='OpenGL', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, vulkan_vals, width, label='Vulkan', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Configurazione PC', fontsize=10, fontweight='bold')
        ax.set_ylabel('FPS Medio', fontsize=10, fontweight='bold')
        ax.set_title(f'Scena {scene}: Prestazioni tra i PC', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pc_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(key_scenes), 4):
        axes[idx].axis('off')
    
    plt.suptitle('Confronto Prestazioni tra PC', 
                 fontsize=16, fontweight='bold')
    
    save_plot(f"{IMG_ROOT}/12_cross_pc_comparison.png")

# =====================================================
# 13. PERFORMANCE SCALING ANALYSIS
# =====================================================
if "Average FPS" in summary.columns and not summary.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    scenes = sorted(summary["Scene"].unique())
    
    # OpenGL scaling across PCs
    for pc in sorted(summary["PC"].unique()):
        pc_data = summary[(summary["PC"] == pc) & (summary["API"] == "opengl")]
        fps_vals = [pc_data[pc_data["Scene"] == s]["Average FPS"].values[0] 
                   if len(pc_data[pc_data["Scene"] == s]) > 0 else np.nan 
                   for s in scenes]
        ax1.plot(scenes, fps_vals, marker='o', label=pc, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Scena', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FPS Medio', fontsize=12, fontweight='bold')
    ax1.set_title('ScalabilitÃ  Prestazioni OpenGL', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(alpha=0.3)
    
    # Vulkan scaling across PCs
    for pc in sorted(summary["PC"].unique()):
        pc_data = summary[(summary["PC"] == pc) & (summary["API"] == "vulkan")]
        fps_vals = [pc_data[pc_data["Scene"] == s]["Average FPS"].values[0] 
                   if len(pc_data[pc_data["Scene"] == s]) > 0 else np.nan 
                   for s in scenes]
        ax2.plot(scenes, fps_vals, marker='o', label=pc, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Scena', fontsize=12, fontweight='bold')
    ax2.set_ylabel('FPS Medio', fontsize=12, fontweight='bold')
    ax2.set_title('ScalabilitÃ  Prestazioni Vulkan', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(alpha=0.3)
    
    plt.suptitle('ScalabilitÃ  Prestazioni tra Scene e Hardware', 
                 fontsize=16, fontweight='bold')
    
    save_plot(f"{IMG_ROOT}/13_performance_scaling.png")

# =====================================================
# 14. STABILITY IMPROVEMENT HEATMAP (Per-PC)
# =====================================================
if not frames.empty and "FrameTime(ms)" in frames.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pcs = sorted(frames["PC"].unique())
    scenes = sorted(frames["Scene"].unique())
    
    stability_data = []
    
    for pc in pcs:
        for scene in scenes:
            # Calculate CV for OpenGL
            opengl_times = frames[(frames["PC"] == pc) & 
                                 (frames["Scene"] == scene) & 
                                 (frames["API"] == "opengl")]["FrameTime(ms)"]
            
            # Calculate CV for Vulkan
            vulkan_times = frames[(frames["PC"] == pc) & 
                                 (frames["Scene"] == scene) & 
                                 (frames["API"] == "vulkan")]["FrameTime(ms)"]
            
            if len(opengl_times) > 0 and len(vulkan_times) > 0:
                opengl_cv = (opengl_times.std() / opengl_times.mean()) * 100 if opengl_times.mean() > 0 else 0
                vulkan_cv = (vulkan_times.std() / vulkan_times.mean()) * 100 if vulkan_times.mean() > 0 else 0
                
                # Positive value = Vulkan is more stable (lower CV)
                stability_improvement = ((opengl_cv - vulkan_cv) / opengl_cv) * 100 if opengl_cv > 0 else 0
                
                stability_data.append({
                    "PC": pc,
                    "Scene": scene,
                    "Stability Improvement (%)": stability_improvement
                })
    
    df_stability = pd.DataFrame(stability_data)
    
    if not df_stability.empty:
        pivot = df_stability.pivot(index="PC", columns="Scene", values="Stability Improvement (%)")
        
        sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".1f", 
                   linewidths=0.5, cbar_kws={"label": "Miglioramento StabilitÃ  (%)"})
        ax.set_title("Miglioramento StabilitÃ  Frame Time (Vulkan vs OpenGL)\nPositivo = Vulkan PiÃ¹ Stabile", 
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Scena', fontsize=12, fontweight='bold')
        ax.set_ylabel('Configurazione Hardware', fontsize=12, fontweight='bold')
        
        save_plot(f"{IMG_ROOT}/14_stability_improvement_heatmap.png")

# ==============================
# NEW PLOT 15: FPS by Hardware for Scenes 0,1,2
# ==============================
if "Average FPS" in summary.columns and not summary.empty:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scenes_group_1 = [0, 1, 2]
    
    for idx, scene in enumerate(scenes_group_1):
        ax = axes[idx]
        scene_data = summary[summary["Scene"] == scene]
        
        pcs = sorted(scene_data["PC"].unique())
        x = np.arange(len(pcs))
        width = 0.35
        
        opengl_vals = []
        vulkan_vals = []
        
        for pc in pcs:
            opengl = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "opengl")]["Average FPS"]
            vulkan = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "vulkan")]["Average FPS"]
            
            opengl_vals.append(opengl.values[0] if len(opengl) > 0 else 0)
            vulkan_vals.append(vulkan.values[0] if len(vulkan) > 0 else 0)
        
        bars1 = ax.bar(x - width/2, opengl_vals, width, label='OpenGL', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, vulkan_vals, width, label='Vulkan', alpha=0.8, color='#e74c3c')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height, 
                           f'{height:.0f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Configurazione Hardware', fontsize=12, fontweight='bold')
        ax.set_ylabel('FPS Medio', fontsize=12, fontweight='bold')
        ax.set_title(f'Scena {scene}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pcs, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Confronto FPS per Hardware: Scene 0, 1, 2', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_plot(f"{IMG_ROOT}/15_fps_by_hardware_scenes_0_1_2.png")

# ==============================
# NEW PLOT 16: FPS by Hardware for Scenes 3,4
# ==============================
if "Average FPS" in summary.columns and not summary.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scenes_group_2 = [3, 4]
    
    for idx, scene in enumerate(scenes_group_2):
        ax = axes[idx]
        scene_data = summary[summary["Scene"] == scene]
        
        pcs = sorted(scene_data["PC"].unique())
        x = np.arange(len(pcs))
        width = 0.35
        
        opengl_vals = []
        vulkan_vals = []
        
        for pc in pcs:
            opengl = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "opengl")]["Average FPS"]
            vulkan = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "vulkan")]["Average FPS"]
            
            opengl_vals.append(opengl.values[0] if len(opengl) > 0 else 0)
            vulkan_vals.append(vulkan.values[0] if len(vulkan) > 0 else 0)
        
        bars1 = ax.bar(x - width/2, opengl_vals, width, label='OpenGL', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, vulkan_vals, width, label='Vulkan', alpha=0.8, color='#e74c3c')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height, 
                           f'{height:.0f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Configurazione Hardware', fontsize=12, fontweight='bold')
        ax.set_ylabel('FPS Medio', fontsize=12, fontweight='bold')
        ax.set_title(f'Scena {scene}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pcs, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Confronto FPS per Hardware: Scene 3, 4', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_plot(f"{IMG_ROOT}/16_fps_by_hardware_scenes_3_4.png")

# ==============================
# NEW PLOT 17: Performance Gain by Hardware for Scenes 0,1,2
# ==============================
if "Average FPS" in summary.columns and not summary.empty:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scenes_group_1 = [0, 1, 2]
    
    for idx, scene in enumerate(scenes_group_1):
        ax = axes[idx]
        scene_data = summary[summary["Scene"] == scene]
        
        pcs = sorted(scene_data["PC"].unique())
        gains = []
        
        for pc in pcs:
            opengl = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "opengl")]["Average FPS"]
            vulkan = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "vulkan")]["Average FPS"]
            
            if len(opengl) > 0 and len(vulkan) > 0:
                gain = ((vulkan.values[0] - opengl.values[0]) / opengl.values[0]) * 100
            else:
                gain = 0
            gains.append(gain)
        
        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in gains]
        bars = ax.bar(range(len(pcs)), gains, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, gains)):
            ax.text(i, val + (2 if val > 0 else -2), f'{val:+.1f}%', 
                   ha='center', va='bottom' if val > 0 else 'top', 
                   fontsize=10, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Configurazione Hardware', fontsize=12, fontweight='bold')
        ax.set_ylabel('Guadagno (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Scena {scene}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(pcs)))
        ax.set_xticklabels(pcs, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Guadagno Percentuale Vulkan vs OpenGL: Scene 0, 1, 2', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_plot(f"{IMG_ROOT}/17_gain_by_hardware_scenes_0_1_2.png")

# ==============================
# NEW PLOT 18: Performance Gain by Hardware for Scenes 3,4
# ==============================
if "Average FPS" in summary.columns and not summary.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scenes_group_2 = [3, 4]
    
    for idx, scene in enumerate(scenes_group_2):
        ax = axes[idx]
        scene_data = summary[summary["Scene"] == scene]
        
        pcs = sorted(scene_data["PC"].unique())
        gains = []
        
        for pc in pcs:
            opengl = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "opengl")]["Average FPS"]
            vulkan = scene_data[(scene_data["PC"] == pc) & (scene_data["API"] == "vulkan")]["Average FPS"]
            
            if len(opengl) > 0 and len(vulkan) > 0:
                gain = ((vulkan.values[0] - opengl.values[0]) / opengl.values[0]) * 100
            else:
                gain = 0
            gains.append(gain)
        
        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in gains]
        bars = ax.bar(range(len(pcs)), gains, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, gains)):
            ax.text(i, val + (2 if val > 0 else -2), f'{val:+.1f}%', 
                   ha='center', va='bottom' if val > 0 else 'top', 
                   fontsize=10, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Configurazione Hardware', fontsize=12, fontweight='bold')
        ax.set_ylabel('Guadagno (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Scena {scene}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(pcs)))
        ax.set_xticklabels(pcs, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Guadagno Percentuale Vulkan vs OpenGL: Scene 3, 4', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_plot(f"{IMG_ROOT}/18_gain_by_hardware_scenes_3_4.png")

# ==============================
# FINAL SUMMARY (UNCHANGED)
# ==============================
print("\n" + "="*60)
print("=== ANALISI COMPLETATA ===")
print("="*60)
print(f"\nCartella output: {IMG_ROOT}/")
print(f"Configurazioni totali analizzate: {len(summary)}")
print(f"Record frame totali: {len(frames)}")
print(f"PC unici: {len(frames['PC'].unique()) if not frames.empty else 0}")
print(f"Scene uniche: {len(frames['Scene'].unique()) if not frames.empty else 0}")

if not summary.empty:
    print("\n--- Riepilogo Prestazioni Generali ---")
    overall_opengl = summary[summary["API"] == "opengl"]["Average FPS"].mean()
    overall_vulkan = summary[summary["API"] == "vulkan"]["Average FPS"].mean()
    overall_gain = ((overall_vulkan - overall_opengl) / overall_opengl) * 100
    
    print(f"FPS Medio OpenGL: {overall_opengl:.1f}")
    print(f"FPS Medio Vulkan: {overall_vulkan:.1f}")
    print(f"Guadagno Generale Vulkan: {overall_gain:+.1f}%")
    
    print("\n--- Prestazioni per Scena ---")
    for scene in sorted(summary["Scene"].unique()):
        opengl_fps = summary[(summary["Scene"] == scene) & (summary["API"] == "opengl")]["Average FPS"].mean()
        vulkan_fps = summary[(summary["Scene"] == scene) & (summary["API"] == "vulkan")]["Average FPS"].mean()
        gain = ((vulkan_fps - opengl_fps) / opengl_fps) * 100
        print(f"Scena {scene}: OpenGL={opengl_fps:.1f} FPS, Vulkan={vulkan_fps:.1f} FPS, Guadagno={gain:+.1f}%")

print("\n" + "="*60)
