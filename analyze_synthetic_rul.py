#!/usr/bin/env python3
"""
Synthetic OBD-II RUL Dataset Analysis
=====================================
Comprehensive analysis of the generated synthetic OBD-II RUL dataset.
Generates publication-quality figures for academic reports.

Usage: python analyze_synthetic_rul.py
Input: output/synthetic_obd_rul_dataset.csv
Output: Multiple PNG figures + CSV summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Academic plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = Path("./output/synthetic_obd_rul_dataset.csv")
OUTPUT_DIR = Path("./output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OP_MODES = ['idle', 'city', 'highway', 'aggressive']

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("SYNTHETIC OBD-II RUL DATASET - ANALYSIS")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset loaded: {DATA_PATH}")
print(f"Shape: {df.shape}")

# Identify columns
meta_cols = ['unit_number', 'time_cycles', 'km_driven', 'total_lifetime_km',
             'operational_mode', 'degradation_factor', 'RUL']
sensor_cols = [c for c in df.columns if c.startswith('PID_')]

# Add phase column
def get_phase(row):
    if row['degradation_factor'] == 0:
        return 'Healthy'
    elif row['degradation_factor'] <= 0.3:
        return 'Wear Onset'
    else:
        return 'Accelerated'

df['phase'] = df.apply(get_phase, axis=1)

# =============================================================================
# 1. BASIC STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("1. BASIC STATISTICS")
print("=" * 80)

print(f"\nMetadata columns ({len(meta_cols)}):")
for c in meta_cols:
    print(f"  {c}")

print(f"\nSensor columns ({len(sensor_cols)}):")
for c in sensor_cols:
    print(f"  {c}")

print(f"\nNumeric summary:")
print(df[meta_cols].describe().to_string())

print(f"\nOperational mode distribution:")
print(df['operational_mode'].value_counts().to_string())

unit_stats = df.groupby('unit_number').agg({
    'total_lifetime_km': 'first',
    'time_cycles': 'max',
}).rename(columns={'time_cycles': 'n_samples'})

print(f"\nUnit statistics:")
print(f"  Total units: {df['unit_number'].nunique()}")
print(f"  Mean lifetime: {unit_stats['total_lifetime_km'].mean():.1f} km")
print(f"  Min lifetime: {unit_stats['total_lifetime_km'].min()} km")
print(f"  Max lifetime: {unit_stats['total_lifetime_km'].max()} km")
print(f"  Mean samples/unit: {unit_stats['n_samples'].mean():.0f}")

# =============================================================================
# 2. RUL ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("2. GENERATING RUL ANALYSIS FIGURE...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Synthetic OBD-II RUL Dataset: RUL Analysis',
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1: RUL distribution
ax1 = axes[0, 0]
ax1.hist(df['RUL'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(x=5000, color='red', linestyle='--', linewidth=2, label='RUL Cap (5000 km)')
ax1.set_xlabel('RUL (km)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('(a) RUL Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: RUL by phase
ax2 = axes[0, 1]
phase_rul = df.groupby('phase')['RUL'].mean().reindex(['Healthy', 'Wear Onset', 'Accelerated'])
colors_phase = ['#2ECC71', '#F39C12', '#E74C3C']
ax2.bar(phase_rul.index, phase_rul.values, color=colors_phase, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Mean RUL (km)', fontsize=11)
ax2.set_title('(b) Mean RUL by Degradation Phase', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Sample RUL trajectories
ax3 = axes[1, 0]
sample_units = [1, 10, 25, 50, 75]
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_units)))
for i, uid in enumerate(sample_units):
    df_u = df[df['unit_number'] == uid]
    ax3.plot(df_u['km_driven'], df_u['RUL'], color=colors[i], alpha=0.8, linewidth=2, label=f'Unit {uid}')
ax3.axhline(y=5000, color='red', linestyle='--', alpha=0.5, label='RUL Cap')
ax3.set_xlabel('KM Driven', fontsize=11)
ax3.set_ylabel('RUL (km)', fontsize=11)
ax3.set_title('(c) RUL Decay Trajectories', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: RUL vs Degradation Factor
ax4 = axes[1, 1]
sample = df.sample(n=min(5000, len(df)), random_state=42)
ax4.scatter(sample['degradation_factor'], sample['RUL'], alpha=0.3, s=10, color='steelblue')
ax4.set_xlabel('Degradation Factor', fontsize=11)
ax4.set_ylabel('RUL (km)', fontsize=11)
ax4.set_title('(d) RUL vs Degradation Factor', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig01_rul_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig01_rul_analysis.png")

# =============================================================================
# 3. SENSOR STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("3. GENERATING SENSOR STATISTICS FIGURE...")
print("=" * 80)

print("\nSensor descriptive statistics:")
sensor_stats = df[sensor_cols].describe().T
sensor_stats['range'] = sensor_stats['max'] - sensor_stats['min']
print(sensor_stats[['mean', 'std', 'min', 'max', 'range']].to_string())

print("\nSensor means by degradation phase:")
phase_sensor_means = df.groupby('phase')[sensor_cols].mean()
print(phase_sensor_means.to_string())

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle('Synthetic OBD-II RUL Dataset: Sensor Statistics',
             fontsize=14, fontweight='bold', y=1.01)

# Plot 1: Sensor means
ax1 = axes[0, 0]
means = df[sensor_cols].mean().sort_values(ascending=True)
ax1.barh(range(len(means)), means.values, color='steelblue', alpha=0.8, edgecolor='black')
ax1.set_yticks(range(len(means)))
ax1.set_yticklabels([s.replace('PID_', '').replace('_', ' ')[:25] for s in means.index], fontsize=8)
ax1.set_xlabel('Mean Value', fontsize=11)
ax1.set_title('(a) Sensor Mean Values', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Sensor standard deviations
ax2 = axes[0, 1]
stds = df[sensor_cols].std().sort_values(ascending=True)
ax2.barh(range(len(stds)), stds.values, color='coral', alpha=0.8, edgecolor='black')
ax2.set_yticks(range(len(stds)))
ax2.set_yticklabels([s.replace('PID_', '').replace('_', ' ')[:25] for s in stds.index], fontsize=8)
ax2.set_xlabel('Standard Deviation', fontsize=11)
ax2.set_title('(b) Sensor Variability', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Sensor distributions (boxplot)
ax3 = axes[1, 0]
df[sensor_cols[:7]].boxplot(ax=ax3, rot=45)
ax3.set_title('(c) Sensor Distributions (Part 1)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Value', fontsize=11)

# Plot 4: Sensor distributions (boxplot continued)
ax4 = axes[1, 1]
df[sensor_cols[7:]].boxplot(ax=ax4, rot=45)
ax4.set_title('(d) Sensor Distributions (Part 2)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Value', fontsize=11)

# Plot 5: Sensor means by phase (heatmap)
ax5 = axes[2, 0]
phase_sensor_means_norm = (phase_sensor_means - phase_sensor_means.min()) / (phase_sensor_means.max() - phase_sensor_means.min())
sns.heatmap(phase_sensor_means_norm.T, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax5, cbar_kws={"shrink": .8})
ax5.set_title('(e) Normalized Sensor Means by Phase', fontsize=12, fontweight='bold')
ax5.set_xlabel('Degradation Phase', fontsize=11)

# Plot 6: Sensor correlation matrix
ax6 = axes[2, 1]
corr = df[sensor_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=ax6, square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 7})
ax6.set_title('(f) Sensor Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig02_sensor_statistics.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig02_sensor_statistics.png")

# =============================================================================
# 4. DEGRADATION TRAJECTORIES
# =============================================================================
print("\n" + "=" * 80)
print("4. GENERATING DEGRADATION TRAJECTORIES FIGURE...")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle('Synthetic OBD-II RUL Dataset: Degradation Trajectories',
             fontsize=14, fontweight='bold', y=1.01)

sample_units = [1, 10, 25, 50, 75]
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_units)))

# Plot 1: Degradation factor curves
ax1 = axes[0, 0]
for i, uid in enumerate(sample_units):
    df_u = df[df['unit_number'] == uid]
    ax1.plot(df_u['km_driven'], df_u['degradation_factor'], color=colors[i], alpha=0.8, linewidth=2, label=f'Unit {uid}')
ax1.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5, label='Wear onset (0.3)')
ax1.set_xlabel('KM Driven', fontsize=11)
ax1.set_ylabel('Degradation Factor', fontsize=11)
ax1.set_title('(a) Degradation Factor Progression', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Coolant temperature degradation
ax2 = axes[0, 1]
for i, uid in enumerate(sample_units):
    df_u = df[df['unit_number'] == uid]
    ax2.plot(df_u['km_driven'], df_u['PID_05_COOLANT_TEMPERATURE'], color=colors[i], alpha=0.7, linewidth=1.5)
ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Healthy baseline')
ax2.axhline(y=115, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
ax2.set_xlabel('KM Driven', fontsize=11)
ax2.set_ylabel('Coolant Temp (°C)', fontsize=11)
ax2.set_title('(b) Coolant Temperature Degradation', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Fuel trim degradation
ax3 = axes[1, 0]
for i, uid in enumerate(sample_units):
    df_u = df[df['unit_number'] == uid]
    ax3.plot(df_u['km_driven'], df_u['PID_44_LONG_TERM_FUEL_TRIM_BANK_1'], color=colors[i], alpha=0.7, linewidth=1.5)
ax3.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Healthy baseline')
ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
ax3.set_xlabel('KM Driven', fontsize=11)
ax3.set_ylabel('Fuel Trim (%)', fontsize=11)
ax3.set_title('(c) Fuel Trim Degradation', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Engine RPM degradation
ax4 = axes[1, 1]
for i, uid in enumerate(sample_units):
    df_u = df[df['unit_number'] == uid]
    ax4.plot(df_u['km_driven'], df_u['PID_0C_ENGINE_RPM'], color=colors[i], alpha=0.7, linewidth=1.5)
ax4.axhline(y=2500, color='green', linestyle='--', alpha=0.5, label='Healthy baseline')
ax4.axhline(y=2000, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
ax4.set_xlabel('KM Driven', fontsize=11)
ax4.set_ylabel('Engine RPM', fontsize=11)
ax4.set_title('(d) Engine RPM Degradation', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: MAF airflow degradation
ax5 = axes[2, 0]
for i, uid in enumerate(sample_units):
    df_u = df[df['unit_number'] == uid]
    ax5.plot(df_u['km_driven'], df_u['PID_10_MAF_AIRFLOW'], color=colors[i], alpha=0.7, linewidth=1.5)
ax5.axhline(y=18, color='green', linestyle='--', alpha=0.5, label='Healthy baseline')
ax5.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
ax5.set_xlabel('KM Driven', fontsize=11)
ax5.set_ylabel('MAF Airflow (g/s)', fontsize=11)
ax5.set_title('(e) MAF Airflow Degradation', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Oil temperature degradation
ax6 = axes[2, 1]
for i, uid in enumerate(sample_units):
    df_u = df[df['unit_number'] == uid]
    ax6.plot(df_u['km_driven'], df_u['PID_5C_OIL_TEMPERATURE'], color=colors[i], alpha=0.7, linewidth=1.5)
ax6.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Healthy baseline')
ax6.axhline(y=140, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
ax6.set_xlabel('KM Driven', fontsize=11)
ax6.set_ylabel('Oil Temp (°C)', fontsize=11)
ax6.set_title('(f) Oil Temperature Degradation', fontsize=12, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig03_degradation_trajectories.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig03_degradation_trajectories.png")

# =============================================================================
# 5. OPERATIONAL MODE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("5. GENERATING OPERATIONAL MODE ANALYSIS FIGURE...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Synthetic OBD-II RUL Dataset: Operational Mode Analysis',
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1: Mode distribution
ax1 = axes[0, 0]
mode_counts = df['operational_mode'].value_counts()
colors_mode = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
ax1.bar(mode_counts.index, mode_counts.values, color=colors_mode, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('(a) Operational Mode Distribution', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Sensor means by mode
ax2 = axes[0, 1]
mode_sensor_means = df.groupby('operational_mode')[['PID_05_COOLANT_TEMPERATURE', 'PID_0C_ENGINE_RPM',
                                                     'PID_04_ENGINE_LOAD', 'PID_10_MAF_AIRFLOW']].mean()
mode_sensor_means.plot(kind='bar', ax=ax2, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Mean Value', fontsize=11)
ax2.set_title('(b) Key Sensors by Operational Mode', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: RUL distribution by mode
ax3 = axes[1, 0]
for mode in OP_MODES:
    mode_data = df[df['operational_mode'] == mode]['RUL']
    ax3.hist(mode_data, bins=30, alpha=0.5, label=mode, density=True)
ax3.set_xlabel('RUL (km)', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('(c) RUL Distribution by Mode', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Sensor variance by mode
ax4 = axes[1, 1]
mode_sensor_std = df.groupby('operational_mode')[sensor_cols].std().mean(axis=1)
ax4.bar(mode_sensor_std.index, mode_sensor_std.values, color=colors_mode, alpha=0.8, edgecolor='black')
ax4.set_ylabel('Mean Sensor Std Dev', fontsize=11)
ax4.set_title('(d) Average Sensor Noise by Mode', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig04_operational_modes.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig04_operational_modes.png")

# =============================================================================
# 6. SUMMARY EXPORT
# =============================================================================
print("\n" + "=" * 80)
print("6. EXPORTING SUMMARY STATISTICS...")
print("=" * 80)

stats_summary = {
    'Metric': [
        'Total Rows', 'Total Units', 'Mean Lifetime (km)', 'Min Lifetime (km)',
        'Max Lifetime (km)', 'Mean Samples/Unit', 'RUL Cap (km)', 'RUL Min (km)',
        'RUL Max (km)', 'Mean RUL (km)', 'Healthy Phase %', 'Wear Onset %',
        'Accelerated %', 'City Mode %', 'Highway Mode %', 'Idle Mode %', 'Aggressive Mode %'
    ],
    'Value': [
        len(df), df['unit_number'].nunique(),
        df.groupby('unit_number')['total_lifetime_km'].first().mean(),
        df.groupby('unit_number')['total_lifetime_km'].first().min(),
        df.groupby('unit_number')['total_lifetime_km'].first().max(),
        len(df) / df['unit_number'].nunique(),
        5000, df['RUL'].min(), df['RUL'].max(), df['RUL'].mean(),
        (df['phase'] == 'Healthy').sum() / len(df) * 100,
        (df['phase'] == 'Wear Onset').sum() / len(df) * 100,
        (df['phase'] == 'Accelerated').sum() / len(df) * 100,
        (df['operational_mode'] == 'city').sum() / len(df) * 100,
        (df['operational_mode'] == 'highway').sum() / len(df) * 100,
        (df['operational_mode'] == 'idle').sum() / len(df) * 100,
        (df['operational_mode'] == 'aggressive').sum() / len(df) * 100,
    ]
}
stats_df = pd.DataFrame(stats_summary)
stats_df.to_csv(OUTPUT_DIR / "dataset_analysis_summary.csv", index=False)
print("  ✓ dataset_analysis_summary.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nAll figures saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")

print(f"\nKEY FINDINGS:")
print(f"  - Total rows: {len(df):,} across {df['unit_number'].nunique()} units")
print(f"  - Mean lifetime: {df.groupby('unit_number')['total_lifetime_km'].first().mean():.0f} km")
print(f"  - RUL range: {df['RUL'].min():.0f} - {df['RUL'].max():.0f} km")
print(f"  - Healthy phase: {(df['phase'] == 'Healthy').sum() / len(df) * 100:.1f}%")
print(f"  - Coolant ↔ Oil Temp correlation: {df['PID_05_COOLANT_TEMPERATURE'].corr(df['PID_5C_OIL_TEMPERATURE']):.3f}")
print(f"  - Load ↔ Fuel Trim correlation: {df['PID_04_ENGINE_LOAD'].corr(df['PID_44_LONG_TERM_FUEL_TRIM_BANK_1']):.3f}")

if __name__ == "__main__":
    pass
