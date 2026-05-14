#!/usr/bin/env python3
"""
Synthetic OBD-II RUL Dataset Generator
======================================
Generates unit-level degradation curves with continuous RUL labels
for automotive predictive maintenance model training.

Usage: python generate_synthetic_rul.py
Output: synthetic_obd_rul_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_UNITS = 100                    # Number of virtual vehicles
MIN_LIFETIME_KM = 6000           # Minimum vehicle lifetime (km)
MAX_LIFETIME_KM = 12000          # Maximum vehicle lifetime (km)
SAMPLING_INTERVAL_KM = 10        # One row per 10 km
RUL_CAP_KM = 3000                # Piecewise linear RUL cap (km)

# Degradation phases (as fraction of lifetime)
HEALTHY_END = 0.30               # Healthy phase ends at 30% of life
WEAR_END = 0.60                  # Wear onset ends at 60% of life

# Operational condition profiles
OP_MODES = ['idle', 'city', 'highway', 'aggressive']
OP_MODE_WEIGHTS = [0.15, 0.40, 0.35, 0.10]

# =============================================================================
# SENSOR CONFIGURATION: 14 OBD-II PIDs
# =============================================================================
SENSOR_CONFIG = {
    'coolant_temp': {
        'name': 'PID_05_COOLANT_TEMPERATURE',
        'unit': '°C',
        'healthy_mean': 90.0,
        'healthy_std': 3.0,
        'degradation_drift_per_1000km': 2.5,
        'critical_threshold': 115.0,
        'min_possible': 70.0,
        'max_possible': 130.0,
    },
    'engine_rpm': {
        'name': 'PID_0C_ENGINE_RPM',
        'unit': 'rpm',
        'healthy_mean': 2500.0,
        'healthy_std': 150.0,
        'degradation_drift_per_1000km': -15.0,
        'critical_threshold': 2000.0,
        'min_possible': 600.0,
        'max_possible': 6000.0,
    },
    'engine_load': {
        'name': 'PID_04_ENGINE_LOAD',
        'unit': '%',
        'healthy_mean': 35.0,
        'healthy_std': 8.0,
        'degradation_drift_per_1000km': 4.0,
        'critical_threshold': 85.0,
        'min_possible': 5.0,
        'max_possible': 100.0,
    },
    'maf_airflow': {
        'name': 'PID_10_MAF_AIRFLOW',
        'unit': 'g/s',
        'healthy_mean': 18.0,
        'healthy_std': 4.0,
        'degradation_drift_per_1000km': -1.2,
        'critical_threshold': 10.0,
        'min_possible': 2.0,
        'max_possible': 200.0,
    },
    'fuel_trim_long': {
        'name': 'PID_44_LONG_TERM_FUEL_TRIM_BANK_1',
        'unit': '%',
        'healthy_mean': 0.0,
        'healthy_std': 1.5,
        'degradation_drift_per_1000km': 2.0,
        'critical_threshold': 20.0,
        'min_possible': -30.0,
        'max_possible': 30.0,
    },
    'intake_air_temp': {
        'name': 'PID_0F_INTAKE_AIR_TEMPERATURE',
        'unit': '°C',
        'healthy_mean': 30.0,
        'healthy_std': 5.0,
        'degradation_drift_per_1000km': 0.3,
        'critical_threshold': 55.0,
        'min_possible': -10.0,
        'max_possible': 80.0,
    },
    'throttle_position': {
        'name': 'PID_11_THROTTLE_POSITION',
        'unit': '%',
        'healthy_mean': 25.0,
        'healthy_std': 10.0,
        'degradation_drift_per_1000km': 1.5,
        'critical_threshold': 80.0,
        'min_possible': 0.0,
        'max_possible': 100.0,
    },
    'o2_voltage': {
        'name': 'PID_13_O2_SENSOR_VOLTAGE_BANK_1_SENSOR_1',
        'unit': 'V',
        'healthy_mean': 0.45,
        'healthy_std': 0.15,
        'degradation_drift_per_1000km': -0.02,
        'critical_threshold': 0.1,
        'min_possible': 0.0,
        'max_possible': 1.2,
    },
    'absolute_load': {
        'name': 'PID_33_ABSOLUTE_LOAD',
        'unit': '%',
        'healthy_mean': 40.0,
        'healthy_std': 12.0,
        'degradation_drift_per_1000km': 3.5,
        'critical_threshold': 90.0,
        'min_possible': 0.0,
        'max_possible': 100.0,
    },
    'control_voltage': {
        'name': 'PID_42_CONTROL_MODULE_VOLTAGE',
        'unit': 'V',
        'healthy_mean': 13.5,
        'healthy_std': 0.3,
        'degradation_drift_per_1000km': -0.15,
        'critical_threshold': 11.5,
        'min_possible': 10.0,
        'max_possible': 16.0,
    },
    'oil_temp': {
        'name': 'PID_5C_OIL_TEMPERATURE',
        'unit': '°C',
        'healthy_mean': 100.0,
        'healthy_std': 5.0,
        'degradation_drift_per_1000km': 2.0,
        'critical_threshold': 140.0,
        'min_possible': 60.0,
        'max_possible': 160.0,
    },
    'fuel_rate': {
        'name': 'PID_5E_FUEL_RATE',
        'unit': 'L/h',
        'healthy_mean': 8.0,
        'healthy_std': 2.5,
        'degradation_drift_per_1000km': 0.8,
        'critical_threshold': 20.0,
        'min_possible': 0.5,
        'max_possible': 50.0,
    },
    'vehicle_speed': {
        'name': 'PID_0D_VEHICLE_SPEED',
        'unit': 'km/h',
        'healthy_mean': 60.0,
        'healthy_std': 25.0,
        'degradation_drift_per_1000km': 0.0,
        'critical_threshold': 200.0,
        'min_possible': 0.0,
        'max_possible': 250.0,
    },
    'o2_b1s2': {
        'name': 'PID_14_O2_SENSOR_VOLTAGE_BANK_1_SENSOR_2',
        'unit': 'V',
        'healthy_mean': 0.50,
        'healthy_std': 0.12,
        'degradation_drift_per_1000km': -0.015,
        'critical_threshold': 0.15,
        'min_possible': 0.0,
        'max_possible': 1.2,
    },
}

# =============================================================================
# DEGRADATION MODEL
# =============================================================================

def generate_degradation_curve(total_km, healthy_end_km, wear_end_km, seed=None):
    """Generate degradation factor curve (0.0=healthy, 1.0=critical)."""
    if seed is not None:
        np.random.seed(seed)

    km_points = np.arange(0, total_km + SAMPLING_INTERVAL_KM, SAMPLING_INTERVAL_KM)
    n_points = len(km_points)

    degradation = np.zeros(n_points)

    # Phase 1: Healthy (flat)
    # Phase 2: Wear onset (linear, up to 0.3)
    wear_mask = (km_points > healthy_end_km) & (km_points <= wear_end_km)
    wear_progress = (km_points[wear_mask] - healthy_end_km) / (wear_end_km - healthy_end_km)
    degradation[wear_mask] = wear_progress * 0.3

    # Phase 3: Accelerated (non-linear to 1.0)
    accel_mask = km_points > wear_end_km
    accel_progress = (km_points[accel_mask] - wear_end_km) / (total_km - wear_end_km)
    degradation[accel_mask] = 0.3 + 0.7 * (accel_progress ** 1.5)

    # Add small noise
    degradation = np.clip(degradation + np.random.normal(0, 0.02, n_points), 0.0, 1.0)

    return km_points, degradation


def get_operational_mode(seed=None):
    """Randomly select driving mode."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(OP_MODES, p=OP_MODE_WEIGHTS)


def get_mode_noise_multiplier(mode):
    """Noise scaling per driving mode."""
    return {'idle': 0.5, 'city': 1.0, 'highway': 0.8, 'aggressive': 1.5}.get(mode, 1.0)


def generate_sensor_value(cfg, degradation_factor, mode, total_km, seed=None):
    """Generate single sensor reading with degradation applied."""
    if seed is not None:
        np.random.seed(seed)

    mode_mult = get_mode_noise_multiplier(mode)
    mode_offsets = {'idle': -0.1, 'city': 0.0, 'highway': 0.05, 'aggressive': 0.15}
    baseline = cfg['healthy_mean'] * (1 + mode_offsets.get(mode, 0))

    total_drift = cfg['degradation_drift_per_1000km'] * (total_km / 1000)
    actual_drift = degradation_factor * total_drift * (1 + 0.3 * degradation_factor ** 2)

    noise_factor = 1.0 + degradation_factor * 0.5
    noise_std = cfg['healthy_std'] * mode_mult * noise_factor
    noise = np.random.normal(0, noise_std)

    value = baseline + actual_drift + noise
    value = np.clip(value, cfg['min_possible'], cfg['max_possible'])
    
    if np.random.random() < 0.01:
        value = np.nan
        
    return value


# =============================================================================
# CROSS-SENSOR COUPLING
# =============================================================================

def apply_cross_sensor_coupling(df, degradation):
    """Apply physically realistic cross-sensor correlations."""
    df = df.copy()

    # Coupling 1: Compression loss -> load, MAF, fuel trim
    load_dev = (df['PID_04_ENGINE_LOAD'] - 35.0) / 35.0
    df['PID_10_MAF_AIRFLOW'] = np.clip(
        df['PID_10_MAF_AIRFLOW'] - load_dev * 0.5 * degradation * df['PID_10_MAF_AIRFLOW'],
        SENSOR_CONFIG['maf_airflow']['min_possible'], SENSOR_CONFIG['maf_airflow']['max_possible'])
    df['PID_44_LONG_TERM_FUEL_TRIM_BANK_1'] = np.clip(
        df['PID_44_LONG_TERM_FUEL_TRIM_BANK_1'] + load_dev * 1.5 * degradation * 5.0,
        SENSOR_CONFIG['fuel_trim_long']['min_possible'], SENSOR_CONFIG['fuel_trim_long']['max_possible'])

    # Coupling 2: Cooling degradation -> oil temp, intake temp
    coolant_dev = (df['PID_05_COOLANT_TEMPERATURE'] - 90.0) / 90.0
    df['PID_5C_OIL_TEMPERATURE'] = np.clip(
        df['PID_5C_OIL_TEMPERATURE'] + coolant_dev * 2.0 * degradation + np.random.normal(0, 1.5),
        SENSOR_CONFIG['oil_temp']['min_possible'], SENSOR_CONFIG['oil_temp']['max_possible'])
    df['PID_0F_INTAKE_AIR_TEMPERATURE'] = np.clip(
        df['PID_0F_INTAKE_AIR_TEMPERATURE'] + coolant_dev * 1.5 * degradation,
        SENSOR_CONFIG['intake_air_temp']['min_possible'], SENSOR_CONFIG['intake_air_temp']['max_possible'])

    # Coupling 3: Electrical stress -> voltage drop
    elec_stress = (load_dev + coolant_dev) * 0.5
    df['PID_42_CONTROL_MODULE_VOLTAGE'] = np.clip(
        df['PID_42_CONTROL_MODULE_VOLTAGE'] - elec_stress * 0.2 * degradation,
        SENSOR_CONFIG['control_voltage']['min_possible'], SENSOR_CONFIG['control_voltage']['max_possible'])

    # Coupling 4: O2 sensors correlation
    o2_diff = df['PID_13_O2_SENSOR_VOLTAGE_BANK_1_SENSOR_1'] - 0.45
    df['PID_14_O2_SENSOR_VOLTAGE_BANK_1_SENSOR_2'] = np.clip(
        df['PID_14_O2_SENSOR_VOLTAGE_BANK_1_SENSOR_2'] + o2_diff * 0.3 * degradation,
        SENSOR_CONFIG['o2_b1s2']['min_possible'], SENSOR_CONFIG['o2_b1s2']['max_possible'])

    # Coupling 5: Fuel trim -> fuel rate
    trim_dev = df['PID_44_LONG_TERM_FUEL_TRIM_BANK_1'] / 100.0
    df['PID_5E_FUEL_RATE'] = np.clip(
        df['PID_5E_FUEL_RATE'] + trim_dev * 2.0 * degradation,
        SENSOR_CONFIG['fuel_rate']['min_possible'], SENSOR_CONFIG['fuel_rate']['max_possible'])

    return df


# =============================================================================
# UNIT GENERATION
# =============================================================================

def generate_single_unit(unit_id):
    """Generate complete degradation trajectory for one virtual vehicle."""
    np.random.seed(RANDOM_SEED + unit_id)

    total_km = np.random.randint(MIN_LIFETIME_KM, MAX_LIFETIME_KM + 1)
    total_km = (total_km // SAMPLING_INTERVAL_KM) * SAMPLING_INTERVAL_KM

    healthy_end_km = int(total_km * HEALTHY_END)
    wear_end_km = int(total_km * WEAR_END)

    km_points, degradation = generate_degradation_curve(
        total_km, healthy_end_km, wear_end_km, seed=RANDOM_SEED + unit_id)

    # RUL labels (piecewise linear)
    remaining = total_km - km_points
    rul = np.where(remaining > RUL_CAP_KM, RUL_CAP_KM, remaining)

    # Operational modes
    modes = np.array([get_operational_mode(seed=RANDOM_SEED + unit_id + i) for i in range(len(km_points))])

    # Build record
    record = {
        'unit_number': np.full(len(km_points), unit_id),
        'time_cycles': np.arange(1, len(km_points) + 1),
        'km_driven': km_points,
        'total_lifetime_km': np.full(len(km_points), total_km),
        'operational_mode': modes,
        'degradation_factor': degradation,
        'RUL': rul,
    }

    # Generate sensors
    for sensor_key, cfg in SENSOR_CONFIG.items():
        values = np.array([
            generate_sensor_value(cfg, degradation[i], modes[i], total_km, 
                                 seed=RANDOM_SEED + unit_id + i)
            for i in range(len(km_points))
        ])
        record[cfg['name']] = values

    df = pd.DataFrame(record)

    # Apply cross-sensor coupling
    for i in range(len(df)):
        df.iloc[i] = apply_cross_sensor_coupling(df.iloc[i:i+1], df.iloc[i]['degradation_factor']).iloc[0]

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SYNTHETIC OBD-II RUL DATASET GENERATOR")
    print("=" * 70)

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating {N_UNITS} virtual vehicles...")
    all_units = []
    for unit_id in range(1, N_UNITS + 1):
        unit_df = generate_single_unit(unit_id)
        all_units.append(unit_df)
        if unit_id % 20 == 0:
            print(f"  Generated {unit_id}/{N_UNITS} units...")

    full_dataset = pd.concat(all_units, ignore_index=True)

    # Save
    output_path = output_dir / "synthetic_obd_rul_dataset.csv"
    full_dataset.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total rows: {len(full_dataset):,}")
    print(f"  Total units: {full_dataset['unit_number'].nunique()}")
    print(f"  Mean lifetime: {full_dataset.groupby('unit_number')['total_lifetime_km'].first().mean():.0f} km")
    print(f"  RUL range: {full_dataset['RUL'].min():.0f} - {full_dataset['RUL'].max():.0f} km")
    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()
