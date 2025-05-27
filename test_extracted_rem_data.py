#!/usr/bin/env python3
"""
Test and validation script for extracted_REM_windows.npz file.

This script performs comprehensive validation of the extracted REM training data
to ensure it's correctly formatted and ready for 1D CNN training.

Author: Benjamin Grayzel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os

def load_and_inspect_data(filepath="extracted_REM_windows.npz"):
    """Load and perform basic inspection of the extracted data."""
    print("="*60)
    print("LOADING AND INSPECTING EXTRACTED REM DATA")
    print("="*60)
    
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} not found!")
        return None
    
    # Load data
    data = np.load(filepath)
    
    print(f"File size: {os.path.getsize(filepath) / (1024**2):.1f} MB")
    print(f"Available keys: {list(data.keys())}")
    print()
    
    # Extract arrays
    windows = data['windows']
    labels = data['labels']
    sampling_rate = data['sampling_rate']
    window_length_sec = data['window_length_sec']
    label_check_sec = data['label_check_sec']
    overlap_sec = data['overlap_sec']
    
    print("METADATA:")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Window length: {window_length_sec}s")
    print(f"  Label check period: {label_check_sec}s")
    print(f"  Overlap: {overlap_sec}s")
    print()
    
    print("DATA SHAPES:")
    print(f"  Windows shape: {windows.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Expected samples per window: {int(sampling_rate * window_length_sec)}")
    print()
    
    return {
        'windows': windows,
        'labels': labels,
        'sampling_rate': sampling_rate,
        'window_length_sec': window_length_sec,
        'label_check_sec': label_check_sec,
        'overlap_sec': overlap_sec
    }

def validate_data_integrity(data_dict):
    """Validate data integrity and consistency."""
    print("="*60)
    print("DATA INTEGRITY VALIDATION")
    print("="*60)
    
    windows = data_dict['windows']
    labels = data_dict['labels']
    sampling_rate = data_dict['sampling_rate']
    window_length_sec = data_dict['window_length_sec']
    
    # Check dimensions
    n_windows, n_samples, n_channels = windows.shape
    expected_samples = int(sampling_rate * window_length_sec)
    
    print("DIMENSION CHECKS:")
    print(f"  ✓ Number of windows: {n_windows}")
    print(f"  ✓ Samples per window: {n_samples} (expected: {expected_samples})")
    print(f"  ✓ Number of channels: {n_channels}")
    print(f"  ✓ Labels length matches windows: {len(labels) == n_windows}")
    
    # Check data types
    print(f"\nDATA TYPES:")
    print(f"  ✓ Windows dtype: {windows.dtype}")
    print(f"  ✓ Labels dtype: {labels.dtype}")
    
    # Check for NaN/Inf values
    nan_count = np.sum(np.isnan(windows))
    inf_count = np.sum(np.isinf(windows))
    print(f"\nDATA QUALITY:")
    print(f"  ✓ NaN values in windows: {nan_count}")
    print(f"  ✓ Inf values in windows: {inf_count}")
    
    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nLABEL DISTRIBUTION:")
    for label, count in zip(unique_labels, counts):
        label_name = "REM" if label == 1 else "Non-REM"
        percentage = (count / len(labels)) * 100
        print(f"  ✓ {label_name} (label {label}): {count} windows ({percentage:.1f}%)")
    
    # Check data ranges
    print(f"\nDATA RANGES:")
    print(f"  ✓ Windows min: {np.min(windows):.2f}")
    print(f"  ✓ Windows max: {np.max(windows):.2f}")
    print(f"  ✓ Windows mean: {np.mean(windows):.2f}")
    print(f"  ✓ Windows std: {np.std(windows):.2f}")
    
    return True

def analyze_signal_characteristics(data_dict):
    """Analyze signal characteristics and preprocessing effects."""
    print("="*60)
    print("SIGNAL CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    windows = data_dict['windows']
    labels = data_dict['labels']
    sampling_rate = data_dict['sampling_rate']
    
    # Separate REM and non-REM windows
    rem_indices = labels == 1
    nonrem_indices = labels == 0
    
    rem_windows = windows[rem_indices]
    nonrem_windows = windows[nonrem_indices]
    
    print(f"REM windows: {len(rem_windows)}")
    print(f"Non-REM windows: {len(nonrem_windows)}")
    print()
    
    # Analyze each channel
    channel_names = ['LF-FpZ', 'OTE_L-FpZ', 'RF-FpZ', 'OTE_R-FpZ']
    
    for ch in range(windows.shape[2]):
        print(f"CHANNEL {ch} ({channel_names[ch]}):")
        
        # Overall statistics
        ch_data = windows[:, :, ch]
        print(f"  Overall - Mean: {np.mean(ch_data):.2f}, Std: {np.std(ch_data):.2f}")
        
        # REM vs Non-REM comparison
        if len(rem_windows) > 0:
            rem_ch = rem_windows[:, :, ch]
            rem_mean = np.mean(rem_ch)
            rem_std = np.std(rem_ch)
            print(f"  REM - Mean: {rem_mean:.2f}, Std: {rem_std:.2f}")
        
        if len(nonrem_windows) > 0:
            nonrem_ch = nonrem_windows[:, :, ch]
            nonrem_mean = np.mean(nonrem_ch)
            nonrem_std = np.std(nonrem_ch)
            print(f"  Non-REM - Mean: {nonrem_mean:.2f}, Std: {nonrem_std:.2f}")
        
        # Statistical test (if both classes exist)
        if len(rem_windows) > 0 and len(nonrem_windows) > 0:
            # Use sample of data for efficiency
            rem_sample = rem_ch.flatten()[:10000]
            nonrem_sample = nonrem_ch.flatten()[:10000]
            
            t_stat, p_value = stats.ttest_ind(rem_sample, nonrem_sample)
            print(f"  T-test p-value: {p_value:.2e} {'(significant)' if p_value < 0.05 else '(not significant)'}")
        
        print()

def create_visualization_plots(data_dict, save_plots=True):
    """Create visualization plots for data validation."""
    print("="*60)
    print("CREATING VISUALIZATION PLOTS")
    print("="*60)
    
    windows = data_dict['windows']
    labels = data_dict['labels']
    sampling_rate = data_dict['sampling_rate']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Extracted REM Data Validation Plots', fontsize=16)
    
    # 1. Label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    ax = axes[0, 0]
    bars = ax.bar(['Non-REM', 'REM'], counts, color=['skyblue', 'orange'])
    ax.set_title('Label Distribution')
    ax.set_ylabel('Number of Windows')
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = (count / len(labels)) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    # 2. Sample REM window
    ax = axes[0, 1]
    if np.sum(labels) > 0:
        rem_idx = np.where(labels == 1)[0][0]
        time_axis = np.arange(windows.shape[1]) / sampling_rate
        for ch in range(windows.shape[2]):
            ax.plot(time_axis, windows[rem_idx, :, ch], label=f'Ch{ch}', alpha=0.7)
        ax.set_title('Sample REM Window')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Sample Non-REM window
    ax = axes[0, 2]
    if np.sum(labels == 0) > 0:
        nonrem_idx = np.where(labels == 0)[0][0]
        time_axis = np.arange(windows.shape[1]) / sampling_rate
        for ch in range(windows.shape[2]):
            ax.plot(time_axis, windows[nonrem_idx, :, ch], label=f'Ch{ch}', alpha=0.7)
        ax.set_title('Sample Non-REM Window')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Data distribution histogram
    ax = axes[1, 0]
    sample_data = windows[:1000, :, :].flatten()  # Sample for efficiency
    ax.hist(sample_data, bins=50, alpha=0.7, density=True)
    ax.set_title('Data Distribution (Sample)')
    ax.set_xlabel('Amplitude (μV)')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # 5. Channel-wise statistics
    ax = axes[1, 1]
    channel_means = [np.mean(windows[:, :, ch]) for ch in range(windows.shape[2])]
    channel_stds = [np.std(windows[:, :, ch]) for ch in range(windows.shape[2])]
    
    x_pos = np.arange(len(channel_means))
    ax.bar(x_pos - 0.2, channel_means, 0.4, label='Mean', alpha=0.7)
    ax.bar(x_pos + 0.2, channel_stds, 0.4, label='Std', alpha=0.7)
    ax.set_title('Channel-wise Statistics')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Ch{i}' for i in range(windows.shape[2])])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Window index vs REM labels
    ax = axes[1, 2]
    window_indices = np.arange(len(labels))
    rem_positions = window_indices[labels == 1]
    ax.scatter(rem_positions, np.ones(len(rem_positions)), 
              alpha=0.6, s=1, c='orange', label='REM')
    ax.set_title('REM Window Distribution Over Time')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('REM Label')
    ax.set_ylim(0.5, 1.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('rem_data_validation_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Plots saved as 'rem_data_validation_plots.png'")
    
    plt.show()

def check_data_readiness_for_training(data_dict):
    """Check if data is ready for CNN training."""
    print("="*60)
    print("TRAINING READINESS CHECK")
    print("="*60)
    
    windows = data_dict['windows']
    labels = data_dict['labels']
    
    checks_passed = []
    
    # Check 1: Sufficient data
    min_samples = 1000
    sufficient_data = len(windows) >= min_samples
    checks_passed.append(sufficient_data)
    print(f"✓ Sufficient data (≥{min_samples}): {len(windows)} windows - {'PASS' if sufficient_data else 'FAIL'}")
    
    # Check 2: Balanced classes (at least some REM examples)
    rem_count = np.sum(labels)
    min_rem = 50
    sufficient_rem = rem_count >= min_rem
    checks_passed.append(sufficient_rem)
    print(f"✓ Sufficient REM examples (≥{min_rem}): {rem_count} - {'PASS' if sufficient_rem else 'FAIL'}")
    
    # Check 3: No NaN/Inf values
    no_invalid = not (np.any(np.isnan(windows)) or np.any(np.isinf(windows)))
    checks_passed.append(no_invalid)
    print(f"✓ No invalid values: {'PASS' if no_invalid else 'FAIL'}")
    
    # Check 4: Reasonable data range
    data_range = np.max(windows) - np.min(windows)
    reasonable_range = 0.1 < data_range < 10000
    checks_passed.append(reasonable_range)
    print(f"✓ Reasonable data range ({data_range:.1f}): {'PASS' if reasonable_range else 'FAIL'}")
    
    # Check 5: Consistent shapes
    expected_samples = int(data_dict['sampling_rate'] * data_dict['window_length_sec'])
    correct_shape = windows.shape[1] == expected_samples and windows.shape[2] == 4
    checks_passed.append(correct_shape)
    print(f"✓ Correct window shape: {'PASS' if correct_shape else 'FAIL'}")
    
    # Overall assessment
    all_passed = all(checks_passed)
    print(f"\nOVERALL TRAINING READINESS: {'✓ READY' if all_passed else '✗ NOT READY'}")
    
    if all_passed:
        print("\nRECOMMENDATIONS:")
        print("• Data is ready for 1D CNN training")
        print("• Consider data augmentation for REM class due to imbalance")
        print("• Use class weights or resampling techniques")
        print("• Split data into train/validation/test sets")
    else:
        print("\nISSUES TO ADDRESS:")
        if not sufficient_data:
            print("• Need more training data")
        if not sufficient_rem:
            print("• Need more REM examples - consider data augmentation")
        if not no_invalid:
            print("• Clean up NaN/Inf values in the data")
        if not reasonable_range:
            print("• Check preprocessing - data range seems unusual")
        if not correct_shape:
            print("• Fix data shape inconsistencies")
    
    return all_passed

def generate_summary_report(data_dict):
    """Generate a summary report."""
    print("="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    windows = data_dict['windows']
    labels = data_dict['labels']
    
    total_duration_hours = (len(windows) * data_dict['window_length_sec'] * 
                           (1 - data_dict['overlap_sec'] / data_dict['window_length_sec'])) / 3600
    
    rem_windows = np.sum(labels)
    rem_duration_hours = (rem_windows * data_dict['window_length_sec']) / 3600
    
    print(f"Dataset Overview:")
    print(f"  • Total windows: {len(windows):,}")
    print(f"  • Total duration: ~{total_duration_hours:.1f} hours")
    print(f"  • REM windows: {rem_windows:,} ({rem_windows/len(windows)*100:.1f}%)")
    print(f"  • REM duration: ~{rem_duration_hours:.1f} hours")
    print(f"  • Non-REM windows: {len(windows)-rem_windows:,}")
    print(f"  • Window size: {windows.shape[1]} samples ({data_dict['window_length_sec']}s)")
    print(f"  • Channels: {windows.shape[2]} (EEG)")
    print(f"  • Data type: {windows.dtype}")
    print(f"  • Memory usage: ~{windows.nbytes / (1024**2):.1f} MB")
    print()
    print(f"Ready for 1D CNN training with input shape: ({windows.shape[1]}, {windows.shape[2]})")

def main():
    """Main test function."""
    print("REM DATA EXTRACTION VALIDATION TEST")
    print("===================================")
    
    # Load and inspect data
    data_dict = load_and_inspect_data()
    if data_dict is None:
        return
    
    # Run validation tests
    validate_data_integrity(data_dict)
    analyze_signal_characteristics(data_dict)
    create_visualization_plots(data_dict)
    training_ready = check_data_readiness_for_training(data_dict)
    generate_summary_report(data_dict)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    if training_ready:
        print("🎉 Data validation PASSED - Ready for CNN training!")
    else:
        print("⚠️  Data validation FAILED - Issues need to be addressed")

if __name__ == "__main__":
    main()
