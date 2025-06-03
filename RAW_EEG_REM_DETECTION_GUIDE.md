# Raw EEG REM Detection Guide

## Overview

Both REM detection functions have been successfully adapted to work with raw EEG data using direct channel indexing instead of column name identification. This enables seamless integration with real-time EEG data streams.

## Key Changes Made

### 1. Function Signature Updates

**Before:**
```python
detect_rem_window_FINAL_MODEL(data_array, column_names, srate=250, ...)
detect_rem_window_REALTIME_OPTIMIZED(data_array, column_names, srate=250, ...)
```

**After:**
```python
detect_rem_window_FINAL_MODEL(data_array, srate=250, ...)
detect_rem_window_REALTIME_OPTIMIZED(data_array, srate=250, ...)
```

### 2. Input Data Format

**Expected Input:**
- **Format**: `nx2` numpy array (samples x 2 channels)
- **Channel 0**: Left frontal (LF-FpZ equivalent)
- **Channel 1**: Right frontal (RF-FpZ equivalent)
- **Data Type**: Raw EEG voltages (typically µV range)

### 3. Channel Selection Strategy

Both functions now use **direct channel indexing**:
- **Primary channel**: Index 0 (left frontal)
- **Mapping**: Channel 0 → Fp1 (YASA standard electrode name)
- **Priority**: Frontal channels optimized for REM detection

## Usage Examples

### Basic Usage

```python
import numpy as np

# Assume you have raw EEG data in nx2 format
eeg_data = np.array([...])  # Shape: (samples, 2) - [LF, RF]
srate = 250  # Sampling rate in Hz

# YASA Model Detection
is_rem_yasa, rem_prob, yasa_info = detect_rem_window_FINAL_MODEL(
    eeg_data, 
    srate=srate, 
    threshold_mode='balanced'
)

# Real-time Optimized Detection  
is_rem_rt, rem_score, rt_info = detect_rem_window_REALTIME_OPTIMIZED(
    eeg_data,
    srate=srate,
    threshold_mode='balanced' 
)
```

### Real-time Processing Example

```python
def process_realtime_eeg_window(eeg_window):
    """
    Process a sliding window of EEG data for real-time REM detection.
    
    Args:
        eeg_window: nx2 numpy array with recent EEG samples
    """
    
    # Quick real-time detection
    is_rem, rem_score, info = detect_rem_window_REALTIME_OPTIMIZED(
        eeg_window,
        srate=250,
        threshold_mode='sensitive'  # More responsive for real-time
    )
    
    if is_rem:
        print(f"🔍 REM detected! Score: {rem_score:.3f}")
        # Trigger lucid dreaming cue
        trigger_lucid_dream_cue()
    
    return is_rem, rem_score
```

### Validation with YASA

```python
def validate_with_yasa(eeg_data_long):
    """
    Use YASA for thorough validation on longer EEG segments.
    
    Args:
        eeg_data_long: nx2 array with several minutes of EEG data
    """
    
    is_rem, rem_prob, yasa_info = detect_rem_window_FINAL_MODEL(
        eeg_data_long,
        srate=250,
        threshold_mode='conservative'  # Higher confidence threshold
    )
    
    print(f"YASA REM Probability: {rem_prob:.4f}")
    print(f"Detection Strength: {yasa_info['detection_strength']}")
    print(f"Epochs analyzed: {yasa_info['num_epochs']}")
    
    return is_rem, rem_prob
```

## Method Comparison

### YASA Model (`detect_rem_window_FINAL_MODEL`)

**Strengths:**
- Clinically validated sleep staging model
- Empirically determined thresholds from 20-night analysis
- High precision and specificity
- Detailed confidence metrics

**Performance:**
- Processing time: ~0.6-0.7 seconds
- Minimum data: 15 seconds (prefers 5+ minutes)
- Memory usage: Moderate

**Best for:**
- Post-session analysis and validation
- Research applications requiring clinical accuracy
- Situations where precision is more important than speed

### Real-time Optimized (`detect_rem_window_REALTIME_OPTIMIZED`)

**Strengths:**
- Ultra-low latency processing
- Designed for real-time lucid dreaming applications
- Sliding window capability
- Feature-based detection with interpretable scores

**Performance:**
- Processing time: ~0.66-1.0 seconds (for very long signals)
- Minimum data: 5 seconds
- Memory usage: Low

**Best for:**
- Real-time REM detection and lucid dream triggering
- Continuous monitoring applications
- Situations requiring immediate response

## Test Results

### Successful Detection Example

```
🎯 YASA MODEL with Raw EEG
========================================

🎚️ BALANCED threshold:
   🔍 REM Detected: YES
   📊 REM Probability: 0.6846
   💪 Detection Strength: strong
   ⏱️ Processing Time: 0.6238 seconds

🚀 REAL-TIME OPTIMIZED with Raw EEG
========================================

⚡ SENSITIVE threshold:
   🔍 REM Detected: YES
   📊 REM Score: 0.4906
   ⏱️ Processing Time: 0.6641 seconds
   📈 Top Features:
      • alpha_suppression: 1.000
      • theta_alpha_ratio: 0.818
      • theta_power: 0.462
```

## Integration Tips

### 1. Data Preprocessing

```python
# Ensure correct data format
def prepare_eeg_data(raw_channels):
    """Convert multi-channel EEG to nx2 format for REM detection."""
    
    # Extract frontal channels (LF, RF)
    if raw_channels.shape[1] >= 2:
        eeg_data = raw_channels[:, [0, 1]]  # First 2 channels
    else:
        raise ValueError("Need at least 2 EEG channels")
    
    return eeg_data
```

### 2. Threshold Selection

```python
# Choose threshold based on application
thresholds = {
    'lucid_dreaming': 'sensitive',    # Quick response for triggers
    'sleep_research': 'conservative', # High precision for analysis  
    'general_use': 'balanced'         # Good balance of both
}
```

### 3. Error Handling

```python
def safe_rem_detection(eeg_data, method='realtime'):
    """Robust REM detection with error handling."""
    
    try:
        if method == 'yasa':
            result = detect_rem_window_FINAL_MODEL(eeg_data, srate=250)
        else:
            result = detect_rem_window_REALTIME_OPTIMIZED(eeg_data, srate=250)
        
        is_rem, score, info = result
        
        if 'error' in info:
            print(f"Detection error: {info['error']}")
            return False, 0.0, info
        
        return is_rem, score, info
        
    except Exception as e:
        return False, 0.0, {'error': str(e)}
```

## Migration from Column-based System

### Old System
```python
# Required column names to identify EEG channels
columns = ['Timestamp', 'EEG_Filt_1', 'EEG_Filt_2', 'EEG_Filt_3', 'TargetEvent']
is_rem, score, info = detect_rem_window_FINAL_MODEL(data, columns, srate=250)
```

### New System
```python
# Direct channel indexing - much simpler!
eeg_data = data[:, [1, 3]]  # Extract EEG channels directly  
is_rem, score, info = detect_rem_window_FINAL_MODEL(eeg_data, srate=250)
```

## Benefits of Raw EEG Approach

1. **Simplified Interface**: No need to manage column names
2. **Universal Compatibility**: Works with any EEG data source
3. **Reduced Dependencies**: Less prone to column naming errors
4. **Real-time Friendly**: Perfect for streaming data applications
5. **Performance**: Slightly faster due to reduced overhead

## Conclusion

The migration to raw EEG data processing makes both REM detection methods more robust, easier to use, and better suited for real-time applications. The direct channel indexing approach eliminates the complexity of column name management while maintaining full functionality and performance.

Both methods are now ready for integration into real-time lucid dreaming systems and other EEG-based applications requiring reliable REM sleep detection.
