import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# Load the detailed comparison
comparison_df = pd.read_csv('yasa_vs_ground_truth_detailed.csv')

print("=== YASA vs Ground Truth Analysis ===\n")

# Overall results
accuracy = comparison_df['Correct'].mean()
print(f"Overall Accuracy: {accuracy:.1%}")
print(f"Total epochs analyzed: {len(comparison_df)}")
print(f"Recording duration: {len(comparison_df)/120:.1f} hours")
print(f"Average YASA confidence: {comparison_df['YASA_Confidence'].mean():.3f}")

print("\n=== Stage Distribution Comparison ===")
print("\nGround Truth Distribution:")
gt_dist = comparison_df['Ground_Truth'].value_counts()
for stage, count in gt_dist.items():
    print(f"  {stage}: {count} epochs ({count/len(comparison_df):.1%})")

print("\nYASA Prediction Distribution:")
yasa_dist = comparison_df['YASA_Raw'].value_counts()
stage_map = {'W': 'Wake', 'N1': 'Light (N1)', 'N2': 'Light (N2)', 'N3': 'Deep', 'R': 'REM'}
for stage, count in yasa_dist.items():
    print(f"  {stage_map[stage]}: {count} epochs ({count/len(comparison_df):.1%})")

print("\n=== Detailed Performance by Stage ===")

stages = ['Wake', 'Light', 'Deep', 'REM']
for stage in stages:
    stage_data = comparison_df[comparison_df['Ground_Truth'] == stage]
    if len(stage_data) > 0:
        stage_accuracy = stage_data['Correct'].mean()
        avg_confidence = stage_data['YASA_Confidence'].mean()
        
        print(f"\n{stage} Sleep:")
        print(f"  Accuracy: {stage_accuracy:.1%} ({stage_data['Correct'].sum()}/{len(stage_data)})")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  YASA predictions when truth is {stage}:")
        
        yasa_when_true = stage_data['YASA_Raw'].value_counts()
        for yasa_pred, count in yasa_when_true.items():
            pct = count / len(stage_data)
            print(f"    {stage_map[yasa_pred]}: {count} ({pct:.1%})")

print("\n=== Key Findings ===")

# Deep sleep analysis
deep_data = comparison_df[comparison_df['Ground_Truth'] == 'Deep']
deep_as_n2 = (deep_data['YASA_Raw'] == 'N2').sum()
print(f"1. Deep Sleep Under-detection:")
print(f"   - YASA misses {len(deep_data) - deep_data['Correct'].sum()}/{len(deep_data)} deep sleep epochs")
print(f"   - {deep_as_n2} epochs of true deep sleep classified as N2 (light sleep)")
print(f"   - Only {deep_data['Correct'].mean():.1%} detection rate for deep sleep")

# REM analysis  
rem_data = comparison_df[comparison_df['Ground_Truth'] == 'REM']
print(f"\n2. REM Sleep Detection Issues:")
print(f"   - YASA correctly identifies only {rem_data['Correct'].sum()}/{len(rem_data)} REM epochs")
print(f"   - {rem_data['Correct'].mean():.1%} detection rate for REM sleep")

# Light sleep analysis
light_data = comparison_df[comparison_df['Ground_Truth'] == 'Light']
n2_predictions = (comparison_df['YASA_Raw'] == 'N2').sum()
true_light = len(light_data)
print(f"\n3. Light Sleep Over-prediction:")
print(f"   - YASA predicts N2 in {n2_predictions} epochs vs {true_light} true light sleep epochs")
print(f"   - Best performing stage with {light_data['Correct'].mean():.1%} accuracy")

# Wake analysis
wake_data = comparison_df[comparison_df['Ground_Truth'] == 'Wake']
print(f"\n4. Wake Detection:")
print(f"   - {wake_data['Correct'].mean():.1%} accuracy for wake detection")
print(f"   - Only {len(wake_data)} wake epochs in total (very short wake periods)")

print(f"\n=== Possible Explanations ===")
print("1. Dataset Mismatch: The ground truth labels may use different criteria")
print("   than standard polysomnography scoring that YASA was trained on.")
print("\n2. Signal Quality: YASA may not perform optimally with the specific")
print("   EEG montage (frontal and temporal derivations only).")
print("\n3. Individual Variability: This specific subject's sleep patterns may")
print("   not match the training data distribution.")
print("\n4. Epoch Alignment: There might be slight timing misalignments")
print("   between the ground truth and YASA predictions.")

print(f"\n=== Recommendations ===")
print("1. Try different EEG channels (e.g., central derivations like C3-M2)")
print("2. Check if ground truth uses different staging criteria")
print("3. Consider ensemble methods or post-processing")
print("4. Validate timing alignment between datasets")
print("5. Compare with manual expert scoring as a baseline")

# Create performance summary plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy by stage
stages = ['Wake', 'Light', 'Deep', 'REM']
accuracies = []
for stage in stages:
    stage_data = comparison_df[comparison_df['Ground_Truth'] == stage]
    accuracies.append(stage_data['Correct'].mean() if len(stage_data) > 0 else 0)

ax1.bar(stages, accuracies, color=['skyblue', 'lightgreen', 'orange', 'salmon'])
ax1.set_ylabel('Accuracy')
ax1.set_title('YASA Accuracy by Sleep Stage')
ax1.set_ylim(0, 1)
for i, acc in enumerate(accuracies):
    ax1.text(i, acc + 0.02, f'{acc:.1%}', ha='center', va='bottom')

# Distribution comparison
gt_counts = [gt_dist.get(stage, 0) for stage in stages]
yasa_counts = []
for stage in stages:
    if stage == 'Light':
        yasa_counts.append(yasa_dist.get('N1', 0) + yasa_dist.get('N2', 0))
    elif stage == 'Wake':
        yasa_counts.append(yasa_dist.get('W', 0))
    elif stage == 'Deep':
        yasa_counts.append(yasa_dist.get('N3', 0))
    elif stage == 'REM':
        yasa_counts.append(yasa_dist.get('R', 0))

x = np.arange(len(stages))
width = 0.35
ax2.bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.8)
ax2.bar(x + width/2, yasa_counts, width, label='YASA Predictions', alpha=0.8)
ax2.set_ylabel('Number of Epochs')
ax2.set_title('Stage Distribution Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(stages)
ax2.legend()

# Confidence vs correctness
correct_conf = comparison_df[comparison_df['Correct'] == 1]['YASA_Confidence']
incorrect_conf = comparison_df[comparison_df['Correct'] == 0]['YASA_Confidence']

ax3.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
ax3.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
ax3.set_xlabel('YASA Confidence')
ax3.set_ylabel('Frequency')
ax3.set_title('Confidence Distribution by Correctness')
ax3.legend()

# Accuracy over time
window_size = 60  # 30-minute windows
num_windows = len(comparison_df) // window_size
window_accuracies = []
window_times = []

for i in range(num_windows):
    start_idx = i * window_size
    end_idx = (i + 1) * window_size
    window_data = comparison_df.iloc[start_idx:end_idx]
    window_acc = window_data['Correct'].mean()
    window_accuracies.append(window_acc)
    window_times.append(i * window_size / 120)  # Convert to hours

ax4.plot(window_times, window_accuracies, 'o-', linewidth=2, markersize=6)
ax4.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall ({accuracy:.1%})')
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Accuracy')
ax4.set_title('YASA Accuracy Over Time (30-min windows)')
ax4.set_ylim(0, 1)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('yasa_detailed_performance_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nDetailed performance analysis saved as 'yasa_detailed_performance_analysis.png'")
plt.close()
