#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation script for the sextortion classifier.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Import our classifier
from classify import is_sextortion

def load_dataset(csv_path):
    """Load and preprocess the evaluation dataset."""
    print("📊 Loading evaluation dataset...")
    df = pd.read_csv(csv_path)
    
    # Convert string TRUE/FALSE to boolean
    df['is_sextortion'] = df['is_sextortion'].map({
        'TRUE': True, 'FALSE': False, True: True, False: False
    })
    
    print("✅ Loaded {} test samples".format(len(df)))
    print("   - Sextortion samples: {}".format(df['is_sextortion'].sum()))
    print("   - Safe samples: {}".format((~df['is_sextortion']).sum()))
    
    return df

def run_classification(df):
    """Run classification on all messages."""
    print("\n🤖 Running classification on {} messages...".format(len(df)))
    print("   This may take a few minutes due to API calls...")
    
    predictions = []
    confidences = []
    start_time = time.time()
    error_count = 0
    
    for idx, row in df.iterrows():
        try:
            pred, conf = is_sextortion(row['message'])
            predictions.append(pred)
            confidences.append(conf)
            
            # Enhanced progress reporting
            if (idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta = (len(df) - idx - 1) / rate if rate > 0 else 0
                print("   📈 Progress: {}/{} ({:.1f}%) - {:.1f} msg/sec - ETA: {:.1f}s".format(
                    idx + 1, len(df), (idx + 1) / len(df) * 100, rate, eta))
                
        except Exception as e:
            error_count += 1
            predictions.append(False)
            confidences.append(0.0)
            if error_count <= 3:  # Only show first few errors
                print("   ⚠️  Error on message {}: {}".format(idx + 1, str(e)[:50]))
    
    elapsed = time.time() - start_time
    rate = len(df) / elapsed
    print("\n✅ Classification complete!")
    print("   📊 Total time: {:.1f}s ({:.2f} messages/second)".format(elapsed, rate))
    if error_count > 0:
        print("   ⚠️  Errors encountered: {} messages".format(error_count))
    
    return predictions, confidences

def calculate_metrics(true_labels, predictions):
    """Calculate evaluation metrics."""
    print("\n📊 Calculating evaluation metrics...")
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    specificity = recall_score(true_labels, predictions, pos_label=False)
    
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    results = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'specificity': specificity,
        'confusion_matrix': cm, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    print("   ✅ Metrics calculated successfully")
    return results

def print_results_table(results):
    """Print formatted results."""
    print("\n" + "="*50)
    print("🎯 CLASSIFICATION RESULTS")
    print("="*50)
    
    metrics = [
        ("Accuracy", "{:.3f}".format(results['accuracy'])),
        ("Precision", "{:.3f}".format(results['precision'])),
        ("Recall", "{:.3f}".format(results['recall'])),
        ("F1-Score", "{:.3f}".format(results['f1_score'])),
        ("Specificity", "{:.3f}".format(results['specificity'])),
    ]
    
    for metric, value in metrics:
        print("{:<20} {:>10}".format(metric, value))
    
    print("\nConfusion Matrix:")
    print("TP: {}, TN: {}".format(results['tp'], results['tn']))
    print("FP: {}, FN: {}".format(results['fp'], results['fn']))

def save_individual_chart(chart_func, filename, title, figsize=(10, 6)):
    """Helper function to save individual charts."""
    plt.figure(figsize=figsize)
    chart_func()
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/{}'.format(filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(true_labels, predictions, confidences, results):
    """Create comprehensive visualizations."""
    print("\n🎨 Generating comprehensive visualizations...")
    print("   Creating 12 individual charts and saving to figures/ folder...")
    
    # Create figures directory
    if not os.path.exists('figures'):
        os.makedirs('figures')
        print("   📁 Created figures/ directory")
    
    # Prepare data for individual charts
    cm = results['confusion_matrix']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metrics_values = [results['accuracy'], results['precision'], results['recall'], 
                     results['f1_score'], results['specificity']]
    y_scores = [conf/100 if pred else (100-conf)/100 for pred, conf in zip(predictions, confidences)]
    
    # 1. Confusion Matrix
    def plot_confusion_matrix():
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Safe', 'Sextortion'],
                    yticklabels=['Safe', 'Sextortion'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    save_individual_chart(plot_confusion_matrix, '01_confusion_matrix.png', 'Confusion Matrix', (8, 6))
    
    # 2. Classification Metrics
    def plot_metrics():
        bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'coral', 'gold', 'plum'])
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '{:.3f}'.format(value), ha='center', va='bottom')
    
    save_individual_chart(plot_metrics, '02_classification_metrics.png', 'Classification Metrics')
    
    # 3. Confidence Distribution
    def plot_confidence_dist():
        plt.hist(confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Confidence (%)')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(confidences), color='red', linestyle='--',
                   label='Mean: {:.1f}%'.format(np.mean(confidences)))
        plt.legend()
    
    save_individual_chart(plot_confidence_dist, '03_confidence_distribution.png', 'Confidence Score Distribution')
    
    # 4. Results Pie Chart
    def plot_results_pie():
        labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
        sizes = [results['tp'], results['tn'], results['fp'], results['fn']]
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
    
    save_individual_chart(plot_results_pie, '04_results_pie_chart.png', 'Classification Results Distribution', (8, 8))
    
    # 5. ROC Curve
    def plot_roc():
        fpr, tpr, _ = roc_curve(true_labels, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = {:.3f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
    
    save_individual_chart(plot_roc, '05_roc_curve.png', 'ROC Curve')
    
    # 6. Precision-Recall Curve
    def plot_pr():
        precision_vals, recall_vals, _ = precision_recall_curve(true_labels, y_scores)
        avg_precision = average_precision_score(true_labels, y_scores)
        plt.plot(recall_vals, precision_vals, color='purple', lw=2,
                label='PR (AP = {:.3f})'.format(avg_precision))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
    
    save_individual_chart(plot_pr, '06_precision_recall_curve.png', 'Precision-Recall Curve')
    
    # 7. Error Analysis
    def plot_errors():
        error_types = ['False Positives', 'False Negatives']
        error_counts = [results['fp'], results['fn']]
        bars = plt.bar(error_types, error_counts, color=['lightcoral', 'lightsalmon'])
        plt.ylabel('Count')
        for bar, count in zip(bars, error_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
    
    save_individual_chart(plot_errors, '07_error_analysis.png', 'Error Analysis', (8, 6))
    
    # 8. Confidence by True Label
    def plot_conf_by_label():
        conf_safe = [conf for conf, label in zip(confidences, true_labels) if not label]
        conf_sext = [conf for conf, label in zip(confidences, true_labels) if label]
        plt.boxplot([conf_safe, conf_sext], labels=['Safe', 'Sextortion'])
        plt.ylabel('Confidence (%)')
    
    save_individual_chart(plot_conf_by_label, '08_confidence_by_label.png', 'Confidence Distribution by True Label')
    
    # 9. Accuracy vs Confidence Threshold
    def plot_acc_vs_thresh():
        thresholds = np.arange(0, 101, 5)
        accuracies = []
        for thresh in thresholds:
            mask = np.array(confidences) >= thresh
            if mask.sum() > 0:
                acc = accuracy_score(np.array(true_labels)[mask], np.array(predictions)[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        plt.plot(thresholds, accuracies, marker='o', color='green')
        plt.xlabel('Min Confidence (%)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
    
    save_individual_chart(plot_acc_vs_thresh, '09_accuracy_vs_threshold.png', 'Accuracy vs Confidence Threshold')
    
    # 10. Sample Count vs Threshold
    def plot_samples_vs_thresh():
        thresholds = np.arange(0, 101, 5)
        sample_counts = [sum(1 for conf in confidences if conf >= thresh) for thresh in thresholds]
        plt.plot(thresholds, sample_counts, marker='s', color='orange')
        plt.xlabel('Min Confidence (%)')
        plt.ylabel('Sample Count')
        plt.grid(True, alpha=0.3)
    
    save_individual_chart(plot_samples_vs_thresh, '10_samples_vs_threshold.png', 'Sample Count vs Confidence Threshold')
    
    # 11. Error Rate by Confidence Bins
    def plot_error_by_bins():
        bins = np.arange(0, 101, 20)
        error_rates = []
        bin_labels = []
        for i in range(len(bins)-1):
            mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
            if mask.sum() > 0:
                error_rate = 1 - accuracy_score(np.array(true_labels)[mask], 
                                              np.array(predictions)[mask])
                error_rates.append(error_rate)
                bin_labels.append('{}-{}'.format(bins[i], bins[i+1]))
        plt.bar(range(len(error_rates)), error_rates, color='salmon')
        plt.xlabel('Confidence Bin (%)')
        plt.ylabel('Error Rate')
        plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
    
    save_individual_chart(plot_error_by_bins, '11_error_rate_by_bins.png', 'Error Rate by Confidence Bins')
    
    # 12. Performance Summary (Text-based)
    def plot_summary():
        plt.axis('off')
        summary_text = """PERFORMANCE SUMMARY
        
Total Samples: {}
Accuracy: {:.3f}
Precision: {:.3f}
Recall: {:.3f}
F1-Score: {:.3f}

True Positives: {}
True Negatives: {}
False Positives: {}
False Negatives: {}

Avg Confidence: {:.1f}%
Std Confidence: {:.1f}%""".format(
            len(true_labels), results['accuracy'], results['precision'], 
            results['recall'], results['f1_score'], results['tp'], 
            results['tn'], results['fp'], results['fn'],
            np.mean(confidences), np.std(confidences))
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    save_individual_chart(plot_summary, '12_performance_summary.png', 'Performance Summary', (10, 8))
    
    print("   ✅ Individual charts saved (01-12)")
    
    # Now create the combined figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Sextortion Classifier Evaluation Results', fontsize=16, fontweight='bold')
    
        # 1. Confusion Matrix
    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Safe', 'Sextortion'],
                yticklabels=['Safe', 'Sextortion'])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # 2. Metrics Bar Chart
    ax = axes[0, 1]
    bars = ax.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'coral', 'gold', 'plum'])
    ax.set_title('Classification Metrics')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1)
    
    for bar, value in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                '{:.3f}'.format(value), ha='center', va='bottom')
     
    # 3. Confidence Distribution
    ax = axes[0, 2]
    ax.hist(confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_title('Confidence Score Distribution')
    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(confidences), color='red', linestyle='--',
               label='Mean: {:.1f}%'.format(np.mean(confidences)))
    ax.legend()
    
    # 4. Results Pie Chart
    ax = axes[0, 3]
    labels = ['TP', 'TN', 'FP', 'FN']
    sizes = [results['tp'], results['tn'], results['fp'], results['fn']]
    colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Classification Results')
    
    # 5. ROC Curve
    ax = axes[1, 0]
    y_scores = [conf/100 if pred else (100-conf)/100 for pred, conf in zip(predictions, confidences)]
    fpr, tpr, _ = roc_curve(true_labels, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = {:.3f})'.format(roc_auc))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    
    # 6. Precision-Recall Curve
    ax = axes[1, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, y_scores)
    avg_precision = average_precision_score(true_labels, y_scores)
    
    ax.plot(recall_vals, precision_vals, color='purple', lw=2,
            label='PR (AP = {:.3f})'.format(avg_precision))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    
    # 7. Error Analysis
    ax = axes[1, 2]
    error_types = ['False Positives', 'False Negatives']
    error_counts = [results['fp'], results['fn']]
    bars = ax.bar(error_types, error_counts, color=['lightcoral', 'lightsalmon'])
    ax.set_title('Error Analysis')
    ax.set_ylabel('Count')
    
    for bar, count in zip(bars, error_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # 8. Confidence by True Label
    ax = axes[1, 3]
    conf_safe = [conf for conf, label in zip(confidences, true_labels) if not label]
    conf_sext = [conf for conf, label in zip(confidences, true_labels) if label]
    
    ax.boxplot([conf_safe, conf_sext], labels=['Safe', 'Sextortion'])
    ax.set_title('Confidence by True Label')
    ax.set_ylabel('Confidence (%)')
    
    # 9. Accuracy vs Confidence Threshold
    ax = axes[2, 0]
    thresholds = np.arange(0, 101, 5)
    accuracies = []
    
    for thresh in thresholds:
        mask = np.array(confidences) >= thresh
        if mask.sum() > 0:
            acc = accuracy_score(np.array(true_labels)[mask], np.array(predictions)[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    ax.plot(thresholds, accuracies, marker='o', color='green')
    ax.set_title('Accuracy vs Confidence Threshold')
    ax.set_xlabel('Min Confidence (%)')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    
    # 10. Sample Count vs Threshold
    ax = axes[2, 1]
    sample_counts = [sum(1 for conf in confidences if conf >= thresh) for thresh in thresholds]
    
    ax.plot(thresholds, sample_counts, marker='s', color='orange')
    ax.set_title('Samples vs Confidence Threshold')
    ax.set_xlabel('Min Confidence (%)')
    ax.set_ylabel('Sample Count')
    ax.grid(True, alpha=0.3)
    
    # 11. Error Rate by Confidence Bins
    ax = axes[2, 2]
    bins = np.arange(0, 101, 20)
    error_rates = []
    bin_labels = []
    
    for i in range(len(bins)-1):
        mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
        if mask.sum() > 0:
            error_rate = 1 - accuracy_score(np.array(true_labels)[mask], 
                                          np.array(predictions)[mask])
            error_rates.append(error_rate)
            bin_labels.append('{}-{}'.format(bins[i], bins[i+1]))
    
    ax.bar(range(len(error_rates)), error_rates, color='salmon')
    ax.set_title('Error Rate by Confidence Bins')
    ax.set_xlabel('Confidence Bin (%)')
    ax.set_ylabel('Error Rate')
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45)
    
    # 12. Performance Summary
    ax = axes[2, 3]
    ax.axis('off')
    summary_text = """
PERFORMANCE SUMMARY

Total Samples: {}
Accuracy: {:.3f}
Precision: {:.3f}
Recall: {:.3f}
F1-Score: {:.3f}

True Positives: {}
True Negatives: {}
False Positives: {}
False Negatives: {}

Avg Confidence: {:.1f}%
Std Confidence: {:.1f}%
    """.format(len(true_labels), results['accuracy'], results['precision'], 
               results['recall'], results['f1_score'], results['tp'], 
               results['tn'], results['fp'], results['fn'],
               np.mean(confidences), np.std(confidences))
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('classifier_evaluation_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ All visualizations completed!")
    print("   💾 Individual charts saved in figures/ folder (01-12)")
    print("   💾 Combined overview saved as 'classifier_evaluation_combined.png'")

def analyze_errors(df, true_labels, predictions, confidences):
    """Analyze misclassified examples."""
    print("\n🔍 ERROR ANALYSIS")
    print("="*50)
    
    # False Positives
    fp_mask = (np.array(predictions) == True) & (np.array(true_labels) == False)
    fp_indices = np.where(fp_mask)[0]
    
    print("\n❌ FALSE POSITIVES ({} examples):".format(len(fp_indices)))
    for i, idx in enumerate(fp_indices[:3]):
        msg = df.iloc[idx]['message']
        conf = confidences[idx]
        print("{}. [{:5.1f}%] {}...".format(i+1, conf, msg[:60]))
    
    # False Negatives
    fn_mask = (np.array(predictions) == False) & (np.array(true_labels) == True)
    fn_indices = np.where(fn_mask)[0]
    
    print("\n❌ FALSE NEGATIVES ({} examples):".format(len(fn_indices)))
    for i, idx in enumerate(fn_indices[:3]):
        msg = df.iloc[idx]['message']
        conf = confidences[idx]
        print("{}. [{:5.1f}%] {}...".format(i+1, conf, msg[:60]))

def main():
    """Main evaluation function."""
    print("🚀 SEXTORTION CLASSIFIER EVALUATION")
    print("="*60)
    print("This script will comprehensively evaluate the classifier with:")
    print("• Load and process {} test messages".format(302))  # We know it's 302 from the CSV
    print("• Run AI classification on each message")
    print("• Calculate detailed metrics and statistics")
    print("• Generate 12 different visualizations")
    print("• Analyze classification errors")
    print("="*60)
    
    start_time = time.time()
    
    # Load dataset
    df = load_dataset('llm_eval_dataset.csv')
    
    # Run classification
    predictions, confidences = run_classification(df)
    true_labels = df['is_sextortion'].tolist()
    
    # Calculate metrics
    results = calculate_metrics(true_labels, predictions)
    
    # Print results
    print_results_table(results)
    
    # Create visualizations
    create_visualizations(true_labels, predictions, confidences, results)
    
    # Analyze errors
    analyze_errors(df, true_labels, predictions, confidences)
    
    # Detailed report
    print("\n📋 DETAILED SKLEARN CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(true_labels, predictions, 
                                 target_names=['Safe', 'Sextortion'], digits=4)
    print(report)
    
    total_time = time.time() - start_time
    print("\n🎉 EVALUATION COMPLETE!")
    print("="*60)
    print("📊 Total evaluation time: {:.1f} minutes".format(total_time / 60))
    print("📈 Average time per message: {:.2f} seconds".format(total_time / len(df)))
    print("📁 Individual charts: figures/ folder (01-12)")
    print("📊 Combined overview: classifier_evaluation_combined.png")
    print("="*60)

if __name__ == "__main__":
    main() 