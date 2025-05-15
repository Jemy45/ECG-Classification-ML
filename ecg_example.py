from ecg_metadata import (
    load_ecg_data, print_ecg_summary, plot_ecg_segment,
    extract_annotations, plot_ecg_with_annotations
)
from ecg_processing import remove_baseline_wander, remove_powerline_interference, apply_bandpass_filter
from ecg_Rcalculations import detect_r_peaks, plot_ecg_with_peaks, plot_rr_intervals, print_rpeak_results
from ecg_ML import process_training_records, prepare_ml_features, train_ml_model, test_ml_model
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load records 100, 101, and 102
    record_ids = ['100', '101', '102']
    records = load_ecg_data(record_ids)  # Load multiple records
    
    # Print summary for each record
    for record_id, metadata in records.items():
        print_ecg_summary(metadata)
        
    # Example of loading a single record
    single_record = load_ecg_data('101')  # Load single record
    
    # Plot original signal
    plot_ecg_segment(single_record, 'MLII', start_time=0, duration=10)
    plt.title('Original ECG Signal - Record 101, Channel MLII')
    plt.show()
    
    # Extract annotations for a 10-second segment
    annotations = extract_annotations('101', start_time=0, duration=10)
    
    # Plot ECG with annotations
    plot_ecg_with_annotations(single_record, 'MLII', annotations, start_time=0, duration=10)
    plt.show()
    
    # Apply filters and create subplots
    plt.figure(figsize=(15, 10))
    
    # Original signal
    plt.subplot(4, 1, 1)
    plot_ecg_segment(single_record, 'MLII', start_time=0, duration=10)
    plt.title('Original ECG Signal - Record 101, Channel MLII')
    
    # Baseline wander filtered
    baseline_filtered = remove_baseline_wander(single_record)
    plt.subplot(4, 1, 2)
    plot_ecg_segment(baseline_filtered, 'MLII', start_time=0, duration=10)
    plt.title('Baseline Wander Filtered ECG Signal')
    
    # Powerline interference filtered 
    powerline_filtered = remove_powerline_interference(baseline_filtered)
    plt.subplot(4, 1, 3)
    plot_ecg_segment(powerline_filtered, 'MLII', start_time=0, duration=10)
    plt.title('Powerline Interference Filtered ECG Signal')
    
    # Bandpass filtered
    bandpass_filtered = apply_bandpass_filter(powerline_filtered)
    plt.subplot(4, 1, 4)
    plot_ecg_segment(bandpass_filtered, 'MLII', start_time=0, duration=10)
    plt.title('Bandpass Filtered ECG Signal')

    plt.tight_layout()
    plt.show()

    # R-peak Analysis Example
    print("\n=== R-peak Analysis Example ===")
    
    # Use the filtered signal for R-peak detection
    r_peaks = detect_r_peaks(bandpass_filtered)
    
    # Plot ECG with detected R-peaks (showing first 10 seconds)
    plot_ecg_with_peaks(
        bandpass_filtered,
        'MLII',  # Using MLII channel which is typically used for QRS detection
        r_peaks,
        start_time=0,
        duration=10
    )
    
    # Plot RR interval analysis
    plot_rr_intervals(r_peaks)

    # Print R-peak detection statistics
    print_rpeak_results(r_peaks)

    # Process training records and get heart rates
    print("\n=== Processing Training Records ===")
    
    # Process training records and store results
    r_peaks_dict = process_training_records()
    
    print("\n=== Training ML Model ===")
    # Prepare features and train the model
    X, y = prepare_ml_features(r_peaks_dict)
    model_dict = train_ml_model(X, y)
    # Test the trained model on test records
    print("\n=== Testing ML Model ===")
    test_ml_model(model_dict,test_records_file='mit-bih-arrhythmia-database-1.0.0/RECORDS_Testing')
if __name__ == "__main__":
    main() 