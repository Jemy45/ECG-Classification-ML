from ecg_metadata import (
    load_ecg_data
)
from ecg_processing import (
    remove_baseline_wander,
    remove_powerline_interference,
    apply_bandpass_filter,
    
)
from ecg_Rcalculations import detect_r_peaks, RPeakData
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
    
import numpy as np
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def process_training_records(records_file='mit-bih-arrhythmia-database-1.0.0/RECORDS_Training') -> Dict[str, Optional[RPeakData]]:
    """
    Process training records in the RECORDS file through the filtering pipeline and R-peak detection.
    
    Args:100', '101', '102'
        records_file: Path to the RECORDS file containing record IDs
        
    Returns:
        Dictionary mapping record IDs to their RPeakData objects (None if MLII channel not found)
    """
    r_peaks_dict = {}
    
    # Read records file
    with open(records_file, 'r') as f:
        record_ids = f.read().splitlines()
    
    print(f"\nProcessing {len(record_ids)} records...")
    
    for record_id in record_ids:
        try:
            # Load the record
            record = load_ecg_data(record_id)
            
            # Apply filtering pipeline
            baseline_filtered = remove_baseline_wander(record)
            powerline_filtered = remove_powerline_interference(baseline_filtered)
            bandpass_filtered = apply_bandpass_filter(powerline_filtered)
            
            # Detect R-peaks
            r_peaks = detect_r_peaks(bandpass_filtered)
            if r_peaks is None:
                print(f"Skipping record {record_id} - No MLII channel found")
            r_peaks_dict[record_id] = r_peaks
            
        except Exception as e:
            print(f"Error processing record {record_id}: {str(e)}")
            r_peaks_dict[record_id] = None
            continue
    
    print("\nProcessing complete!")
    processed_records = sum(1 for r in r_peaks_dict.values() if r is not None)
    print(f"Successfully processed {processed_records} out of {len(r_peaks_dict)} records")
    
    return r_peaks_dict

def prepare_ml_features(r_peaks_dict: Dict[str, Optional[RPeakData]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for machine learning from R-peaks data.
    
    Args:
        r_peaks_dict: Dictionary mapping record IDs to their RPeakData objects
        
    Returns:
        Tuple containing:
        - X: numpy array of shape (n_samples, n_features) containing the features
        - y: numpy array of shape (n_samples,) containing the labels (0 for normal, 1 for abnormal)
    """
    features = []
    labels = []
    
    print("\nClassification Results:")
    print("-" * 50)
    print("Record ID | Classification | Abnormalities")
    print("-" * 50)
    
    for record_id, r_peaks in r_peaks_dict.items():
        if r_peaks is not None:
            # Extract features from RR intervals and heart rate data
            record_features = [
                r_peaks.mean_hr,  # Mean heart rate
                r_peaks.std_hr,   # Heart rate variability
                r_peaks.hrv_metrics['SDNN'],  # Standard deviation of RR intervals
                r_peaks.hrv_metrics['RMSSD'], # Root mean square of successive RR differences
                r_peaks.hrv_metrics['pNN50']  # Percentage of successive RR intervals > 50ms
            ]
            
            # Determine label based on detected abnormalities
            has_abnormality = any(r_peaks.abnormalities.values())
            label = 1 if has_abnormality else 0  # 1 for abnormal, 0 for normal
            
            # Print classification details
            abnormalities = [k for k, v in r_peaks.abnormalities.items() if v]
            classification = "Abnormal" if has_abnormality else "Normal"
            abnormalities_str = ", ".join(abnormalities) if abnormalities else "None"
            print(f"{record_id:8} | {classification:13} | {abnormalities_str}")
            
            features.append(record_features)
            labels.append(label)
    
    print("-" * 50)
    
    # Convert to numpy arrays for ML processing
    x = np.array(features)
    y = np.array(labels)
    
    print(f"\nPrepared {len(features)} samples for ML model")
    print(f"Feature shape: {x.shape}")
    print(f"Number of normal records: {sum(y == 0)}")
    print(f"Number of abnormal records: {sum(y == 1)}")
    
    return x, y

def train_ml_model(x, y):
    """
    Train a logistic regression model using the prepared features and labels.
    
    Args:
        x: numpy array of shape (n_samples, n_features) containing the features
        y: numpy array of shape (n_samples,) containing the labels (0 for normal, 1 for abnormal)
        
    Returns:
        Trained LogisticRegression model and performance metrics
    """
   
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the Logistic Regression model
    lr_model = LogisticRegression(
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        max_iter=1000  # Increase max iterations to ensure convergence
    )
    
    # Perform cross-validation on scaled training data
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
    
    # Train the model on the full training set
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    y_pred = lr_model.predict(X_test_scaled)
    
    # Print model performance
    print("\nModel Performance:")
    print("-" * 50)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Create a dictionary to store the scaler with the model
    model_dict = {
        'model': lr_model,
        'scaler': scaler
    }
    
    return model_dict

def test_ml_model(model_dict, test_records_file='mit-bih-arrhythmia-database-1.0.0/RECORDS_Training'):
    """
    Test the trained ML model on a set of test records and provide detailed analysis of classification results.
    
    Args:
        model_dict: Dictionary containing the trained model and scaler
        test_records_file: Path to the file containing test record IDs
        
    Returns:
        Dictionary containing test performance metrics
    """
  
    print("\n=== Testing ML Model ===")
    
    # Process test records
    r_peaks_dict = process_training_records(test_records_file)
    
    # Prepare features and labels for test data
    X_test, y_true = prepare_ml_features(r_peaks_dict)
    
    # Scale features using the same scaler used during training
    X_test_scaled = model_dict['scaler'].transform(X_test)
    
    # Make predictions
    y_pred = model_dict['model'].predict(X_test_scaled)
    y_pred_proba = model_dict['model'].predict_proba(X_test_scaled)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    
    # Print comprehensive performance analysis
    print("\nDetailed Classification Analysis:")
    print("=" * 50)
    print("\n1. Overall Performance Metrics:")
    print("-" * 30)
    print(f"Accuracy:     {accuracy:.3f}")
    print(f"F1 Score:     {f1:.3f}")
    print(f"Precision:    {precision:.3f}")
    print(f"Recall:       {recall:.3f}")
  
    
    print("\n2. Error Analysis:")
    print("-" * 30)
    print(f"False Positive Rate: {false_positive_rate:.3f}")
    print(f"False Negative Rate: {false_negative_rate:.3f}")
    print(f"Number of False Positives: {fp}")
    print(f"Number of False Negatives: {fn}")
    
    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    
    # 1. Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. ROC Curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze misclassifications in detail
    print("\n3. Detailed Misclassification Analysis:")
    print("-" * 50)
    
    record_ids = list(r_peaks_dict.keys())
    misclassified_indices = np.where(y_pred != y_true)[0]
    
    if len(misclassified_indices) > 0:
        print("\nMisclassified Records:")
        print("Record ID | True Label | Predicted | Confidence | Abnormalities")
        print("-" * 75)
        
        for idx in misclassified_indices:
            record_id = record_ids[idx]
            true_label = "Abnormal" if y_true[idx] == 1 else "Normal"
            pred_label = "Abnormal" if y_pred[idx] == 1 else "Normal"
            confidence = max(y_pred_proba[idx])
            
            # Get actual abnormalities for the record
            r_peaks = r_peaks_dict[record_id]
            abnormalities = []
            if r_peaks is not None:
                abnormalities = [k for k, v in r_peaks.abnormalities.items() if v]
            
            abnormalities_str = ", ".join(abnormalities) if abnormalities else "None"
            print(f"{record_id:8} | {true_label:10} | {pred_label:9} | {confidence:.3f}     | {abnormalities_str}")
            
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

