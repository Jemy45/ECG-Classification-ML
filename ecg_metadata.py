import numpy as np
import wfdb
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple
import matplotlib.pyplot as plt

@dataclass
class ECGMetadata:
    """Class to store ECG record metadata"""
    record_id: str
    sampling_frequency: int
    channel_names: List[str]
    duration: float
    num_samples: int
    signal_data: np.ndarray

@dataclass
class AnnotationData:
    """Class to store annotation data"""
    sample_points: np.ndarray  # Sample points where annotations occur
    symbols: np.ndarray       # Annotation symbols
    descriptions: Dict[str, str]  # Dictionary mapping symbols to descriptions

def extract_annotations(record_id: str, start_time: float = 0, duration: float = 10, base_path: str = 'mit-bih-arrhythmia-database-1.0.0') -> AnnotationData:
    """
    Extract annotations for a given record and time window.
    
    Args:
        record_id: The record ID to analyze
        start_time: Start time in seconds
        duration: Duration in seconds
        base_path: Path to the database directory
    
    Returns:
        AnnotationData object containing sample points, symbols, and descriptions
    """
    # Load annotations
    record_path = f"{base_path}/{record_id}"
    ann = wfdb.rdann(record_path, 'atr')
    
    # Define annotation descriptions
    descriptions = {
        'N': 'Normal beat',
        'L': 'Left bundle branch block beat',
        'R': 'Right bundle branch block beat',
        'A': 'Atrial premature beat',
        'V': 'Premature ventricular contraction',
        'F': 'Fusion of ventricular and normal beat',
        'J': 'Nodal (junctional) premature beat',
        'a': 'Aberrated atrial premature beat',
        'S': 'Premature or ectopic supraventricular beat',
        'E': 'Ventricular escape beat',
        'j': 'Nodal (junctional) escape beat',
        '/': 'Paced beat',
        'Q': 'Unclassifiable beat',
        '~': 'Signal quality change',
        '|': 'Isolated QRS-like artifact',
        '+': 'Rhythm change'
    }
    
    # Convert time to samples (MIT-BIH records use 360 Hz)
    start_sample = int(start_time * 360)
    end_sample = int((start_time + duration) * 360)
    
    # Filter annotations within the time window
    mask = (np.array(ann.sample) >= start_sample) & (np.array(ann.sample) < end_sample)
    sample_points = np.array(ann.sample)[mask]
    symbols = np.array(ann.symbol)[mask]
    # Print unique symbols and their descriptions
    unique_symbols = np.unique(ann.symbol)
    print("\nAnnotation types found:")
    for symbol in unique_symbols:
        if symbol in descriptions:
            print(f"{symbol}: {descriptions[symbol]}")
    
    return AnnotationData(
        sample_points=sample_points,
        symbols=symbols,
        descriptions=descriptions
    )

def plot_ecg_with_annotations(metadata: ECGMetadata, channel_name: str, annotations: AnnotationData, 
                            start_time: float = 0, duration: float = 10):
    """
    Plot ECG data with annotations overlay.
    
    Args:
        metadata: ECGMetadata object containing the signal data
        channel_name: Name of the channel to plot
        annotations: AnnotationData object containing the annotations
        start_time: Start time in seconds
        duration: Duration in seconds
    """
    # First plot the ECG signal
    fig = plot_ecg_segment(metadata, channel_name, start_time, duration)
    
    # Find channel index
    channel_idx = metadata.channel_names.index(channel_name)
    
    # Convert sample points to times
    annotation_times = annotations.sample_points / metadata.sampling_frequency
    
    # Get unique symbols for coloring
    unique_symbols = np.unique(annotations.symbols)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_symbols)))
    
    # Plot each annotation type
    for symbol, color in zip(unique_symbols, colors):
        # Get positions for this symbol
        symbol_mask = annotations.symbols == symbol
        symbol_times = annotation_times[symbol_mask]
        
        # Get signal values at annotation points
        signal_values = np.interp(symbol_times, 
                                np.linspace(start_time, start_time + duration, 
                                          int(duration * metadata.sampling_frequency)),
                                metadata.signal_data[int(start_time * metadata.sampling_frequency):
                                                   int((start_time + duration) * metadata.sampling_frequency),
                                                   channel_idx])
        
        # Plot markers with label
        label = f'{symbol}: {annotations.descriptions.get(symbol, "Other annotation")}'
        plt.plot(symbol_times, signal_values, 'o', color=color, label=label, markersize=8)
    
    # Adjust legend position and layout
    if len(unique_symbols) > 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
    return plt.gcf()

def load_ecg_data(record_ids: Union[str, List[str]], base_path: str = 'mit-bih-arrhythmia-database-1.0.0') -> Union[ECGMetadata, Dict[str, ECGMetadata]]:
    """Load one or multiple ECG records and return their metadata"""
    
    # Handle single record case
    if isinstance(record_ids, str):
        record_path = f"{base_path}/{record_ids}"
        record = wfdb.rdrecord(record_path)
        
        return ECGMetadata(
            record_id=record_ids,
            sampling_frequency=record.fs,
            channel_names=record.sig_name,
            num_samples=record.sig_len,
            duration=record.sig_len / record.fs,
            signal_data=record.p_signal
        )
    
    # Handle multiple records case
    records = {}
    for record_id in record_ids:
        records[record_id] = load_ecg_data(record_id, base_path)
    return records

def print_ecg_summary(metadata: ECGMetadata):
    """Print a summary of the ECG record metadata"""
    print(f"\nRecord {metadata.record_id} Summary:")
    print(f"Sampling frequency: {metadata.sampling_frequency} Hz")
    print(f"Channel names: {metadata.channel_names}")
    print(f"Recording duration: {metadata.duration:.2f} seconds")
    print(f"Number of samples: {metadata.num_samples}")

def plot_ecg_segment(metadata: ECGMetadata, channel_name: str, start_time: float = 0, duration: float = 10):
    """
    Plot a segment of ECG data for a specific channel.
    
    Args:
        metadata: ECGMetadata object containing the signal data
        channel_name: Name of the channel to plot
        start_time: Start time in seconds (default: 0)
        duration: Duration in seconds (default: 10)
    """
    # Find the channel index
    try:
        channel_idx = metadata.channel_names.index(channel_name)
    except ValueError:
        raise ValueError(f"Channel {channel_name} not found. Available channels: {metadata.channel_names}")
    
    # Convert time to samples
    start_sample = int(start_time * metadata.sampling_frequency)
    end_sample = int((start_time + duration) * metadata.sampling_frequency)
    
    # Extract the signal segment
    signal = metadata.signal_data[start_sample:end_sample, channel_idx]
    
    # Create time array
    time = np.linspace(start_time, start_time + duration, len(signal))
    
    # Plot using current axes if they exist, otherwise create new figure
    if plt.get_fignums():
        current_ax = plt.gca()
    else:
        plt.figure(figsize=(15, 5))
        current_ax = plt.gca()
    
    # Plot the signal
    current_ax.plot(time, signal, 'b-', linewidth=1)
    current_ax.grid(True)
    current_ax.set_xlabel('Time [s]')
    current_ax.set_ylabel('Amplitude [mV]')
    
    # Set x-axis limits to exactly match the requested duration
    current_ax.set_xlim(start_time, start_time + duration)
    
    # Calculate y-axis limits with some padding
    y_min, y_max = np.min(signal), np.max(signal)
    y_range = y_max - y_min
    current_ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    return plt.gcf()
