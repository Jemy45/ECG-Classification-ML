import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from ecg_metadata import ECGMetadata

@dataclass
class RPeakData:
    """Class to store R-peak detection results and analysis"""
    peak_indices: np.ndarray  # Sample indices where R-peaks occur
    peak_amplitudes: np.ndarray  # Amplitude values at R-peaks
    rr_intervals: np.ndarray  # RR intervals in seconds
    mean_hr: float  # Mean heart rate in BPM
    std_hr: float  # Standard deviation of heart rate
    hrv_metrics: Dict[str, float]  # Heart rate variability metrics
    abnormalities: Dict[str, bool]  # Detected abnormalities

def detect_r_peaks(metadata: ECGMetadata) -> Optional[RPeakData]:
    """
    Detect R-peaks in the ECG signal using scipy.signal.find_peaks.
    Only processes records with MLII channel.
    
    Args:
        metadata: ECGMetadata object containing the signal data
    
    Returns:
        RPeakData object containing peak information and heart rate statistics,
        or None if MLII channel is not found
    """
    # Only use MLII channel
    try:
        channel_idx = metadata.channel_names.index('MLII')
    except ValueError:
        return None
    
    # Get the signal data for MLII channel
    signal_data = metadata.signal_data[:, channel_idx]
    
    # Set default parameters
    # Use 0.6 times the standard deviation above the mean as default threshold
    height = np.mean(signal_data) + 0.6 * np.std(signal_data)
    
    # Use 0.2 seconds worth of samples as minimum distance (typical heart rate < 300 BPM)
    distance = int(0.2 * metadata.sampling_frequency)
    
    # Find peaks
    peak_indices, peak_properties = signal.find_peaks(
        signal_data,
        height=height,
        distance=distance
    )
    
    # Get peak amplitudes
    peak_amplitudes = signal_data[peak_indices]
    
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(peak_indices) / metadata.sampling_frequency
    
    # Calculate heart rate statistics
    hr = 60 / rr_intervals  # Convert RR intervals to heart rate in BPM
    mean_hr = np.mean(hr)
    std_hr = np.std(hr)
    
    return RPeakData(
        peak_indices=peak_indices,
        peak_amplitudes=peak_amplitudes,
        rr_intervals=rr_intervals,
        mean_hr=mean_hr,
        std_hr=std_hr,
        hrv_metrics=calculate_hrv_metrics(rr_intervals),
        abnormalities=detect_abnormalities(mean_hr, rr_intervals)
    )

def calculate_hrv_metrics(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Calculate various heart rate variability metrics.
    
    Args:
        rr_intervals: Array of RR intervals in seconds
    
    Returns:
        Dictionary containing HRV metrics
    """
    # SDNN (Standard deviation of NN intervals)
    sdnn = np.std(rr_intervals) * 1000  # Convert to ms
    
    # RMSSD (Root mean square of successive differences)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) * 1000  # Convert to ms
    
    # pNN50 (Proportion of NN50 divided by total number of NN intervals)
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)  # Count differences > 50ms
    pnn50 = (nn50 / len(rr_intervals)) * 100
    
    return {
        'SDNN': sdnn,  # Standard deviation of NN intervals (ms)
        'RMSSD': rmssd,  # Root mean square of successive differences (ms)
        'pNN50': pnn50,  # Percentage of successive RR intervals differing by > 50ms
    }

def detect_abnormalities(mean_hr: float, rr_intervals: np.ndarray) -> Dict[str, bool]:
    """
    Detect potential cardiac abnormalities based on heart rate and RR intervals.
    
    Args:
        mean_hr: Mean heart rate in BPM
        rr_intervals: Array of RR intervals in seconds
    
    Returns:
        Dictionary containing detected abnormalities
    """
    # Define thresholds
    bradycardia_threshold = 60  # BPM
    tachycardia_threshold = 100  # BPM
    
    return {
        'bradycardia': mean_hr < bradycardia_threshold,
        'tachycardia': mean_hr > tachycardia_threshold
    }

def plot_ecg_with_peaks(metadata: ECGMetadata,
                       channel_name: str,
                       r_peaks: RPeakData,
                       start_time: float = 0,
                       duration: float = 10) -> None:
    """
    Plot ECG signal with detected R-peaks and analysis results.
    
    Args:
        metadata: ECGMetadata object containing the signal data
        channel_name: Name of the channel to plot
        r_peaks: RPeakData object containing peak information
        start_time: Start time in seconds for plotting
        duration: Duration in seconds for plotting
    """
    # Get channel data
    channel_idx = metadata.channel_names.index(channel_name)
    
    # Calculate sample range for plotting
    start_sample = int(start_time * metadata.sampling_frequency)
    end_sample = int((start_time + duration) * metadata.sampling_frequency)
    
    # Get signal segment
    signal = metadata.signal_data[start_sample:end_sample, channel_idx]
    time = np.linspace(start_time, start_time + duration, len(signal))
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot ECG signal
    plt.plot(time, signal, 'b-', label='ECG Signal', linewidth=1)
    
    # Plot R-peaks that fall within the time window
    mask = (r_peaks.peak_indices >= start_sample) & (r_peaks.peak_indices < end_sample)
    peak_times = r_peaks.peak_indices[mask] / metadata.sampling_frequency
    peak_amplitudes = r_peaks.peak_amplitudes[mask]
    
    plt.plot(peak_times, peak_amplitudes, 'ro', label='R-peaks', markersize=8)
    
    # Add analysis results to title
    abnormalities = []
    for condition, present in r_peaks.abnormalities.items():
        if present:
            abnormalities.append(condition.capitalize())
    
    title = f'ECG Signal with Detected R-peaks\n'
    title += f'Mean HR: {r_peaks.mean_hr:.1f} ± {r_peaks.std_hr:.1f} BPM\n'
    if abnormalities:
        title += f'Detected: {", ".join(abnormalities)}'
    
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.title(title)
    plt.legend()
    plt.show()

def print_rpeak_results(r_peaks: Optional[RPeakData]) -> None:
    """
    Print detailed results of R-peak detection analysis.
    
    Args:
        r_peaks: RPeakData object containing peak information and analysis results,
                or None if analysis was not successful
    """
    if r_peaks is None:
        print("\nNo R-peak detection results available (MLII channel not found)")
        return
        
    print(f"\nR-peak Detection Results:")
    print(f"Number of R-peaks detected: {len(r_peaks.peak_indices)}")
    print(f"Mean heart rate: {r_peaks.mean_hr:.1f} BPM")
    print(f"Heart rate standard deviation: {r_peaks.std_hr:.1f} BPM")
    print(f"Mean RR interval: {np.mean(r_peaks.rr_intervals):.3f} seconds")
    print(f"RR interval standard deviation: {np.std(r_peaks.rr_intervals):.3f} seconds")

def plot_rr_intervals(r_peaks: RPeakData) -> None:
    """
    Create visualizations of RR intervals and HRV analysis.
    
    Args:
        r_peaks: RPeakData object containing RR intervals and analysis
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: RR interval time series
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(r_peaks.rr_intervals)), r_peaks.rr_intervals, 'b.-')
    plt.xlabel('Beat number')
    plt.ylabel('RR Interval [s]')
    plt.title('RR Intervals Over Time')
    plt.grid(True)
    
    # Plot 2: RR interval histogram with HRV metrics
    plt.subplot(2, 1, 2)
    plt.hist(r_peaks.rr_intervals, bins=30, color='blue', alpha=0.7)
    plt.xlabel('RR Interval [s]')
    plt.ylabel('Count')
    
    # Add HRV metrics to the plot
    hrv_text = f'HRV Metrics:\n'
    hrv_text += f'SDNN: {r_peaks.hrv_metrics["SDNN"]:.1f} ms\n'
    hrv_text += f'RMSSD: {r_peaks.hrv_metrics["RMSSD"]:.1f} ms\n'
    hrv_text += f'pNN50: {r_peaks.hrv_metrics["pNN50"]:.1f}%'
    
    plt.text(0.98, 0.95, hrv_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('RR Interval Distribution and HRV Analysis')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 