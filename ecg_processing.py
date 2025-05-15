import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from ecg_metadata import ECGMetadata
from typing import Dict, Optional

def remove_baseline_wander(metadata: ECGMetadata, cutoff_freq: float = 0.5) -> ECGMetadata:
    """
    Remove baseline wander using Fourier Transform.
    
    Args:
        metadata: ECGMetadata object containing the signal data
        cutoff_freq: Cutoff frequency for baseline wander (default: 0.5 Hz)
    
    Returns:
        ECGMetadata object containing filtered signal
    """
    filtered_signal = np.zeros_like(metadata.signal_data)
    
    for i in range(metadata.signal_data.shape[1]):
        # Get the signal for this channel
        ecg_signal = metadata.signal_data[:, i]
        
        # Design high-pass filter
        nyquist_freq = metadata.sampling_frequency * 0.5
        normalized_cutoff = cutoff_freq / nyquist_freq
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        
        # Apply filter
        filtered_signal[:, i] = signal.filtfilt(b, a, ecg_signal)
    
    return ECGMetadata(
        record_id=f"{metadata.record_id}_baseline_filtered",
        sampling_frequency=metadata.sampling_frequency,
        channel_names=metadata.channel_names,
        duration=metadata.duration,
        num_samples=metadata.num_samples,
        signal_data=filtered_signal
    )

def remove_powerline_interference(metadata: ECGMetadata, 
                                notch_freq: float = 50.0, 
                                quality_factor: float = 30.0) -> ECGMetadata:
    """
    Remove powerline interference using a notch filter.
    
    Args:
        metadata: ECGMetadata object containing the signal data
        notch_freq: Frequency to remove (default: 50 Hz)
        quality_factor: Quality factor of the notch filter (default: 30.0)
    
    Returns:
        ECGMetadata object containing filtered signal
    """
    filtered_signal = np.zeros_like(metadata.signal_data)
    
    # Design notch filter
    nyquist_freq = metadata.sampling_frequency * 0.5
    normalized_freq = notch_freq / nyquist_freq
    b, a = signal.iirnotch(normalized_freq, quality_factor)
    
    # Apply filter to each channel
    for i in range(metadata.signal_data.shape[1]):
        ecg_signal = metadata.signal_data[:, i]
        filtered_signal[:, i] = signal.filtfilt(b, a, ecg_signal)
    
    return ECGMetadata(
        record_id=f"{metadata.record_id}_powerline_filtered",
        sampling_frequency=metadata.sampling_frequency,
        channel_names=metadata.channel_names,
        duration=metadata.duration,
        num_samples=metadata.num_samples,
        signal_data=filtered_signal
    )

def apply_bandpass_filter(metadata: ECGMetadata,
                         lowcut: float = 0.05, 
                         highcut: float = 40.0, 
                         order: int = 4) -> ECGMetadata:
    """
    Apply bandpass filter to reduce noise while preserving ECG morphology.
    
    Args:
        metadata: ECGMetadata object containing the signal data
        lowcut: Lower cutoff frequency (default: 0.05 Hz)
        highcut: Higher cutoff frequency (default: 40 Hz)
        order: Filter order (default: 4)
    
    Returns:
        ECGMetadata object containing filtered signal
    """
    filtered_signal = np.zeros_like(metadata.signal_data)
    
    # Design bandpass filter
    nyquist_freq = metadata.sampling_frequency * 0.5
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    for i in range(metadata.signal_data.shape[1]):
        ecg_signal = metadata.signal_data[:, i]
        filtered_signal[:, i] = signal.filtfilt(b, a, ecg_signal)
    
    return ECGMetadata(
        record_id=f"{metadata.record_id}_bandpass_filtered",
        sampling_frequency=metadata.sampling_frequency,
        channel_names=metadata.channel_names,
        duration=metadata.duration,
        num_samples=metadata.num_samples,
        signal_data=filtered_signal
    )

