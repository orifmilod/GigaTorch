import numpy as np

def normalize(input, mean, std):
    """
    Normalize a tensor image with mean and standard deviation.
    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels, 
    this transform will normalize each channel of the input Tensor i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]    
    Args:
        input (Tensor): Input image tensor.
        mean (tuple): Sequence of means for each channel.
        std (tuple): Sequence of standard deviations for each channel.
    
    Returns:
        Tensor: Normalized image tensor.
    """
    input.data = input.data.astype(np.float32)  # Convert to float32
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    # Normalize each channel
    for i in range(input.shape[0]):  # Loop over channels
        input[i] = (input[i] - mean[i]) / std[i]

    return input

def standardize(tensor):
    min_val = np.min(tensor.data)
    max_val = np.max(tensor.data)
    return np.clip((tensor.data - min_val) / (max_val.data - min_val), 0, 1)