"""
Numba-accelerated core functions for bootstrap implementations.

This module provides JIT-compiled versions of performance-critical operations
for bootstrap methods in the MFE Toolbox. These optimized functions significantly
improve performance for bootstrap operations, especially with large datasets or
many replications.

The module leverages Numba's just-in-time compilation capabilities through the
@jit decorator to transform Python functions into optimized machine code at runtime.
This approach provides near-native performance while maintaining the readability
and maintainability of Python code.

Key optimized operations include:
- Block generation for block bootstrap
- Random block length generation for stationary bootstrap
- Efficient index sampling for bootstrap resampling
- Accelerated test statistic calculations for bootstrap inference
- Memory-efficient operations for large bootstrap samples

These functions are designed to be called from higher-level bootstrap classes
and are not typically used directly by end users.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numba import jit, prange, float64, int64, boolean


@jit(nopython=True, cache=True)
def generate_block_bootstrap_indices(
    data_length: int,
    n_bootstraps: int,
    block_length: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate block bootstrap indices using Numba acceleration.
    
    This function implements the core algorithm for generating block bootstrap
    indices with circular wrapping. It is accelerated using Numba's @jit decorator
    for improved performance on large datasets and many bootstrap replications.
    
    Args:
        data_length: Length of the original data
        n_bootstraps: Number of bootstrap samples to generate
        block_length: Length of each block
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
    """
    # Initialize random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate number of blocks needed to cover the data length
    n_blocks = int(np.ceil(data_length / block_length))
    
    # Initialize indices array
    indices = np.zeros((n_bootstraps, data_length), dtype=np.int64)
    
    # Generate bootstrap indices for each bootstrap sample
    for i in range(n_bootstraps):
        # Generate random starting positions for blocks
        block_starts = np.random.randint(0, data_length, size=n_blocks)
        
        # Fill the indices array with blocks
        idx = 0
        for start in block_starts:
            # Add indices from the current block
            for j in range(block_length):
                if idx >= data_length:
                    break
                # Use modulo to implement circular wrapping
                indices[i, idx] = (start + j) % data_length
                idx += 1
    
    return indices


@jit(nopython=True, cache=True)
def generate_stationary_bootstrap_indices(
    data_length: int,
    n_bootstraps: int,
    expected_block_length: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate stationary bootstrap indices using Numba acceleration.
    
    This function implements the core algorithm for generating stationary bootstrap
    indices with random block lengths following a geometric distribution. It is
    accelerated using Numba's @jit decorator for improved performance on large
    datasets and many bootstrap replications.
    
    Args:
        data_length: Length of the original data
        n_bootstraps: Number of bootstrap samples to generate
        expected_block_length: Expected length of each block (parameter for geometric distribution)
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
    """
    # Initialize random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate probability parameter for geometric distribution
    # p is the probability of starting a new block
    p = 1.0 / expected_block_length
    
    # Initialize indices array
    indices = np.zeros((n_bootstraps, data_length), dtype=np.int64)
    
    # Generate bootstrap indices for each bootstrap sample
    for i in range(n_bootstraps):
        # Initialize the first index randomly
        idx = 0
        indices[i, 0] = np.random.randint(0, data_length)
        
        # Generate the rest of the indices
        while idx < data_length - 1:
            # Decide whether to start a new block or continue the current one
            if np.random.random() < p:
                # Start a new block with a random index
                indices[i, idx + 1] = np.random.randint(0, data_length)
            else:
                # Continue the current block (with circular wrapping)
                indices[i, idx + 1] = (indices[i, idx] + 1) % data_length
            
            idx += 1
    
    return indices


@jit(nopython=True, cache=True, parallel=True)
def resample_data(
    data: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Resample data using bootstrap indices with Numba acceleration.
    
    This function efficiently resamples data using the provided bootstrap indices.
    It is optimized for both 1D and 2D data arrays and uses parallel processing
    when possible for improved performance.
    
    Args:
        data: Original data array (1D or 2D)
        indices: Bootstrap indices with shape (n_bootstraps, data_length)
        
    Returns:
        np.ndarray: Resampled data with shape (n_bootstraps, data_length, ...)
    """
    n_bootstraps, sample_length = indices.shape
    
    # Handle different data dimensions
    if data.ndim == 1:
        # For 1D data, use simple indexing
        bootstrap_samples = np.zeros((n_bootstraps, sample_length), dtype=data.dtype)
        for i in prange(n_bootstraps):
            for j in range(sample_length):
                bootstrap_samples[i, j] = data[indices[i, j]]
    elif data.ndim == 2:
        # For 2D data, index along the first dimension
        n_vars = data.shape[1]
        bootstrap_samples = np.zeros((n_bootstraps, sample_length, n_vars), dtype=data.dtype)
        for i in prange(n_bootstraps):
            for j in range(sample_length):
                bootstrap_samples[i, j, :] = data[indices[i, j], :]
    else:
        # This should not happen as validation is done at a higher level
        # But we include it for completeness
        raise ValueError("Data must be 1D or 2D")
    
    return bootstrap_samples


@jit(nopython=True, cache=True)
def compute_bootstrap_mean(
    data: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Compute bootstrap means using Numba acceleration.
    
    This function efficiently computes the mean of each bootstrap sample.
    It is optimized for both 1D and 2D data arrays.
    
    Args:
        data: Original data array (1D or 2D)
        indices: Bootstrap indices with shape (n_bootstraps, sample_length)
        
    Returns:
        np.ndarray: Bootstrap means with shape (n_bootstraps,) or (n_bootstraps, n_vars)
    """
    n_bootstraps, sample_length = indices.shape
    
    # Handle different data dimensions
    if data.ndim == 1:
        # For 1D data, compute scalar means
        bootstrap_means = np.zeros(n_bootstraps, dtype=np.float64)
        for i in range(n_bootstraps):
            sum_val = 0.0
            for j in range(sample_length):
                sum_val += data[indices[i, j]]
            bootstrap_means[i] = sum_val / sample_length
    elif data.ndim == 2:
        # For 2D data, compute means for each variable
        n_vars = data.shape[1]
        bootstrap_means = np.zeros((n_bootstraps, n_vars), dtype=np.float64)
        for i in range(n_bootstraps):
            for v in range(n_vars):
                sum_val = 0.0
                for j in range(sample_length):
                    sum_val += data[indices[i, j], v]
                bootstrap_means[i, v] = sum_val / sample_length
    else:
        # This should not happen as validation is done at a higher level
        # But we include it for completeness
        raise ValueError("Data must be 1D or 2D")
    
    return bootstrap_means


@jit(nopython=True, cache=True)
def compute_bootstrap_variance(
    data: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Compute bootstrap variances using Numba acceleration.
    
    This function efficiently computes the variance of each bootstrap sample.
    It is optimized for both 1D and 2D data arrays.
    
    Args:
        data: Original data array (1D or 2D)
        indices: Bootstrap indices with shape (n_bootstraps, sample_length)
        
    Returns:
        np.ndarray: Bootstrap variances with shape (n_bootstraps,) or (n_bootstraps, n_vars)
    """
    n_bootstraps, sample_length = indices.shape
    
    # Handle different data dimensions
    if data.ndim == 1:
        # For 1D data, compute scalar variances
        bootstrap_vars = np.zeros(n_bootstraps, dtype=np.float64)
        for i in range(n_bootstraps):
            # First compute the mean
            sum_val = 0.0
            for j in range(sample_length):
                sum_val += data[indices[i, j]]
            mean_val = sum_val / sample_length
            
            # Then compute the variance
            sum_sq_diff = 0.0
            for j in range(sample_length):
                diff = data[indices[i, j]] - mean_val
                sum_sq_diff += diff * diff
            
            # Use n-1 for unbiased estimator
            bootstrap_vars[i] = sum_sq_diff / (sample_length - 1) if sample_length > 1 else 0.0
    elif data.ndim == 2:
        # For 2D data, compute variances for each variable
        n_vars = data.shape[1]
        bootstrap_vars = np.zeros((n_bootstraps, n_vars), dtype=np.float64)
        for i in range(n_bootstraps):
            for v in range(n_vars):
                # First compute the mean
                sum_val = 0.0
                for j in range(sample_length):
                    sum_val += data[indices[i, j], v]
                mean_val = sum_val / sample_length
                
                # Then compute the variance
                sum_sq_diff = 0.0
                for j in range(sample_length):
                    diff = data[indices[i, j], v] - mean_val
                    sum_sq_diff += diff * diff
                
                # Use n-1 for unbiased estimator
                bootstrap_vars[i, v] = sum_sq_diff / (sample_length - 1) if sample_length > 1 else 0.0
    else:
        # This should not happen as validation is done at a higher level
        # But we include it for completeness
        raise ValueError("Data must be 1D or 2D")
    
    return bootstrap_vars


@jit(nopython=True, cache=True)
def compute_bootstrap_quantiles(
    data: np.ndarray,
    indices: np.ndarray,
    quantiles: np.ndarray
) -> np.ndarray:
    """
    Compute bootstrap quantiles using Numba acceleration.
    
    This function efficiently computes quantiles for each bootstrap sample.
    It is optimized for both 1D and 2D data arrays.
    
    Args:
        data: Original data array (1D or 2D)
        indices: Bootstrap indices with shape (n_bootstraps, sample_length)
        quantiles: Quantile values to compute (between 0 and 1)
        
    Returns:
        np.ndarray: Bootstrap quantiles with shape (n_bootstraps, n_quantiles) or 
                   (n_bootstraps, n_vars, n_quantiles)
    """
    n_bootstraps, sample_length = indices.shape
    n_quantiles = len(quantiles)
    
    # Handle different data dimensions
    if data.ndim == 1:
        # For 1D data, compute scalar quantiles
        bootstrap_quantiles = np.zeros((n_bootstraps, n_quantiles), dtype=np.float64)
        for i in range(n_bootstraps):
            # Extract the bootstrap sample
            sample = np.zeros(sample_length, dtype=data.dtype)
            for j in range(sample_length):
                sample[j] = data[indices[i, j]]
            
            # Sort the sample
            sample = np.sort(sample)
            
            # Compute quantiles
            for q in range(n_quantiles):
                # Linear interpolation for quantiles
                idx = quantiles[q] * (sample_length - 1)
                idx_floor = int(np.floor(idx))
                idx_ceil = int(np.ceil(idx))
                
                if idx_floor == idx_ceil:
                    bootstrap_quantiles[i, q] = sample[idx_floor]
                else:
                    weight_ceil = idx - idx_floor
                    weight_floor = 1.0 - weight_ceil
                    bootstrap_quantiles[i, q] = weight_floor * sample[idx_floor] + weight_ceil * sample[idx_ceil]
    elif data.ndim == 2:
        # For 2D data, compute quantiles for each variable
        n_vars = data.shape[1]
        bootstrap_quantiles = np.zeros((n_bootstraps, n_vars, n_quantiles), dtype=np.float64)
        for i in range(n_bootstraps):
            for v in range(n_vars):
                # Extract the bootstrap sample for this variable
                sample = np.zeros(sample_length, dtype=data.dtype)
                for j in range(sample_length):
                    sample[j] = data[indices[i, j], v]
                
                # Sort the sample
                sample = np.sort(sample)
                
                # Compute quantiles
                for q in range(n_quantiles):
                    # Linear interpolation for quantiles
                    idx = quantiles[q] * (sample_length - 1)
                    idx_floor = int(np.floor(idx))
                    idx_ceil = int(np.ceil(idx))
                    
                    if idx_floor == idx_ceil:
                        bootstrap_quantiles[i, v, q] = sample[idx_floor]
                    else:
                        weight_ceil = idx - idx_floor
                        weight_floor = 1.0 - weight_ceil
                        bootstrap_quantiles[i, v, q] = weight_floor * sample[idx_floor] + weight_ceil * sample[idx_ceil]
    else:
        # This should not happen as validation is done at a higher level
        # But we include it for completeness
        raise ValueError("Data must be 1D or 2D")
    
    return bootstrap_quantiles


@jit(nopython=True, cache=True)
def compute_loss_differentials(
    losses: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise loss differentials for all models.
    
    This function computes the pairwise loss differentials between all models
    for each time period. It is accelerated using Numba's @jit decorator for
    improved performance on large datasets.
    
    Args:
        losses: Loss matrix of shape (T, M) where T is the number of time periods
               and M is the number of models
        
    Returns:
        np.ndarray: Loss differential matrix of shape (T, M, M)
    """
    T, M = losses.shape
    loss_diffs = np.zeros((T, M, M))
    
    for i in range(M):
        for j in range(M):
            if i != j:
                loss_diffs[:, i, j] = losses[:, i] - losses[:, j]
    
    return loss_diffs


@jit(nopython=True, cache=True)
def compute_t_max_statistic(
    loss_diffs: np.ndarray,
    model_indices: np.ndarray
) -> Tuple[float, int, np.ndarray]:
    """
    Compute the T_max test statistic for the MCS procedure.
    
    This function computes the T_max test statistic, which is the maximum t-statistic
    across all models in the current set. It is accelerated using Numba's @jit decorator
    for improved performance.
    
    Args:
        loss_diffs: Loss differential matrix of shape (T, M, M)
        model_indices: Indices of models in the current set
        
    Returns:
        Tuple containing:
        - float: The T_max test statistic
        - int: Index of the model with the worst performance
        - np.ndarray: Array of t-statistics for each model
    """
    T = loss_diffs.shape[0]
    n_models = len(model_indices)
    
    # Initialize arrays for mean loss differentials and variances
    mean_diffs = np.zeros(n_models)
    t_stats = np.zeros(n_models)
    
    # Compute mean loss differential for each model
    for i in range(n_models):
        model_idx = model_indices[i]
        # Compute mean loss differential against all other models in the set
        diff_sum = 0.0
        count = 0
        for j in range(n_models):
            if i != j:
                model_j_idx = model_indices[j]
                diff_sum += np.mean(loss_diffs[:, model_idx, model_j_idx])
                count += 1
        
        if count > 0:
            mean_diffs[i] = diff_sum / count
        
        # Compute variance of loss differentials
        var_sum = 0.0
        count = 0
        for j in range(n_models):
            if i != j:
                model_j_idx = model_indices[j]
                diffs = loss_diffs[:, model_idx, model_j_idx]
                var_sum += np.var(diffs) * (T - 1) / T  # Unbiased estimator
                count += 1
        
        # Compute t-statistic
        if count > 0 and var_sum > 0:
            t_stats[i] = mean_diffs[i] / np.sqrt(var_sum / (count * T))
        else:
            t_stats[i] = 0.0
    
    # Find the maximum t-statistic and corresponding model index
    max_t_idx = np.argmax(t_stats)
    max_t = t_stats[max_t_idx]
    worst_model_idx = model_indices[max_t_idx]
    
    return max_t, worst_model_idx, t_stats


@jit(nopython=True, cache=True)
def compute_t_r_statistic(
    loss_diffs: np.ndarray,
    model_indices: np.ndarray
) -> Tuple[float, int, np.ndarray]:
    """
    Compute the T_R test statistic for the MCS procedure.
    
    This function computes the T_R test statistic, which is the sum of squared
    t-statistics across all models in the current set. It is accelerated using
    Numba's @jit decorator for improved performance.
    
    Args:
        loss_diffs: Loss differential matrix of shape (T, M, M)
        model_indices: Indices of models in the current set
        
    Returns:
        Tuple containing:
        - float: The T_R test statistic
        - int: Index of the model with the worst performance
        - np.ndarray: Array of t-statistics for each model
    """
    T = loss_diffs.shape[0]
    n_models = len(model_indices)
    
    # Initialize arrays for mean loss differentials and variances
    mean_diffs = np.zeros(n_models)
    t_stats = np.zeros(n_models)
    
    # Compute mean loss differential for each model
    for i in range(n_models):
        model_idx = model_indices[i]
        # Compute mean loss differential against all other models in the set
        diff_sum = 0.0
        count = 0
        for j in range(n_models):
            if i != j:
                model_j_idx = model_indices[j]
                diff_sum += np.mean(loss_diffs[:, model_idx, model_j_idx])
                count += 1
        
        if count > 0:
            mean_diffs[i] = diff_sum / count
        
        # Compute variance of loss differentials
        var_sum = 0.0
        count = 0
        for j in range(n_models):
            if i != j:
                model_j_idx = model_indices[j]
                diffs = loss_diffs[:, model_idx, model_j_idx]
                var_sum += np.var(diffs) * (T - 1) / T  # Unbiased estimator
                count += 1
        
        # Compute t-statistic
        if count > 0 and var_sum > 0:
            t_stats[i] = mean_diffs[i] / np.sqrt(var_sum / (count * T))
        else:
            t_stats[i] = 0.0
    
    # Compute T_R statistic (sum of squared t-statistics)
    t_r = np.sum(t_stats**2)
    
    # Find the model with the highest individual t-statistic
    max_t_idx = np.argmax(t_stats)
    worst_model_idx = model_indices[max_t_idx]
    
    return t_r, worst_model_idx, t_stats


@jit(nopython=True, cache=True)
def bootstrap_test_distribution(
    loss_diffs: np.ndarray,
    model_indices: np.ndarray,
    bootstrap_indices: np.ndarray,
    use_t_max: bool = True
) -> np.ndarray:
    """
    Compute the bootstrap distribution of the test statistic.
    
    This function computes the bootstrap distribution of either the T_max or T_R
    test statistic. It is accelerated using Numba's @jit decorator for improved
    performance.
    
    Args:
        loss_diffs: Loss differential matrix of shape (T, M, M)
        model_indices: Indices of models in the current set
        bootstrap_indices: Bootstrap indices of shape (n_bootstraps, T)
        use_t_max: Whether to use the T_max statistic (True) or T_R statistic (False)
        
    Returns:
        np.ndarray: Bootstrap distribution of the test statistic
    """
    T = loss_diffs.shape[0]
    n_bootstraps = bootstrap_indices.shape[0]
    n_models = len(model_indices)
    
    # Initialize array for bootstrap test statistics
    bootstrap_stats = np.zeros(n_bootstraps)
    
    # Compute bootstrap test statistics
    for b in range(n_bootstraps):
        # Get bootstrap indices for this replication
        indices = bootstrap_indices[b]
        
        # Initialize arrays for mean loss differentials and variances
        mean_diffs = np.zeros(n_models)
        t_stats = np.zeros(n_models)
        
        # Compute mean loss differential for each model
        for i in range(n_models):
            model_idx = model_indices[i]
            # Compute mean loss differential against all other models in the set
            diff_sum = 0.0
            count = 0
            for j in range(n_models):
                if i != j:
                    model_j_idx = model_indices[j]
                    # Use bootstrap indices to compute mean
                    diff_sum += np.mean(loss_diffs[indices, model_idx, model_j_idx])
                    count += 1
            
            if count > 0:
                mean_diffs[i] = diff_sum / count
            
            # Compute variance of loss differentials
            var_sum = 0.0
            count = 0
            for j in range(n_models):
                if i != j:
                    model_j_idx = model_indices[j]
                    # Use bootstrap indices to compute variance
                    diffs = loss_diffs[indices, model_idx, model_j_idx]
                    var_sum += np.var(diffs) * (T - 1) / T  # Unbiased estimator
                    count += 1
            
            # Compute t-statistic
            if count > 0 and var_sum > 0:
                t_stats[i] = mean_diffs[i] / np.sqrt(var_sum / (count * T))
            else:
                t_stats[i] = 0.0
        
        # Compute test statistic based on the specified type
        if use_t_max:
            # T_max is the maximum t-statistic
            bootstrap_stats[b] = np.max(t_stats)
        else:
            # T_R is the sum of squared t-statistics
            bootstrap_stats[b] = np.sum(t_stats**2)
    
    return bootstrap_stats


@jit(nopython=True, cache=True)
def compute_bsds_statistics(
    benchmark_losses: np.ndarray,
    model_losses: np.ndarray
) -> np.ndarray:
    """
    Compute BSDS test statistics using Numba acceleration.
    
    This function computes the loss differentials between the benchmark model
    and each alternative model, and returns the mean loss differentials.
    
    Args:
        benchmark_losses: Loss values for the benchmark model
        model_losses: Loss values for the alternative models
        
    Returns:
        np.ndarray: BSDS test statistics (mean loss differentials)
    """
    # Get dimensions
    n_obs = benchmark_losses.shape[0]
    n_models = model_losses.shape[1]
    
    # Compute loss differentials
    loss_diffs = np.zeros((n_obs, n_models))
    for i in range(n_models):
        loss_diffs[:, i] = benchmark_losses - model_losses[:, i]
    
    # Compute mean loss differentials
    mean_loss_diffs = np.mean(loss_diffs, axis=0)
    
    return mean_loss_diffs


@jit(nopython=True, cache=True)
def compute_bsds_p_values(
    original_stats: np.ndarray,
    bootstrap_stats: np.ndarray,
    p_value_type: str
) -> np.ndarray:
    """
    Compute BSDS p-values using Numba acceleration.
    
    This function computes the p-values for the BSDS test based on the original
    statistics and bootstrap statistics.
    
    Args:
        original_stats: Original test statistics
        bootstrap_stats: Bootstrap test statistics
        p_value_type: Type of p-value to compute ('consistent', 'upper', 'lower')
        
    Returns:
        np.ndarray: BSDS p-values
    """
    n_bootstraps = bootstrap_stats.shape[0]
    n_models = bootstrap_stats.shape[1]
    
    # Initialize p-values
    p_values = np.zeros(n_models)
    
    # Compute p-values based on the specified type
    if p_value_type == 'consistent':
        # Consistent p-values (Hansen, 2005)
        for i in range(n_models):
            p_values[i] = np.mean(bootstrap_stats[:, i] >= original_stats[i])
    elif p_value_type == 'upper':
        # Upper p-values (White, 2000)
        bootstrap_max = np.max(bootstrap_stats, axis=1)
        for i in range(n_models):
            p_values[i] = np.mean(bootstrap_max >= original_stats[i])
    else:  # p_value_type == 'lower'
        # Lower p-values
        for i in range(n_models):
            p_values[i] = np.mean(bootstrap_stats[:, i] >= original_stats[i])
    
    return p_values


@jit(nopython=True, cache=True, parallel=True)
def compute_bootstrap_statistics_parallel(
    data: np.ndarray,
    indices: np.ndarray,
    statistic_func_id: int
) -> np.ndarray:
    """
    Compute bootstrap statistics in parallel using Numba acceleration.
    
    This function computes bootstrap statistics for each bootstrap sample using
    a specified statistic function. It uses parallel processing for improved
    performance on multi-core systems.
    
    Args:
        data: Original data array
        indices: Bootstrap indices with shape (n_bootstraps, data_length)
        statistic_func_id: ID of the statistic function to use:
                          1 = mean, 2 = variance, 3 = median, 4 = max, 5 = min
        
    Returns:
        np.ndarray: Bootstrap statistics
    """
    n_bootstraps, sample_length = indices.shape
    
    # Initialize result array
    if data.ndim == 1:
        bootstrap_stats = np.zeros(n_bootstraps, dtype=np.float64)
    else:  # data.ndim == 2
        n_vars = data.shape[1]
        bootstrap_stats = np.zeros((n_bootstraps, n_vars), dtype=np.float64)
    
    # Compute bootstrap statistics in parallel
    for i in prange(n_bootstraps):
        # Extract the bootstrap sample
        if data.ndim == 1:
            sample = np.zeros(sample_length, dtype=data.dtype)
            for j in range(sample_length):
                sample[j] = data[indices[i, j]]
            
            # Compute the statistic based on the function ID
            if statistic_func_id == 1:  # mean
                bootstrap_stats[i] = np.mean(sample)
            elif statistic_func_id == 2:  # variance
                bootstrap_stats[i] = np.var(sample)
            elif statistic_func_id == 3:  # median
                bootstrap_stats[i] = np.median(sample)
            elif statistic_func_id == 4:  # max
                bootstrap_stats[i] = np.max(sample)
            elif statistic_func_id == 5:  # min
                bootstrap_stats[i] = np.min(sample)
        else:  # data.ndim == 2
            n_vars = data.shape[1]
            sample = np.zeros((sample_length, n_vars), dtype=data.dtype)
            for j in range(sample_length):
                sample[j, :] = data[indices[i, j], :]
            
            # Compute the statistic based on the function ID
            for v in range(n_vars):
                if statistic_func_id == 1:  # mean
                    bootstrap_stats[i, v] = np.mean(sample[:, v])
                elif statistic_func_id == 2:  # variance
                    bootstrap_stats[i, v] = np.var(sample[:, v])
                elif statistic_func_id == 3:  # median
                    bootstrap_stats[i, v] = np.median(sample[:, v])
                elif statistic_func_id == 4:  # max
                    bootstrap_stats[i, v] = np.max(sample[:, v])
                elif statistic_func_id == 5:  # min
                    bootstrap_stats[i, v] = np.min(sample[:, v])
    
    return bootstrap_stats


@jit(nopython=True, cache=True)
def compute_confidence_interval(
    bootstrap_statistics: np.ndarray,
    confidence_level: float
) -> np.ndarray:
    """
    Compute confidence intervals from bootstrap statistics.
    
    This function computes confidence intervals from bootstrap statistics using
    the percentile method. It is accelerated using Numba's @jit decorator for
    improved performance.
    
    Args:
        bootstrap_statistics: Bootstrap statistics
        confidence_level: Confidence level (between 0 and 1)
        
    Returns:
        np.ndarray: Confidence intervals [lower, upper]
    """
    alpha = 1.0 - confidence_level
    lower_percentile = alpha / 2.0 * 100.0
    upper_percentile = (1.0 - alpha / 2.0) * 100.0
    
    # Handle different dimensions of bootstrap_statistics
    if bootstrap_statistics.ndim == 1:
        # For 1D statistics (scalar statistic)
        sorted_stats = np.sort(bootstrap_statistics)
        n = len(sorted_stats)
        
        # Compute indices for percentiles
        lower_idx = int(np.floor(n * lower_percentile / 100.0))
        upper_idx = int(np.ceil(n * upper_percentile / 100.0)) - 1
        
        # Ensure indices are within bounds
        lower_idx = max(0, min(lower_idx, n - 1))
        upper_idx = max(0, min(upper_idx, n - 1))
        
        # Get confidence interval bounds
        lower = sorted_stats[lower_idx]
        upper = sorted_stats[upper_idx]
        
        return np.array([lower, upper])
    else:
        # For multi-dimensional statistics
        n_dims = bootstrap_statistics.shape[1]
        confidence_intervals = np.zeros((2, n_dims))
        
        for i in range(n_dims):
            sorted_stats = np.sort(bootstrap_statistics[:, i])
            n = len(sorted_stats)
            
            # Compute indices for percentiles
            lower_idx = int(np.floor(n * lower_percentile / 100.0))
            upper_idx = int(np.ceil(n * upper_percentile / 100.0)) - 1
            
            # Ensure indices are within bounds
            lower_idx = max(0, min(lower_idx, n - 1))
            upper_idx = max(0, min(upper_idx, n - 1))
            
            # Get confidence interval bounds
            confidence_intervals[0, i] = sorted_stats[lower_idx]
            confidence_intervals[1, i] = sorted_stats[upper_idx]
        
        return confidence_intervals


@jit(nopython=True, cache=True)
def compute_p_value(
    bootstrap_statistics: np.ndarray,
    original_statistic: Union[float, np.ndarray],
    alternative: str
) -> float:
    """
    Compute bootstrap p-value.
    
    This function computes the bootstrap p-value based on the original statistic
    and bootstrap statistics. It is accelerated using Numba's @jit decorator for
    improved performance.
    
    Args:
        bootstrap_statistics: Bootstrap statistics
        original_statistic: Statistic computed on the original data
        alternative: Alternative hypothesis ('two-sided', 'greater', or 'less')
        
    Returns:
        float: Bootstrap p-value
    """
    # Convert alternative to a numeric code for Numba compatibility
    alt_code = 0  # two-sided
    if alternative == 'greater':
        alt_code = 1
    elif alternative == 'less':
        alt_code = 2
    
    # Compute p-value based on the alternative hypothesis
    if alt_code == 0:  # two-sided
        if isinstance(original_statistic, float):
            p_value = np.mean(np.abs(bootstrap_statistics) >= np.abs(original_statistic))
        else:
            p_value = np.mean(np.abs(bootstrap_statistics) >= np.abs(original_statistic[0]))
    elif alt_code == 1:  # greater
        if isinstance(original_statistic, float):
            p_value = np.mean(bootstrap_statistics >= original_statistic)
        else:
            p_value = np.mean(bootstrap_statistics >= original_statistic[0])
    else:  # less
        if isinstance(original_statistic, float):
            p_value = np.mean(bootstrap_statistics <= original_statistic)
        else:
            p_value = np.mean(bootstrap_statistics <= original_statistic[0])
    
    return float(p_value)


@jit(nopython=True, cache=True)
def generate_moving_block_bootstrap_indices(
    data_length: int,
    n_bootstraps: int,
    block_length: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate moving block bootstrap indices using Numba acceleration.
    
    This function implements the core algorithm for generating moving block bootstrap
    indices. Unlike the circular block bootstrap, the moving block bootstrap does not
    wrap around the end of the data. It is accelerated using Numba's @jit decorator
    for improved performance.
    
    Args:
        data_length: Length of the original data
        n_bootstraps: Number of bootstrap samples to generate
        block_length: Length of each block
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
    """
    # Initialize random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate number of possible block starts
    n_possible_blocks = data_length - block_length + 1
    
    # Calculate number of blocks needed to cover the data length
    n_blocks = int(np.ceil(data_length / block_length))
    
    # Initialize indices array
    indices = np.zeros((n_bootstraps, data_length), dtype=np.int64)
    
    # Generate bootstrap indices for each bootstrap sample
    for i in range(n_bootstraps):
        # Generate random starting positions for blocks
        block_starts = np.random.randint(0, n_possible_blocks, size=n_blocks)
        
        # Fill the indices array with blocks
        idx = 0
        for start in block_starts:
            # Add indices from the current block
            for j in range(block_length):
                if idx >= data_length:
                    break
                indices[i, idx] = start + j
                idx += 1
    
    return indices


@jit(nopython=True, cache=True)
def generate_circular_block_bootstrap_indices(
    data_length: int,
    n_bootstraps: int,
    block_length: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate circular block bootstrap indices using Numba acceleration.
    
    This function is an alias for generate_block_bootstrap_indices, which
    implements the circular block bootstrap algorithm. It is provided for
    API consistency.
    
    Args:
        data_length: Length of the original data
        n_bootstraps: Number of bootstrap samples to generate
        block_length: Length of each block
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
    """
    return generate_block_bootstrap_indices(
        data_length=data_length,
        n_bootstraps=n_bootstraps,
        block_length=block_length,
        seed=seed
    )


@jit(nopython=True, cache=True)
def compute_bootstrap_correlation(
    data: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Compute bootstrap correlations using Numba acceleration.
    
    This function efficiently computes the correlation matrix for each bootstrap sample.
    It is optimized for 2D data arrays.
    
    Args:
        data: Original data array (2D)
        indices: Bootstrap indices with shape (n_bootstraps, sample_length)
        
    Returns:
        np.ndarray: Bootstrap correlation matrices with shape (n_bootstraps, n_vars, n_vars)
    """
    n_bootstraps, sample_length = indices.shape
    
    # Ensure data is 2D
    if data.ndim != 2:
        raise ValueError("Data must be 2D for correlation computation")
    
    n_vars = data.shape[1]
    bootstrap_corrs = np.zeros((n_bootstraps, n_vars, n_vars), dtype=np.float64)
    
    for i in range(n_bootstraps):
        # Extract the bootstrap sample
        sample = np.zeros((sample_length, n_vars), dtype=data.dtype)
        for j in range(sample_length):
            sample[j, :] = data[indices[i, j], :]
        
        # Compute means for each variable
        means = np.zeros(n_vars)
        for v in range(n_vars):
            means[v] = np.mean(sample[:, v])
        
        # Compute standard deviations for each variable
        stds = np.zeros(n_vars)
        for v in range(n_vars):
            sum_sq_diff = 0.0
            for j in range(sample_length):
                diff = sample[j, v] - means[v]
                sum_sq_diff += diff * diff
            stds[v] = np.sqrt(sum_sq_diff / sample_length)
        
        # Compute correlation matrix
        for v1 in range(n_vars):
            # Diagonal elements are always 1
            bootstrap_corrs[i, v1, v1] = 1.0
            
            # Compute off-diagonal elements
            for v2 in range(v1 + 1, n_vars):
                # Skip if either standard deviation is zero
                if stds[v1] == 0.0 or stds[v2] == 0.0:
                    bootstrap_corrs[i, v1, v2] = 0.0
                    bootstrap_corrs[i, v2, v1] = 0.0
                    continue
                
                # Compute covariance
                cov = 0.0
                for j in range(sample_length):
                    cov += (sample[j, v1] - means[v1]) * (sample[j, v2] - means[v2])
                cov /= sample_length
                
                # Compute correlation
                corr = cov / (stds[v1] * stds[v2])
                
                # Ensure correlation is in [-1, 1]
                corr = max(-1.0, min(1.0, corr))
                
                # Store correlation (matrix is symmetric)
                bootstrap_corrs[i, v1, v2] = corr
                bootstrap_corrs[i, v2, v1] = corr
    
    return bootstrap_corrs
