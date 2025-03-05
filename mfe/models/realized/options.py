'''
Configuration options for realized volatility estimators.

This module provides a comprehensive set of configuration options for all realized
volatility estimators in the MFE Toolbox. It centralizes parameter management across
estimators, ensuring consistent interfaces and behavior while providing robust validation.

The module implements a hierarchy of dataclass-based option containers with proper
inheritance, type validation, and serialization capabilities. Each estimator type
has its own specialized configuration class that inherits from a base configuration,
adding type-specific parameters and validation logic.

Key features:
- Type-safe parameter containers using Python dataclasses
- Comprehensive validation logic for all parameters
- Inheritance hierarchy for estimator-specific options
- Serialization to/from various formats (dict, JSON, YAML)
- Post-initialization validation hooks
- Default values for all parameters
''' 

import json
import logging
import warnings
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, Set, 
    Tuple, Type, TypeVar, Union, cast, get_type_hints, overload
)

import numpy as np

from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import DimensionError, NumericError
from .base import RealizedEstimatorConfig

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.options")

# Type variables for generic option classes
T = TypeVar('T')  # Generic type for options
O = TypeVar('O', bound='BaseOptions')  # Generic type for option subclasses


class ReturnType(Enum):
    """Enumeration of return types for realized volatility estimators."""
    LOG = "log"
    SIMPLE = "simple"


class TimeUnit(Enum):
    """Enumeration of time units for high-frequency data."""
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    TICKS = "ticks"


class KernelType(Enum):
    """Enumeration of kernel types for kernel-based estimators."""
    BARTLETT = "bartlett"
    PARZEN = "parzen"
    TUKEY_HANNING = "tukey-hanning"
    QUADRATIC = "quadratic"
    FLAT_TOP = "flat-top"


@dataclass
class BaseOptions(ParameterBase):
    """Base class for all realized volatility estimator options.
    
    This class provides common configuration parameters and validation logic
    that apply to all realized volatility estimators.
    
    Attributes:
        sampling_frequency: Sampling frequency for price data (e.g., '5min', 300)
        annualize: Whether to annualize the volatility estimate
        annualization_factor: Factor to use for annualization (e.g., 252 for daily data)
        return_type: Type of returns to compute ('log', 'simple')
        time_unit: Unit of time for high-frequency data ('seconds', 'minutes', etc.)
    """
    
    sampling_frequency: Optional[Union[str, float, int]] = None
    annualize: bool = False
    annualization_factor: float = 252.0
    return_type: Union[str, ReturnType] = "log"
    time_unit: Union[str, TimeUnit] = "seconds"
    
    def __post_init__(self) -> None:
        """Validate options after initialization."""
        # Convert enum values to strings if needed
        if isinstance(self.return_type, ReturnType):
            self.return_type = self.return_type.value
        
        if isinstance(self.time_unit, TimeUnit):
            self.time_unit = self.time_unit.value
        
        # Validate options
        self.validate()
    
    def validate(self) -> None:
        """Validate option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Validate annualization_factor if annualize is True
        if self.annualize:
            validate_positive(self.annualization_factor, "annualization_factor")
        
        # Validate return_type
        valid_return_types = [rt.value for rt in ReturnType] + [rt.name.lower() for rt in ReturnType]
        if self.return_type not in valid_return_types:
            raise ParameterError(
                f"return_type must be one of {valid_return_types}, got {self.return_type}"
            )
        
        # Validate time_unit
        valid_time_units = [tu.value for tu in TimeUnit] + [tu.name.lower() for tu in TimeUnit]
        if self.time_unit not in valid_time_units:
            raise ParameterError(
                f"time_unit must be one of {valid_time_units}, got {self.time_unit}"
            )
        
        # Validate sampling_frequency if provided
        if self.sampling_frequency is not None:
            if isinstance(self.sampling_frequency, (int, float)):
                validate_positive(self.sampling_frequency, "sampling_frequency")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert options to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of options
        """
        return asdict(self)
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert options to a JSON string.
        
        Args:
            indent: Number of spaces for indentation (None for compact format)
            
        Returns:
            str: JSON representation of options
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save options to a file.
        
        Args:
            file_path: Path to save the options to (JSON or YAML format)
            
        Raises:
            ValueError: If the file format is not supported
        """
        file_path = Path(file_path)
        
        # Determine file format from extension
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                f.write(self.to_json())
        elif file_path.suffix.lower() in ('.yaml', '.yml'):
            try:
                import yaml
                with open(file_path, 'w') as f:
                    yaml.dump(self.to_dict(), f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML serialization")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @classmethod
    def from_dict(cls: Type[O], data: Dict[str, Any]) -> O:
        """Create options from a dictionary.
        
        Args:
            data: Dictionary containing option values
            
        Returns:
            O: Options object
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls: Type[O], json_str: str) -> O:
        """Create options from a JSON string.
        
        Args:
            json_str: JSON string containing option values
            
        Returns:
            O: Options object
            
        Raises:
            json.JSONDecodeError: If the JSON string is invalid
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def load(cls: Type[O], file_path: Union[str, Path]) -> O:
        """Load options from a file.
        
        Args:
            file_path: Path to load the options from (JSON or YAML format)
            
        Returns:
            O: Options object
            
        Raises:
            ValueError: If the file format is not supported
            FileNotFoundError: If the file does not exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file format from extension
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                return cls.from_json(f.read())
        elif file_path.suffix.lower() in ('.yaml', '.yml'):
            try:
                import yaml
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                return cls.from_dict(data)
            except ImportError:
                raise ImportError("PyYAML is required for YAML deserialization")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def copy(self: O) -> O:
        """Create a copy of the options.
        
        Returns:
            O: Copy of the options
        """
        return type(self)(**self.to_dict())
    
    def update(self, **kwargs: Any) -> None:
        """Update options with new values.
        
        Args:
            **kwargs: New option values
            
        Raises:
            ParameterError: If the updated options violate constraints
        """
        # Update attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown option: {key}")
        
        # Validate updated options
        self.validate()


@dataclass
class SubsamplingOptions(BaseOptions):
    """Options for subsampling in realized volatility estimators.
    
    This class extends BaseOptions to provide configuration parameters
    for subsampling-based realized volatility estimators.
    
    Attributes:
        use_subsampling: Whether to use subsampling for noise reduction
        subsampling_factor: Factor for subsampling (number of subsamples)
        subsampling_method: Method for subsampling ('regular', 'random', 'jittered')
        combine_method: Method for combining subsamples ('average', 'median')
    """
    
    use_subsampling: bool = False
    subsampling_factor: int = 1
    subsampling_method: str = "regular"
    combine_method: str = "average"
    
    def validate(self) -> None:
        """Validate subsampling option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation
        super().validate()
        
        # Validate subsampling_factor if use_subsampling is True
        if self.use_subsampling:
            if not isinstance(self.subsampling_factor, int):
                raise ParameterError(
                    f"subsampling_factor must be an integer, got {type(self.subsampling_factor)}"
                )
            validate_positive(self.subsampling_factor, "subsampling_factor")
        
        # Validate subsampling_method
        valid_methods = ["regular", "random", "jittered"]
        if self.subsampling_method not in valid_methods:
            raise ParameterError(
                f"subsampling_method must be one of {valid_methods}, got {self.subsampling_method}"
            )
        
        # Validate combine_method
        valid_combine_methods = ["average", "median"]
        if self.combine_method not in valid_combine_methods:
            raise ParameterError(
                f"combine_method must be one of {valid_combine_methods}, got {self.combine_method}"
            )


@dataclass
class NoiseRobustOptions(SubsamplingOptions):
    """Options for noise-robust realized volatility estimators.
    
    This class extends SubsamplingOptions to provide configuration parameters
    for noise-robust realized volatility estimators.
    
    Attributes:
        apply_noise_correction: Whether to apply microstructure noise correction
        noise_variance_estimator: Method for estimating noise variance
        bias_correction: Whether to apply bias correction
        jitter_correction: Whether to apply jitter correction
    """
    
    apply_noise_correction: bool = False
    noise_variance_estimator: str = "autocovariance"
    bias_correction: bool = True
    jitter_correction: bool = False
    
    def validate(self) -> None:
        """Validate noise-robust option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation
        super().validate()
        
        # Validate noise_variance_estimator
        valid_estimators = ["autocovariance", "first-order", "second-order", "optimal"]
        if self.noise_variance_estimator not in valid_estimators:
            raise ParameterError(
                f"noise_variance_estimator must be one of {valid_estimators}, "
                f"got {self.noise_variance_estimator}"
            )


@dataclass
class JumpRobustOptions(BaseOptions):
    """Options for jump-robust realized volatility estimators.
    
    This class extends BaseOptions to provide configuration parameters
    for jump-robust realized volatility estimators.
    
    Attributes:
        detect_jumps: Whether to detect and remove jumps
        jump_threshold_multiplier: Multiplier for jump detection threshold
        jump_detection_method: Method for detecting jumps
        truncation_level: Truncation level for threshold-based estimators
    """
    
    detect_jumps: bool = False
    jump_threshold_multiplier: float = 3.0
    jump_detection_method: str = "threshold"
    truncation_level: Optional[float] = None
    
    def validate(self) -> None:
        """Validate jump-robust option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation
        super().validate()
        
        # Validate jump_threshold_multiplier
        validate_positive(self.jump_threshold_multiplier, "jump_threshold_multiplier")
        
        # Validate jump_detection_method
        valid_methods = ["threshold", "lee-mykland", "ait-sahalia", "bipower"]
        if self.jump_detection_method not in valid_methods:
            raise ParameterError(
                f"jump_detection_method must be one of {valid_methods}, "
                f"got {self.jump_detection_method}"
            )
        
        # Validate truncation_level if provided
        if self.truncation_level is not None:
            validate_positive(self.truncation_level, "truncation_level")


@dataclass
class KernelOptions(NoiseRobustOptions):
    """Options for kernel-based realized volatility estimators.
    
    This class extends NoiseRobustOptions to provide configuration parameters
    for kernel-based realized volatility estimators.
    
    Attributes:
        kernel_type: Type of kernel function
        bandwidth: Bandwidth parameter for kernel function
        max_lags: Maximum number of lags to consider
        auto_bandwidth: Whether to automatically determine optimal bandwidth
        bandwidth_method: Method for determining optimal bandwidth
    """
    
    kernel_type: Union[str, KernelType] = "bartlett"
    bandwidth: Optional[float] = None
    max_lags: Optional[int] = None
    auto_bandwidth: bool = True
    bandwidth_method: str = "optimal"
    
    def __post_init__(self) -> None:
        """Validate options after initialization."""
        # Convert enum values to strings if needed
        if isinstance(self.kernel_type, KernelType):
            self.kernel_type = self.kernel_type.value
        
        # Call parent validation
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate kernel option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation
        super().validate()
        
        # Validate kernel_type
        valid_kernels = [kt.value for kt in KernelType] + [kt.name.lower() for kt in KernelType]
        valid_kernels.extend(["tukey", "hanning"])  # Aliases
        
        if self.kernel_type not in valid_kernels:
            raise ParameterError(
                f"kernel_type must be one of {valid_kernels}, got {self.kernel_type}"
            )
        
        # Validate bandwidth if provided
        if self.bandwidth is not None:
            validate_positive(self.bandwidth, "bandwidth")
        
        # Validate max_lags if provided
        if self.max_lags is not None:
            if not isinstance(self.max_lags, int):
                raise ParameterError(f"max_lags must be an integer, got {type(self.max_lags)}")
            validate_positive(self.max_lags, "max_lags")
        
        # Validate bandwidth_method
        valid_methods = ["optimal", "asymptotic", "improved-asymptotic", "manual"]
        if self.bandwidth_method not in valid_methods:
            raise ParameterError(
                f"bandwidth_method must be one of {valid_methods}, got {self.bandwidth_method}"
            )


@dataclass
class MultiscaleOptions(NoiseRobustOptions):
    """Options for multiscale realized volatility estimators.
    
    This class extends NoiseRobustOptions to provide configuration parameters
    for multiscale realized volatility estimators.
    
    Attributes:
        num_scales: Number of scales to use
        scale_factor: Factor between consecutive scales
        min_scale: Minimum scale (as a fraction of data length)
        max_scale: Maximum scale (as a fraction of data length)
        weight_function: Function for weighting different scales
    """
    
    num_scales: int = 5
    scale_factor: float = 2.0
    min_scale: float = 0.01
    max_scale: float = 0.5
    weight_function: str = "optimal"
    
    def validate(self) -> None:
        """Validate multiscale option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation
        super().validate()
        
        # Validate num_scales
        if not isinstance(self.num_scales, int):
            raise ParameterError(f"num_scales must be an integer, got {type(self.num_scales)}")
        validate_positive(self.num_scales, "num_scales")
        
        # Validate scale_factor
        validate_positive(self.scale_factor, "scale_factor")
        
        # Validate min_scale and max_scale
        validate_positive(self.min_scale, "min_scale")
        validate_positive(self.max_scale, "max_scale")
        
        if self.min_scale >= self.max_scale:
            raise ParameterError(
                f"min_scale ({self.min_scale}) must be less than max_scale ({self.max_scale})"
            )
        
        # Validate weight_function
        valid_functions = ["optimal", "equal", "linear", "quadratic", "exponential"]
        if self.weight_function not in valid_functions:
            raise ParameterError(
                f"weight_function must be one of {valid_functions}, got {self.weight_function}"
            )


@dataclass
class PreaveragingOptions(NoiseRobustOptions):
    """Options for preaveraging-based realized volatility estimators.
    
    This class extends NoiseRobustOptions to provide configuration parameters
    for preaveraging-based realized volatility estimators.
    
    Attributes:
        window_size: Size of the preaveraging window
        auto_window_size: Whether to automatically determine optimal window size
        theta: Preaveraging parameter (typically around 0.5)
        kernel_function: Kernel function for preaveraging
    """
    
    window_size: Optional[int] = None
    auto_window_size: bool = True
    theta: float = 0.5
    kernel_function: str = "triangular"
    
    def validate(self) -> None:
        """Validate preaveraging option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation
        super().validate()
        
        # Validate window_size if provided
        if self.window_size is not None:
            if not isinstance(self.window_size, int):
                raise ParameterError(f"window_size must be an integer, got {type(self.window_size)}")
            validate_positive(self.window_size, "window_size")
        
        # Validate theta
        validate_positive(self.theta, "theta")
        
        # Validate kernel_function
        valid_functions = ["triangular", "gaussian", "uniform", "epanechnikov"]
        if self.kernel_function not in valid_functions:
            raise ParameterError(
                f"kernel_function must be one of {valid_functions}, got {self.kernel_function}"
            )


@dataclass
class RangeOptions(BaseOptions):
    """Options for range-based realized volatility estimators.
    
    This class extends BaseOptions to provide configuration parameters
    for range-based realized volatility estimators.
    
    Attributes:
        scaling_factor: Scaling factor for range-based estimators
        use_log_ranges: Whether to use log ranges
        range_type: Type of range to compute
    """
    
    scaling_factor: float = 4.0 * np.log(2.0)
    use_log_ranges: bool = True
    range_type: str = "parkinson"
    
    def validate(self) -> None:
        """Validate range option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation
        super().validate()
        
        # Validate scaling_factor
        validate_positive(self.scaling_factor, "scaling_factor")
        
        # Validate range_type
        valid_types = ["parkinson", "garman-klass", "rogers-satchell", "yang-zhang"]
        if self.range_type not in valid_types:
            raise ParameterError(
                f"range_type must be one of {valid_types}, got {self.range_type}"
            )


@dataclass
class MultivariateOptions(BaseOptions):
    """Options for multivariate realized volatility estimators.
    
    This class extends BaseOptions to provide configuration parameters
    for multivariate realized volatility estimators.
    
    Attributes:
        synchronization_method: Method for synchronizing multiple time series
        positive_definite: Whether to enforce positive definiteness
        regularization: Whether to apply regularization
        regularization_method: Method for regularization
        regularization_lambda: Regularization parameter
    """
    
    synchronization_method: str = "refresh-time"
    positive_definite: bool = True
    regularization: bool = False
    regularization_method: str = "shrinkage"
    regularization_lambda: float = 0.0
    
    def validate(self) -> None:
        """Validate multivariate option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation
        super().validate()
        
        # Validate synchronization_method
        valid_methods = ["refresh-time", "previous-tick", "linear-interpolation", "hayashi-yoshida"]
        if self.synchronization_method not in valid_methods:
            raise ParameterError(
                f"synchronization_method must be one of {valid_methods}, "
                f"got {self.synchronization_method}"
            )
        
        # Validate regularization_method
        valid_methods = ["shrinkage", "thresholding", "eigenvalue", "none"]
        if self.regularization_method not in valid_methods:
            raise ParameterError(
                f"regularization_method must be one of {valid_methods}, "
                f"got {self.regularization_method}"
            )
        
        # Validate regularization_lambda
        validate_non_negative(self.regularization_lambda, "regularization_lambda")


@dataclass
class MultivariateKernelOptions(KernelOptions, MultivariateOptions):
    """Options for multivariate kernel-based realized volatility estimators.
    
    This class extends both KernelOptions and MultivariateOptions to provide
    configuration parameters for multivariate kernel-based realized volatility estimators.
    
    Attributes:
        multivariate_kernel_type: Type of multivariate kernel function
        separate_bandwidths: Whether to use separate bandwidths for each asset
    """
    
    multivariate_kernel_type: str = "product"
    separate_bandwidths: bool = False
    
    def validate(self) -> None:
        """Validate multivariate kernel option constraints.
        
        Raises:
            ParameterError: If option constraints are violated
        """
        # Call parent validation from both parent classes
        KernelOptions.validate(self)
        MultivariateOptions.validate(self)
        
        # Validate multivariate_kernel_type
        valid_types = ["product", "sum", "matrix"]
        if self.multivariate_kernel_type not in valid_types:
            raise ParameterError(
                f"multivariate_kernel_type must be one of {valid_types}, "
                f"got {self.multivariate_kernel_type}"
            )


# Create a mapping from estimator type to options class
ESTIMATOR_OPTIONS_MAP = {
    "RealizedVariance": BaseOptions,
    "BiPowerVariation": JumpRobustOptions,
    "RealizedKernel": KernelOptions,
    "RealizedSemivariance": BaseOptions,
    "RealizedCovariance": MultivariateOptions,
    "MultivariateRealizedKernel": MultivariateKernelOptions,
    "RealizedRange": RangeOptions,
    "RealizedQuarticity": BaseOptions,
    "TwoScaleRealizedVariance": MultiscaleOptions,
    "MultiscaleRealizedVariance": MultiscaleOptions,
    "QMLERealizedVariance": NoiseRobustOptions,
    "ThresholdMultipowerVariation": JumpRobustOptions,
    "ThresholdRealizedVariance": JumpRobustOptions,
    "PreaveragedBiPowerVariation": PreaveragingOptions,
    "PreaveragedRealizedVariance": PreaveragingOptions
}


def create_options(estimator_type: str, **kwargs: Any) -> BaseOptions:
    """Create options for the specified estimator type.
    
    Args:
        estimator_type: Type of estimator to create options for
        **kwargs: Option values to set
        
    Returns:
        BaseOptions: Options for the specified estimator type
        
    Raises:
        ValueError: If estimator_type is not recognized
    """
    if estimator_type not in ESTIMATOR_OPTIONS_MAP:
        valid_types = list(ESTIMATOR_OPTIONS_MAP.keys())
        raise ValueError(
            f"Unrecognized estimator type: {estimator_type}. "
            f"Supported types are {valid_types}."
        )
    
    options_class = ESTIMATOR_OPTIONS_MAP[estimator_type]
    return options_class(**kwargs)


def convert_to_realized_estimator_config(options: BaseOptions) -> RealizedEstimatorConfig:
    """Convert options to a RealizedEstimatorConfig object.
    
    This function converts a BaseOptions object (or any of its subclasses)
    to a RealizedEstimatorConfig object that can be used with realized
    volatility estimators.
    
    Args:
        options: Options to convert
        
    Returns:
        RealizedEstimatorConfig: Converted options
    """
    # Extract common parameters from options
    config_dict = {
        "sampling_frequency": options.sampling_frequency,
        "annualize": options.annualize,
        "annualization_factor": options.annualization_factor,
        "return_type": options.return_type,
        "time_unit": options.time_unit
    }
    
    # Add subsampling parameters if available
    if hasattr(options, "use_subsampling"):
        config_dict["use_subsampling"] = options.use_subsampling
        config_dict["subsampling_factor"] = options.subsampling_factor
    
    # Add noise correction parameters if available
    if hasattr(options, "apply_noise_correction"):
        config_dict["apply_noise_correction"] = options.apply_noise_correction
    
    # Add kernel parameters if available
    if hasattr(options, "kernel_type"):
        config_dict["kernel_type"] = options.kernel_type
        config_dict["bandwidth"] = options.bandwidth
    
    # Create and return RealizedEstimatorConfig
    return RealizedEstimatorConfig(**config_dict)


def convert_from_realized_estimator_config(config: RealizedEstimatorConfig, 
                                          estimator_type: str) -> BaseOptions:
    """Convert a RealizedEstimatorConfig object to options for the specified estimator type.
    
    This function converts a RealizedEstimatorConfig object to a BaseOptions object
    (or one of its subclasses) for the specified estimator type.
    
    Args:
        config: RealizedEstimatorConfig to convert
        estimator_type: Type of estimator to create options for
        
    Returns:
        BaseOptions: Options for the specified estimator type
        
    Raises:
        ValueError: If estimator_type is not recognized
    """
    # Extract parameters from config
    config_dict = {
        "sampling_frequency": config.sampling_frequency,
        "annualize": config.annualize,
        "annualization_factor": config.annualization_factor,
        "return_type": config.return_type,
        "time_unit": config.time_unit
    }
    
    # Add subsampling parameters if available
    if hasattr(config, "use_subsampling"):
        config_dict["use_subsampling"] = config.use_subsampling
        config_dict["subsampling_factor"] = config.subsampling_factor
    
    # Add noise correction parameters if available
    if hasattr(config, "apply_noise_correction"):
        config_dict["apply_noise_correction"] = config.apply_noise_correction
    
    # Add kernel parameters if available
    if hasattr(config, "kernel_type"):
        config_dict["kernel_type"] = config.kernel_type
        config_dict["bandwidth"] = config.bandwidth
    
    # Create and return options for the specified estimator type
    return create_options(estimator_type, **config_dict)


def load_default_options(estimator_type: str) -> BaseOptions:
    """Load default options for the specified estimator type.
    
    Args:
        estimator_type: Type of estimator to load default options for
        
    Returns:
        BaseOptions: Default options for the specified estimator type
        
    Raises:
        ValueError: If estimator_type is not recognized
    """
    return create_options(estimator_type)


def save_options_to_file(options: BaseOptions, file_path: Union[str, Path]) -> None:
    """Save options to a file.
    
    Args:
        options: Options to save
        file_path: Path to save the options to (JSON or YAML format)
        
    Raises:
        ValueError: If the file format is not supported
    """
    options.save(file_path)


def load_options_from_file(file_path: Union[str, Path], 
                          estimator_type: Optional[str] = None) -> BaseOptions:
    """Load options from a file.
    
    Args:
        file_path: Path to load the options from (JSON or YAML format)
        estimator_type: Type of estimator to load options for (if None, determined from file)
        
    Returns:
        BaseOptions: Options loaded from the file
        
    Raises:
        ValueError: If the file format is not supported or if estimator_type is required but not provided
        FileNotFoundError: If the file does not exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file format from extension
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
    elif file_path.suffix.lower() in ('.yaml', '.yml'):
        try:
            import yaml
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required for YAML deserialization")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Determine estimator type if not provided
    if estimator_type is None:
        if "estimator_type" in data:
            estimator_type = data["estimator_type"]
        else:
            raise ValueError(
                "estimator_type not provided and not found in file. "
                "Please specify the estimator type."
            )
    
    # Create options for the specified estimator type
    return create_options(estimator_type, **data)
