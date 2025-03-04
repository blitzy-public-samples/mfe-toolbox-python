'''
Configuration management system for the MFE Toolbox.

This module provides a comprehensive configuration system for the MFE Toolbox,
allowing users to customize the behavior of various components through a
hierarchical configuration structure. It supports configuration via environment
variables, user-specific configuration files, and runtime modifications.

The configuration system follows a layered approach:
1. Default configurations built into the package
2. User-specific configuration files
3. Environment variables
4. Runtime modifications

This approach ensures that the toolbox can be customized without modifying
source code, while maintaining sensible defaults for all settings.

Key features:
- Type-safe configuration with validation
- Environment variable integration
- User-specific configuration files
- Hierarchical configuration structure
- Runtime configuration modification
- Configuration reset capabilities
'''

import os
import json
import logging
import platform
import tempfile
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast, get_type_hints

import numpy as np

from .exceptions import ConfigurationError
from .types import ConfigDict, ConfigPath, LogLevel

# Set up module-level logger
logger = logging.getLogger("mfe.core.config")

# Constants for configuration paths and environment variables
CONFIG_ENV_PREFIX = "MFE_"
DEFAULT_CONFIG_FILENAME = "mfe_config.json"
USER_CONFIG_DIR_ENV = "MFE_CONFIG_DIR"


class ConfigSection(Enum):
    """Enumeration of configuration sections."""
    CORE = "core"
    PERFORMANCE = "performance"
    NUMERICAL = "numerical"
    OUTPUT = "output"
    MODELS = "models"
    UI = "ui"
    LOGGING = "logging"
    PATHS = "paths"
    FEATURES = "features"


class ConfigValueType(Enum):
    """Enumeration of configuration value types."""
    BOOLEAN = auto()
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    PATH = auto()
    LIST = auto()
    DICT = auto()
    ENUM = auto()


@dataclass(frozen=True)
class ConfigOption:
    """
    Definition of a configuration option with metadata.
    
    Attributes:
        name: The name of the configuration option
        default_value: The default value for the option
        value_type: The type of the configuration value
        description: Description of the configuration option
        validation: Optional validation function for the option
        env_var: Optional environment variable name to override this option
        deprecated: Whether this option is deprecated
        section: The configuration section this option belongs to
    """
    name: str
    default_value: Any
    value_type: ConfigValueType
    description: str
    validation: Optional[callable] = None
    env_var: Optional[str] = None
    deprecated: bool = False
    section: ConfigSection = ConfigSection.CORE
    
    def __post_init__(self):
        """Validate the configuration option definition."""
        if self.env_var is None:
            # Create standard environment variable name if not specified
            env_var = f"{CONFIG_ENV_PREFIX}{self.section.value.upper()}_{self.name.upper()}"
            object.__setattr__(self, "env_var", env_var)


@dataclass
class CoreConfig:
    """
    Core configuration settings for the MFE Toolbox.
    
    This dataclass defines the core configuration options that control
    the general behavior of the toolbox.
    
    Attributes:
        version: The version of the configuration format
        user_config_dir: Directory for user-specific configuration files
        enable_numba: Whether to use Numba acceleration when available
        use_async: Whether to use asynchronous processing for long operations
        show_deprecation_warnings: Whether to show deprecation warnings
        random_seed: Seed for random number generation (None for random seed)
    """
    version: str = "4.0.0"
    user_config_dir: Path = field(default_factory=lambda: Path.home() / ".mfe")
    enable_numba: bool = True
    use_async: bool = True
    show_deprecation_warnings: bool = True
    random_seed: Optional[int] = None


@dataclass
class PerformanceConfig:
    """
    Performance configuration settings for the MFE Toolbox.
    
    This dataclass defines configuration options related to performance
    optimization and resource usage.
    
    Attributes:
        cache_results: Whether to cache computation results
        cache_dir: Directory for cached results
        cache_size_limit_mb: Maximum cache size in megabytes
        parallel_threshold: Minimum data size to trigger parallel processing
        max_workers: Maximum number of worker threads/processes
        optimize_memory: Whether to optimize for memory usage (vs. speed)
        jit_cache: Whether to cache JIT-compiled functions
        vectorize_threshold: Minimum array size for vectorized operations
    """
    cache_results: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "mfe_cache")
    cache_size_limit_mb: int = 1024
    parallel_threshold: int = 10000
    max_workers: int = 4
    optimize_memory: bool = False
    jit_cache: bool = True
    vectorize_threshold: int = 1000


@dataclass
class NumericalConfig:
    """
    Numerical configuration settings for the MFE Toolbox.
    
    This dataclass defines configuration options related to numerical
    precision, tolerances, and algorithm parameters.
    
    Attributes:
        optimization_method: Default optimization method
        optimization_tol: Optimization convergence tolerance
        max_iterations: Maximum number of iterations for optimization
        gradient_method: Method for numerical gradient computation
        hessian_method: Method for numerical Hessian computation
        finite_difference_step: Step size for finite difference methods
        eigenvalue_tolerance: Tolerance for eigenvalue computations
        matrix_rank_tolerance: Tolerance for matrix rank determination
        default_float_type: Default floating-point type (float32 or float64)
    """
    optimization_method: str = "BFGS"
    optimization_tol: float = 1e-8
    max_iterations: int = 1000
    gradient_method: str = "central"
    hessian_method: str = "central"
    finite_difference_step: float = 1e-8
    eigenvalue_tolerance: float = 1e-10
    matrix_rank_tolerance: float = 1e-10
    default_float_type: str = "float64"


@dataclass
class OutputConfig:
    """
    Output configuration settings for the MFE Toolbox.
    
    This dataclass defines configuration options related to output
    formatting, display, and export.
    
    Attributes:
        float_precision: Number of decimal places for float display
        display_width: Width for formatted output
        display_max_rows: Maximum number of rows to display
        display_max_columns: Maximum number of columns to display
        plot_style: Default style for plots
        plot_dpi: Default DPI for plots
        plot_figsize: Default figure size for plots
        export_format: Default format for data export
    """
    float_precision: int = 4
    display_width: int = 120
    display_max_rows: int = 100
    display_max_columns: int = 20
    plot_style: str = "default"
    plot_dpi: int = 100
    plot_figsize: Tuple[int, int] = (10, 6)
    export_format: str = "csv"


@dataclass
class ModelsConfig:
    """
    Model-specific configuration settings for the MFE Toolbox.
    
    This dataclass defines configuration options related to model
    estimation, validation, and defaults.
    
    Attributes:
        default_volatility_model: Default univariate volatility model
        default_multivariate_model: Default multivariate volatility model
        default_distribution: Default error distribution
        default_bootstrap_method: Default bootstrap method
        default_realized_estimator: Default realized volatility estimator
        robust_covariance: Whether to use robust covariance estimation
        backcast_method: Method for volatility backcasting
        simulate_method: Method for simulation
        forecast_method: Method for forecasting
    """
    default_volatility_model: str = "GARCH"
    default_multivariate_model: str = "DCC"
    default_distribution: str = "normal"
    default_bootstrap_method: str = "stationary"
    default_realized_estimator: str = "RV"
    robust_covariance: bool = True
    backcast_method: str = "power"
    simulate_method: str = "direct"
    forecast_method: str = "analytic"


@dataclass
class UIConfig:
    """
    User interface configuration settings for the MFE Toolbox.
    
    This dataclass defines configuration options related to the
    graphical user interface components.
    
    Attributes:
        theme: UI theme (light, dark, system)
        font_family: Font family for UI elements
        font_size: Base font size for UI elements
        icon_size: Size of icons in UI
        show_tooltips: Whether to show tooltips
        confirmation_dialogs: Whether to show confirmation dialogs
        progress_updates: Whether to show progress updates
        save_window_positions: Whether to save window positions
    """
    theme: str = "system"
    font_family: str = ""  # Empty string means system default
    font_size: int = 10
    icon_size: int = 16
    show_tooltips: bool = True
    confirmation_dialogs: bool = True
    progress_updates: bool = True
    save_window_positions: bool = True


@dataclass
class LoggingConfig:
    """
    Logging configuration settings for the MFE Toolbox.
    
    This dataclass defines configuration options related to logging
    and diagnostic output.
    
    Attributes:
        log_level: Default logging level
        log_file: Path to log file (None for no file logging)
        log_format: Format string for log messages
        log_date_format: Format string for log message timestamps
        console_logging: Whether to log to console
        file_logging: Whether to log to file
        log_performance: Whether to log performance metrics
        log_numerical_warnings: Whether to log numerical warnings
    """
    log_level: LogLevel = "INFO"
    log_file: Optional[Path] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    console_logging: bool = True
    file_logging: bool = False
    log_performance: bool = False
    log_numerical_warnings: bool = True


@dataclass
class PathsConfig:
    """
    Path configuration settings for the MFE Toolbox.
    
    This dataclass defines configuration options related to file paths
    and directories.
    
    Attributes:
        data_dir: Directory for data files
        results_dir: Directory for results
        temp_dir: Directory for temporary files
        user_scripts_dir: Directory for user scripts
        examples_dir: Directory for example files
    """
    data_dir: Path = field(default_factory=lambda: Path.home() / "mfe_data")
    results_dir: Path = field(default_factory=lambda: Path.home() / "mfe_results")
    temp_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "mfe_temp")
    user_scripts_dir: Path = field(default_factory=lambda: Path.home() / "mfe_scripts")
    examples_dir: Optional[Path] = None  # Will be set during initialization


@dataclass
class FeaturesConfig:
    """
    Feature toggle configuration settings for the MFE Toolbox.
    
    This dataclass defines configuration options that enable or disable
    specific features of the toolbox.
    
    Attributes:
        enable_ui: Whether to enable UI components
        enable_experimental: Whether to enable experimental features
        enable_bootstrap: Whether to enable bootstrap methods
        enable_realized: Whether to enable realized volatility estimators
        enable_multivariate: Whether to enable multivariate models
        enable_cross_section: Whether to enable cross-sectional analysis
    """
    enable_ui: bool = True
    enable_experimental: bool = False
    enable_bootstrap: bool = True
    enable_realized: bool = True
    enable_multivariate: bool = True
    enable_cross_section: bool = True


@dataclass
class MFEConfig:
    """
    Complete configuration for the MFE Toolbox.
    
    This dataclass combines all configuration sections into a single
    comprehensive configuration object.
    
    Attributes:
        core: Core configuration settings
        performance: Performance configuration settings
        numerical: Numerical configuration settings
        output: Output configuration settings
        models: Model-specific configuration settings
        ui: User interface configuration settings
        logging: Logging configuration settings
        paths: Path configuration settings
        features: Feature toggle configuration settings
    """
    core: CoreConfig = field(default_factory=CoreConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)


class ConfigManager:
    """
    Configuration manager for the MFE Toolbox.
    
    This class manages the configuration settings for the MFE Toolbox,
    providing methods to get, set, and reset configuration options.
    It implements a hierarchical configuration system with support for
    environment variables, user-specific configuration files, and
    runtime modifications.
    
    Attributes:
        _config: The current configuration object
        _initialized: Whether the configuration manager has been initialized
        _config_file: Path to the user configuration file
    """
    
    def __init__(self):
        """Initialize the configuration manager with default settings."""
        self._config = MFEConfig()
        self._initialized = False
        self._config_file = None
        self._modified_keys = set()
    
    def initialize(self) -> None:
        """
        Initialize the configuration manager.
        
        This method:
        1. Creates the user configuration directory if it doesn't exist
        2. Loads user configuration from file if available
        3. Applies environment variable overrides
        4. Sets up logging based on configuration
        5. Validates the configuration
        """
        if self._initialized:
            return
        
        # Create user configuration directory if it doesn't exist
        self._ensure_user_config_dir()
        
        # Load user configuration from file if available
        self._load_user_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Set up logging based on configuration
        self._setup_logging()
        
        # Set examples_dir if not already set
        if self._config.paths.examples_dir is None:
            # Try to find the package installation directory
            try:
                import mfe
                package_dir = Path(mfe.__file__).parent
                examples_dir = package_dir / "examples"
                if examples_dir.exists():
                    self._config.paths.examples_dir = examples_dir
            except (ImportError, AttributeError):
                # If we can't find the package, use a default
                self._config.paths.examples_dir = self._config.paths.data_dir / "examples"
        
        # Validate the configuration
        self._validate_config()
        
        self._initialized = True
        logger.debug("Configuration manager initialized")
    
    def _ensure_user_config_dir(self) -> None:
        """
        Ensure the user configuration directory exists.
        
        This method:
        1. Checks for the MFE_CONFIG_DIR environment variable
        2. Creates the user configuration directory if it doesn't exist
        3. Sets the config_file path
        """
        # Check for environment variable override
        env_config_dir = os.environ.get(USER_CONFIG_DIR_ENV)
        if env_config_dir:
            user_config_dir = Path(env_config_dir)
        else:
            user_config_dir = self._config.core.user_config_dir
        
        # Create directory if it doesn't exist
        try:
            user_config_dir.mkdir(parents=True, exist_ok=True)
            self._config.core.user_config_dir = user_config_dir
            self._config_file = user_config_dir / DEFAULT_CONFIG_FILENAME
        except Exception as e:
            logger.warning(f"Failed to create user config directory: {e}")
            # Fall back to temporary directory
            temp_dir = Path(tempfile.gettempdir()) / "mfe_config"
            temp_dir.mkdir(exist_ok=True)
            self._config.core.user_config_dir = temp_dir
            self._config_file = temp_dir / DEFAULT_CONFIG_FILENAME
    
    def _load_user_config(self) -> None:
        """
        Load user configuration from file.
        
        This method:
        1. Checks if the user configuration file exists
        2. Loads and parses the configuration file
        3. Updates the configuration with user settings
        """
        if not self._config_file or not self._config_file.exists():
            logger.debug("No user configuration file found")
            return
        
        try:
            with open(self._config_file, 'r') as f:
                user_config = json.load(f)
            
            # Update configuration with user settings
            self._update_from_dict(user_config)
            logger.debug(f"Loaded user configuration from {self._config_file}")
        except Exception as e:
            logger.warning(f"Failed to load user configuration: {e}")
    
    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to the configuration.
        
        This method:
        1. Checks for environment variables with the MFE_ prefix
        2. Updates the corresponding configuration options
        """
        # Process all environment variables with the MFE_ prefix
        for env_var, value in os.environ.items():
            if not env_var.startswith(CONFIG_ENV_PREFIX):
                continue
            
            # Remove prefix and split into section and option
            key = env_var[len(CONFIG_ENV_PREFIX):]
            parts = key.lower().split('_', 1)
            
            if len(parts) != 2:
                continue
            
            section, option = parts
            
            # Check if this is a valid section
            try:
                section_enum = ConfigSection(section)
            except ValueError:
                continue
            
            # Get the section dataclass
            section_obj = getattr(self._config, section, None)
            if section_obj is None:
                continue
            
            # Check if this is a valid option for the section
            if not hasattr(section_obj, option):
                continue
            
            # Get the current value and its type
            current_value = getattr(section_obj, option)
            value_type = type(current_value)
            
            # Convert the environment variable value to the appropriate type
            try:
                if value_type is bool:
                    # Handle boolean values
                    typed_value = value.lower() in ('true', 'yes', '1', 'y')
                elif value_type is int:
                    typed_value = int(value)
                elif value_type is float:
                    typed_value = float(value)
                elif value_type is Path:
                    typed_value = Path(value)
                elif value_type is list:
                    # Assume comma-separated list
                    typed_value = [item.strip() for item in value.split(',')]
                elif value_type is tuple and len(current_value) == 2:
                    # Handle tuple of two values (e.g., figsize)
                    parts = value.split(',')
                    if len(parts) == 2:
                        typed_value = (int(parts[0]), int(parts[1]))
                    else:
                        continue
                else:
                    # Use the value as is for strings and other types
                    typed_value = value
                
                # Set the option
                setattr(section_obj, option, typed_value)
                logger.debug(f"Applied environment override: {env_var}={value}")
            except Exception as e:
                logger.warning(f"Failed to apply environment override {env_var}: {e}")
    
    def _setup_logging(self) -> None:
        """
        Set up logging based on the configuration.
        
        This method:
        1. Configures the root logger based on the logging configuration
        2. Sets up console and file logging as specified
        """
        # Get the root logger
        root_logger = logging.getLogger("mfe")
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set log level
        log_level = getattr(logging, self._config.logging.log_level)
        root_logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt=self._config.logging.log_format,
            datefmt=self._config.logging.log_date_format
        )
        
        # Add console handler if enabled
        if self._config.logging.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler if enabled and log file is specified
        if self._config.logging.file_logging and self._config.logging.log_file:
            try:
                # Ensure directory exists
                log_dir = self._config.logging.log_file.parent
                log_dir.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(self._config.logging.log_file)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Failed to set up file logging: {e}")
    
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        
        This method:
        1. Checks that all configuration values are of the correct type
        2. Validates specific constraints on configuration values
        3. Logs warnings for any validation issues
        """
        # Validate core configuration
        self._validate_section(self._config.core, "core")
        
        # Validate performance configuration
        self._validate_section(self._config.performance, "performance")
        
        # Validate numerical configuration
        self._validate_section(self._config.numerical, "numerical")
        
        # Validate output configuration
        self._validate_section(self._config.output, "output")
        
        # Validate models configuration
        self._validate_section(self._config.models, "models")
        
        # Validate UI configuration
        self._validate_section(self._config.ui, "ui")
        
        # Validate logging configuration
        self._validate_section(self._config.logging, "logging")
        
        # Validate paths configuration
        self._validate_section(self._config.paths, "paths")
        
        # Validate features configuration
        self._validate_section(self._config.features, "features")
    
    def _validate_section(self, section: Any, section_name: str) -> None:
        """
        Validate a configuration section.
        
        Args:
            section: The configuration section to validate
            section_name: The name of the section
        """
        # Get type hints for the section
        hints = get_type_hints(type(section))
        
        # Check each attribute
        for attr_name, attr_type in hints.items():
            value = getattr(section, attr_name)
            
            # Skip None values for Optional types
            if value is None and "Optional" in str(attr_type):
                continue
            
            # Check type
            try:
                # Handle Path type specially
                if attr_type == Path and isinstance(value, str):
                    setattr(section, attr_name, Path(value))
                # Handle LogLevel enum
                elif attr_name == "log_level" and isinstance(value, str):
                    if value not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
                        logger.warning(f"Invalid log level: {value}, using INFO")
                        setattr(section, attr_name, "INFO")
                # Check other types
                elif not isinstance(value, attr_type):
                    logger.warning(
                        f"Invalid type for {section_name}.{attr_name}: "
                        f"expected {attr_type}, got {type(value)}"
                    )
            except TypeError:
                # Complex types like Union, Optional, etc.
                pass
            
            # Validate specific constraints
            self._validate_constraint(section, attr_name, value, section_name)
    
    def _validate_constraint(self, section: Any, attr_name: str, value: Any, section_name: str) -> None:
        """
        Validate a specific constraint on a configuration value.
        
        Args:
            section: The configuration section
            attr_name: The attribute name
            value: The attribute value
            section_name: The name of the section
        """
        # Validate numeric constraints
        if attr_name == "max_iterations" and value <= 0:
            logger.warning(f"Invalid max_iterations: {value}, must be positive")
            setattr(section, attr_name, 1000)
        
        elif attr_name == "optimization_tol" and (value <= 0 or value >= 1):
            logger.warning(f"Invalid optimization_tol: {value}, must be between 0 and 1")
            setattr(section, attr_name, 1e-8)
        
        elif attr_name == "finite_difference_step" and (value <= 0 or value >= 1):
            logger.warning(f"Invalid finite_difference_step: {value}, must be between 0 and 1")
            setattr(section, attr_name, 1e-8)
        
        elif attr_name == "float_precision" and (value < 0 or value > 16):
            logger.warning(f"Invalid float_precision: {value}, must be between 0 and 16")
            setattr(section, attr_name, 4)
        
        elif attr_name == "cache_size_limit_mb" and value < 0:
            logger.warning(f"Invalid cache_size_limit_mb: {value}, must be non-negative")
            setattr(section, attr_name, 1024)
        
        # Validate enum-like constraints
        elif attr_name == "optimization_method" and value not in (
            "BFGS", "L-BFGS-B", "SLSQP", "Nelder-Mead", "Powell", "CG", "Newton-CG"
        ):
            logger.warning(f"Invalid optimization_method: {value}, using BFGS")
            setattr(section, attr_name, "BFGS")
        
        elif attr_name == "gradient_method" and value not in ("central", "forward", "backward"):
            logger.warning(f"Invalid gradient_method: {value}, using central")
            setattr(section, attr_name, "central")
        
        elif attr_name == "hessian_method" and value not in ("central", "forward", "backward"):
            logger.warning(f"Invalid hessian_method: {value}, using central")
            setattr(section, attr_name, "central")
        
        elif attr_name == "default_float_type" and value not in ("float32", "float64"):
            logger.warning(f"Invalid default_float_type: {value}, using float64")
            setattr(section, attr_name, "float64")
        
        elif attr_name == "theme" and value not in ("light", "dark", "system"):
            logger.warning(f"Invalid theme: {value}, using system")
            setattr(section, attr_name, "system")
        
        elif attr_name == "default_volatility_model" and value not in (
            "GARCH", "EGARCH", "TARCH", "GJR-GARCH", "APARCH", "FIGARCH", "IGARCH", "HEAVY", "AGARCH"
        ):
            logger.warning(f"Invalid default_volatility_model: {value}, using GARCH")
            setattr(section, attr_name, "GARCH")
        
        elif attr_name == "default_multivariate_model" and value not in (
            "BEKK", "DCC", "ADCC", "CCC", "OGARCH", "GOGARCH", "RARCH", "RCC", "MATRIX-GARCH", "RISKMETRICS"
        ):
            logger.warning(f"Invalid default_multivariate_model: {value}, using DCC")
            setattr(section, attr_name, "DCC")
        
        elif attr_name == "default_distribution" and value not in (
            "normal", "t", "skewed_t", "ged"
        ):
            logger.warning(f"Invalid default_distribution: {value}, using normal")
            setattr(section, attr_name, "normal")
        
        elif attr_name == "default_bootstrap_method" and value not in (
            "block", "stationary", "moving-block", "circular-block"
        ):
            logger.warning(f"Invalid default_bootstrap_method: {value}, using stationary")
            setattr(section, attr_name, "stationary")
        
        elif attr_name == "default_realized_estimator" and value not in (
            "RV", "BPV", "MinRV", "MedRV", "RK", "TSRV", "MSRV", "QRV", "QMLE", "PAV", "PBPV", "TMPV", "TV"
        ):
            logger.warning(f"Invalid default_realized_estimator: {value}, using RV")
            setattr(section, attr_name, "RV")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update the configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
        """
        # Update each section
        for section_name, section_dict in config_dict.items():
            if not hasattr(self._config, section_name):
                logger.warning(f"Unknown configuration section: {section_name}")
                continue
            
            section = getattr(self._config, section_name)
            
            # Update each option in the section
            for option_name, option_value in section_dict.items():
                if not hasattr(section, option_name):
                    logger.warning(f"Unknown configuration option: {section_name}.{option_name}")
                    continue
                
                # Convert Path strings to Path objects
                if isinstance(getattr(section, option_name), Path) and isinstance(option_value, str):
                    option_value = Path(option_value)
                
                # Set the option
                try:
                    setattr(section, option_name, option_value)
                except Exception as e:
                    logger.warning(f"Failed to set {section_name}.{option_name}: {e}")
    
    def save_user_config(self) -> None:
        """
        Save the current configuration to the user configuration file.
        
        This method:
        1. Converts the configuration to a dictionary
        2. Writes the dictionary to the user configuration file
        """
        if not self._config_file:
            logger.warning("No user configuration file path available")
            return
        
        try:
            # Convert configuration to dictionary
            config_dict = self.to_dict()
            
            # Ensure the directory exists
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(self._config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=self._json_serialize)
            
            logger.debug(f"Saved user configuration to {self._config_file}")
        except Exception as e:
            logger.warning(f"Failed to save user configuration: {e}")
    
    def _json_serialize(self, obj: Any) -> Any:
        """
        Custom JSON serializer for types that aren't directly serializable.
        
        Args:
            obj: The object to serialize
            
        Returns:
            A JSON-serializable representation of the object
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        result = {}
        
        # Convert each section to a dictionary
        for section_name in dir(self._config):
            # Skip private attributes and methods
            if section_name.startswith('_') or callable(getattr(self._config, section_name)):
                continue
            
            section = getattr(self._config, section_name)
            
            # Convert dataclass to dictionary
            if hasattr(section, "__dataclass_fields__"):
                section_dict = {}
                for field_name in section.__dataclass_fields__:
                    value = getattr(section, field_name)
                    
                    # Convert Path objects to strings
                    if isinstance(value, Path):
                        value = str(value)
                    
                    section_dict[field_name] = value
                
                result[section_name] = section_dict
        
        return result
    
    def get(self, section: str, option: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: The configuration section
            option: The configuration option
            default: Default value if the option is not found
        
        Returns:
            The configuration value, or the default if not found
        """
        if not hasattr(self._config, section):
            return default
        
        section_obj = getattr(self._config, section)
        
        if not hasattr(section_obj, option):
            return default
        
        return getattr(section_obj, option)
    
    def set(self, section: str, option: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: The configuration section
            option: The configuration option
            value: The value to set
        
        Raises:
            ConfigurationError: If the section or option is not found
        """
        if not hasattr(self._config, section):
            raise ConfigurationError(
                f"Unknown configuration section: {section}",
                setting=f"{section}.{option}",
                value=value,
                issue="Section not found"
            )
        
        section_obj = getattr(self._config, section)
        
        if not hasattr(section_obj, option):
            raise ConfigurationError(
                f"Unknown configuration option: {section}.{option}",
                setting=f"{section}.{option}",
                value=value,
                issue="Option not found"
            )
        
        # Get the current value and its type
        current_value = getattr(section_obj, option)
        value_type = type(current_value)
        
        # Convert the value to the appropriate type
        try:
            if value_type is bool and isinstance(value, str):
                # Handle boolean values from strings
                typed_value = value.lower() in ('true', 'yes', '1', 'y')
            elif value_type is Path and isinstance(value, str):
                # Convert string to Path
                typed_value = Path(value)
            elif value_type is not type(value):
                # Try to convert the value
                typed_value = value_type(value)
            else:
                # Use the value as is
                typed_value = value
            
            # Set the option
            setattr(section_obj, option, typed_value)
            
            # Add to modified keys
            self._modified_keys.add(f"{section}.{option}")
            
            logger.debug(f"Set configuration option: {section}.{option}={value}")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to set configuration option: {section}.{option}",
                setting=f"{section}.{option}",
                value=value,
                issue=str(e)
            )
    
    def reset(self, section: Optional[str] = None, option: Optional[str] = None) -> None:
        """
        Reset configuration to default values.
        
        Args:
            section: The configuration section to reset, or None to reset all
            option: The configuration option to reset, or None to reset the entire section
        
        Raises:
            ConfigurationError: If the section or option is not found
        """
        if section is None:
            # Reset all configuration
            self._config = MFEConfig()
            self._modified_keys.clear()
            logger.debug("Reset all configuration to defaults")
            return
        
        if not hasattr(self._config, section):
            raise ConfigurationError(
                f"Unknown configuration section: {section}",
                setting=section,
                issue="Section not found"
            )
        
        if option is None:
            # Reset entire section
            default_config = MFEConfig()
            setattr(self._config, section, getattr(default_config, section))
            
            # Remove modified keys for this section
            self._modified_keys = {k for k in self._modified_keys if not k.startswith(f"{section}.")}
            
            logger.debug(f"Reset configuration section: {section}")
            return
        
        section_obj = getattr(self._config, section)
        
        if not hasattr(section_obj, option):
            raise ConfigurationError(
                f"Unknown configuration option: {section}.{option}",
                setting=f"{section}.{option}",
                issue="Option not found"
            )
        
        # Reset the option to its default value
        default_config = MFEConfig()
        default_section = getattr(default_config, section)
        default_value = getattr(default_section, option)
        
        setattr(section_obj, option, default_value)
        
        # Remove from modified keys
        self._modified_keys.discard(f"{section}.{option}")
        
        logger.debug(f"Reset configuration option: {section}.{option}")
    
    def get_modified_options(self) -> Dict[str, Any]:
        """
        Get a dictionary of modified configuration options.
        
        Returns:
            Dictionary of modified options with their current values
        """
        result = {}
        
        for key in self._modified_keys:
            section, option = key.split(".", 1)
            value = self.get(section, option)
            
            if section not in result:
                result[section] = {}
            
            result[section][option] = value
        
        return result
    
    def is_modified(self, section: str, option: str) -> bool:
        """
        Check if a configuration option has been modified.
        
        Args:
            section: The configuration section
            option: The configuration option
        
        Returns:
            True if the option has been modified, False otherwise
        """
        return f"{section}.{option}" in self._modified_keys
    
    def has_section(self, section: str) -> bool:
        """
        Check if a configuration section exists.
        
        Args:
            section: The configuration section
        
        Returns:
            True if the section exists, False otherwise
        """
        return hasattr(self._config, section)
    
    def has_option(self, section: str, option: str) -> bool:
        """
        Check if a configuration option exists.
        
        Args:
            section: The configuration section
            option: The configuration option
        
        Returns:
            True if the option exists, False otherwise
        """
        if not self.has_section(section):
            return False
        
        section_obj = getattr(self._config, section)
        return hasattr(section_obj, option)
    
    def get_sections(self) -> List[str]:
        """
        Get a list of all configuration sections.
        
        Returns:
            List of configuration section names
        """
        return [
            attr for attr in dir(self._config)
            if not attr.startswith('_') and not callable(getattr(self._config, attr))
        ]
    
    def get_options(self, section: str) -> List[str]:
        """
        Get a list of all options in a configuration section.
        
        Args:
            section: The configuration section
        
        Returns:
            List of configuration option names
        
        Raises:
            ConfigurationError: If the section is not found
        """
        if not self.has_section(section):
            raise ConfigurationError(
                f"Unknown configuration section: {section}",
                setting=section,
                issue="Section not found"
            )
        
        section_obj = getattr(self._config, section)
        
        return [
            attr for attr in dir(section_obj)
            if not attr.startswith('_') and not callable(getattr(section_obj, attr))
        ]
    
    def get_config_file(self) -> Optional[Path]:
        """
        Get the path to the user configuration file.
        
        Returns:
            Path to the user configuration file, or None if not available
        """
        return self._config_file
    
    def get_user_config_dir(self) -> Path:
        """
        Get the user configuration directory.
        
        Returns:
            Path to the user configuration directory
        """
        return self._config.core.user_config_dir
    
    def get_section(self, section: str) -> Any:
        """
        Get a configuration section object.
        
        Args:
            section: The configuration section
        
        Returns:
            The configuration section object
        
        Raises:
            ConfigurationError: If the section is not found
        """
        if not self.has_section(section):
            raise ConfigurationError(
                f"Unknown configuration section: {section}",
                setting=section,
                issue="Section not found"
            )
        
        return getattr(self._config, section)
    
    def get_full_config(self) -> MFEConfig:
        """
        Get the full configuration object.
        
        Returns:
            The complete configuration object
        """
        return self._config


# Create a singleton instance of the configuration manager
_config_manager = ConfigManager()


def initialize_config() -> None:
    """
    Initialize the configuration system.
    
    This function initializes the configuration manager, loading user
    configuration and applying environment variable overrides.
    """
    _config_manager.initialize()


def get_config(section: str, option: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        section: The configuration section
        option: The configuration option
        default: Default value if the option is not found
    
    Returns:
        The configuration value, or the default if not found
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get(section, option, default)


def set_config(section: str, option: str, value: Any) -> None:
    """
    Set a configuration value.
    
    Args:
        section: The configuration section
        option: The configuration option
        value: The value to set
    
    Raises:
        ConfigurationError: If the section or option is not found
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    _config_manager.set(section, option, value)


def reset_config(section: Optional[str] = None, option: Optional[str] = None) -> None:
    """
    Reset configuration to default values.
    
    Args:
        section: The configuration section to reset, or None to reset all
        option: The configuration option to reset, or None to reset the entire section
    
    Raises:
        ConfigurationError: If the section or option is not found
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    _config_manager.reset(section, option)


def save_config() -> None:
    """
    Save the current configuration to the user configuration file.
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    _config_manager.save_user_config()


def get_config_manager() -> ConfigManager:
    """
    Get the configuration manager instance.
    
    Returns:
        The configuration manager instance
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager


def get_core_config() -> CoreConfig:
    """
    Get the core configuration.
    
    Returns:
        The core configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("core")


def get_performance_config() -> PerformanceConfig:
    """
    Get the performance configuration.
    
    Returns:
        The performance configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("performance")


def get_numerical_config() -> NumericalConfig:
    """
    Get the numerical configuration.
    
    Returns:
        The numerical configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("numerical")


def get_output_config() -> OutputConfig:
    """
    Get the output configuration.
    
    Returns:
        The output configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("output")


def get_models_config() -> ModelsConfig:
    """
    Get the models configuration.
    
    Returns:
        The models configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("models")


def get_ui_config() -> UIConfig:
    """
    Get the UI configuration.
    
    Returns:
        The UI configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("ui")


def get_logging_config() -> LoggingConfig:
    """
    Get the logging configuration.
    
    Returns:
        The logging configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("logging")


def get_paths_config() -> PathsConfig:
    """
    Get the paths configuration.
    
    Returns:
        The paths configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("paths")


def get_features_config() -> FeaturesConfig:
    """
    Get the features configuration.
    
    Returns:
        The features configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_section("features")


def get_full_config() -> MFEConfig:
    """
    Get the full configuration.
    
    Returns:
        The complete configuration object
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_full_config()


def get_config_file() -> Optional[Path]:
    """
    Get the path to the user configuration file.
    
    Returns:
        Path to the user configuration file, or None if not available
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_config_file()


def get_user_config_dir() -> Path:
    """
    Get the user configuration directory.
    
    Returns:
        Path to the user configuration directory
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_user_config_dir()


def get_modified_options() -> Dict[str, Any]:
    """
    Get a dictionary of modified configuration options.
    
    Returns:
        Dictionary of modified options with their current values
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_modified_options()


def is_modified(section: str, option: str) -> bool:
    """
    Check if a configuration option has been modified.
    
    Args:
        section: The configuration section
        option: The configuration option
    
    Returns:
        True if the option has been modified, False otherwise
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.is_modified(section, option)


def has_section(section: str) -> bool:
    """
    Check if a configuration section exists.
    
    Args:
        section: The configuration section
    
    Returns:
        True if the section exists, False otherwise
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.has_section(section)


def has_option(section: str, option: str) -> bool:
    """
    Check if a configuration option exists.
    
    Args:
        section: The configuration section
        option: The configuration option
    
    Returns:
        True if the option exists, False otherwise
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.has_option(section, option)


def get_sections() -> List[str]:
    """
    Get a list of all configuration sections.
    
    Returns:
        List of configuration section names
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_sections()


def get_options(section: str) -> List[str]:
    """
    Get a list of all options in a configuration section.
    
    Args:
        section: The configuration section
    
    Returns:
        List of configuration option names
    
    Raises:
        ConfigurationError: If the section is not found
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.get_options(section)


def to_dict() -> Dict[str, Any]:
    """
    Convert the configuration to a dictionary.
    
    Returns:
        Dictionary representation of the configuration
    """
    # Ensure the configuration manager is initialized
    if not _config_manager._initialized:
        initialize_config()
    
    return _config_manager.to_dict()


# Initialize the configuration when the module is imported
initialize_config()
