# docs/user_guide/gui_interface.rst
====================
ARMAX GUI Interface
====================

Introduction
===========

The MFE Toolbox includes a graphical user interface (GUI) for ARMAX (AutoRegressive Moving Average with eXogenous inputs) modeling. This interface provides an intuitive way to build, estimate, and diagnose time series models without writing code. The GUI is built using PyQt6, a modern Python binding for the Qt application framework, providing a responsive and feature-rich user experience.

This guide covers:

- Launching the ARMAX GUI
- Building and estimating models
- Interpreting results
- Diagnostic analysis
- Exporting results

Getting Started
==============

Launching the ARMAX GUI
----------------------

The ARMAX GUI can be launched directly from Python:

.. code-block:: python

    from mfe.ui.armax_app import launch_armax_gui
    
    # Launch the GUI
    launch_armax_gui()

This will open the main ARMAX interface window, which provides all the tools needed for time series modeling.

Alternatively, you can launch the GUI with your own data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.ui.armax_app import launch_armax_gui
    
    # Create or load your time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    data = pd.Series(np.random.normal(0, 1, 500), index=dates)
    
    # Add some seasonality
    data += np.sin(np.arange(len(data)) * 2 * np.pi / 20)  # 20-day cycle
    
    # Launch the GUI with your data
    launch_armax_gui(data)

Main Interface Overview
---------------------

.. image:: ../_static/images/armax_main_ui.png
   :width: 800
   :alt: ARMAX GUI Main Interface

The main interface consists of several key components:

1. **Model Specification Area**: Controls for setting AR and MA orders, including/excluding constant terms, and defining exogenous variables
2. **Action Buttons**: Buttons for estimating models, resetting parameters, and accessing help
3. **Visualization Area**: Displays the original time series data and model residuals
4. **Tabbed Interface**: Provides access to model results, diagnostic statistics, and additional plots
5. **Model Statistics Panel**: Displays model fit metrics such as AIC, BIC, and log-likelihood values

Building and Estimating Models
============================

Setting Model Parameters
----------------------

To configure your ARMAX model:

1. **AR Order**: Set the autoregressive order using the spin box or directly enter a value
   - Click the [+] button to increment the order
   - Click the [-] button to decrement the order
   - Alternatively, type the desired order directly in the text field

2. **MA Order**: Set the moving average order using the spin box or directly enter a value
   - Controls work the same as for AR order

3. **Include Constant**: Toggle the checkbox to include or exclude a constant term in the model

4. **Exogenous Variables**: If your data includes exogenous variables, select them from the dropdown list
   - Multiple variables can be selected by holding Ctrl (or Cmd on macOS) while clicking

Estimating the Model
-----------------

Once you've configured your model parameters:

1. Click the **Estimate Model** button to start the estimation process
2. A progress indicator will appear during estimation, showing the current status
3. The interface remains responsive during estimation thanks to asynchronous processing
4. When estimation completes, the results will automatically update in the interface

The asynchronous processing is a key feature of the PyQt6-based implementation, allowing the interface to remain responsive even during computationally intensive operations. This is implemented using Python's async/await pattern integrated with PyQt6's event loop.

Interpreting Results
==================

Model Equation Display
-------------------

After estimation, the model equation is displayed in mathematical notation:

.. image:: ../_static/images/armax_equation_render.png
   :width: 600
   :alt: ARMAX Model Equation

The equation shows the estimated model in standard notation, with coefficients rounded to a reasonable precision. This is rendered using matplotlib's LaTeX capabilities embedded within the PyQt6 interface.

Parameter Table
------------

The parameter table provides detailed information about the estimated coefficients:

.. image:: ../_static/images/parameter_table_example.png
   :width: 700
   :alt: Parameter Table

The table includes:

- Parameter names (Constant, AR terms, MA terms, Exogenous variables)
- Estimated coefficient values
- Standard errors
- t-statistics
- p-values with significance indicators

You can sort the table by clicking on column headers, and copy values to the clipboard by selecting cells and using Ctrl+C (or Cmd+C on macOS).

Model Statistics
-------------

The Model Statistics panel displays goodness-of-fit measures:

- **Log-likelihood**: Higher values indicate better fit
- **AIC** (Akaike Information Criterion): Lower values indicate better fit
- **BIC** (Bayesian Information Criterion): Lower values indicate better fit
- **RMSE** (Root Mean Square Error): Lower values indicate better fit

These statistics help you compare different model specifications to find the best fit for your data.

Diagnostic Analysis
================

Residual Plots
------------

The Residuals tab displays plots of the model residuals:

.. image:: ../_static/images/model_diagnostic_plots.png
   :width: 700
   :alt: Residual Diagnostic Plots

These plots help assess model adequacy:

1. **Residual Time Series**: Should show no obvious patterns
2. **Residual Histogram**: Should approximate a normal distribution
3. **Q-Q Plot**: Points should follow the diagonal line for normally distributed residuals
4. **Residual Autocorrelation**: Should show no significant autocorrelation beyond lag 0

ACF and PACF Plots
---------------

The Correlogram tab displays Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots:

.. image:: ../_static/images/acf_pacf_plots.png
   :width: 700
   :alt: ACF and PACF Plots

These plots help assess:

1. **Residual Independence**: Significant spikes in the ACF or PACF of residuals may indicate model inadequacy
2. **Model Order Selection**: For the original series, patterns in these plots help identify appropriate AR and MA orders

Statistical Tests
--------------

The Diagnostics tab provides formal statistical tests:

1. **Ljung-Box Test**: Tests for autocorrelation in residuals
2. **Jarque-Bera Test**: Tests for normality of residuals
3. **ARCH LM Test**: Tests for heteroskedasticity in residuals

Each test includes:
- Test statistic value
- p-value
- Critical values (where applicable)
- Pass/fail indication

Forecasting
=========

Generating Forecasts
-----------------

To generate forecasts:

1. Navigate to the Forecasts tab
2. Set the forecast horizon (number of periods to forecast)
3. Click the **Generate Forecast** button
4. The forecast will be computed asynchronously, with a progress indicator
5. When complete, the forecast plot will display:
   - Point forecasts
   - Confidence intervals (default 95%)
   - Historical data for context

Forecast Options
-------------

You can customize forecasts with these options:

- **Confidence Level**: Set the confidence level for prediction intervals (e.g., 90%, 95%, 99%)
- **Simulation Paths**: Set the number of Monte Carlo simulations for generating prediction intervals
- **Include Exogenous Forecasts**: If your model includes exogenous variables, you can provide forecasts for these variables

Exporting Results
==============

Saving Model Results
-----------------

To save your model results:

1. Click the **Export Results** button
2. Choose from the available export formats:
   - **CSV**: Exports parameter estimates and diagnostics as CSV files
   - **Excel**: Exports all results to an Excel workbook with multiple sheets
   - **JSON**: Exports results in JSON format for programmatic use
   - **HTML**: Exports a formatted HTML report with embedded visualizations

Copying Visualizations
-------------------

To copy visualizations for use in other applications:

1. Right-click on any plot
2. Select **Copy Image** from the context menu
3. Paste the image into your document or presentation

Alternatively, you can save plots directly:

1. Right-click on any plot
2. Select **Save Image As...**
3. Choose your preferred image format (PNG, JPG, SVG, PDF)
4. Select the save location and filename

Advanced Features
==============

Model Viewer Dialog
----------------

For a more detailed view of model results, you can open the Model Viewer dialog:

.. image:: ../_static/images/armax_viewer_ui.png
   :width: 700
   :alt: ARMAX Model Viewer

To access this dialog:

1. Click the **View Detailed Results** button after estimating a model
2. The dialog provides a comprehensive view of all model information
3. For models with many parameters, pagination controls allow navigation through multiple pages

About Dialog
---------

The About dialog provides information about the ARMAX GUI:

.. image:: ../_static/images/armax_about_dialog.png
   :width: 400
   :alt: About Dialog

To access this dialog:

1. Click the **About** button in the main interface
2. The dialog displays version information and credits
3. Click **Close** to dismiss the dialog

Close Confirmation Dialog
---------------------

When closing the application, a confirmation dialog appears:

.. image:: ../_static/images/armax_close_dialog.png
   :width: 400
   :alt: Close Confirmation Dialog

This dialog helps prevent accidental data loss by confirming your intention to close the application.

Technical Implementation
=====================

The ARMAX GUI is built using a modern Model-View-Controller (MVC) architecture with PyQt6:

.. image:: ../_static/images/pyqt6_mvc_architecture.png
   :width: 700
   :alt: PyQt6 MVC Architecture

Key implementation features include:

1. **Signal-Slot Architecture**: UI interactions emit signals that connect to controller slots, creating a reactive programming pattern
2. **Asynchronous Processing**: Long-running computations execute asynchronously using Python's async/await pattern, maintaining UI responsiveness
3. **Embedded Visualization**: LaTeX rendering for mathematical equations uses matplotlib's LaTeX interpreter embedded in PyQt6 widgets
4. **Type-Safe Implementation**: Python type hints throughout the codebase improve reliability and aid development

The implementation follows these design principles:

- **Separation of Concerns**: Clean separation between UI code (views), application logic (controllers), and mathematical models
- **Reactive UI Design**: Real-time updates during model estimation provide progress feedback without UI freezing
- **Modular Component Design**: UI components are organized into reusable modules with clear separation from business logic

Troubleshooting
=============

Common Issues
-----------

**Issue**: GUI fails to launch
**Solution**: Ensure PyQt6 is properly installed: `pip install PyQt6`

**Issue**: Slow performance during estimation
**Solution**: For large datasets, consider reducing the data size or using a more powerful computer

**Issue**: Error messages about invalid model parameters
**Solution**: Ensure your AR and MA orders are appropriate for your data size (typically orders should be much smaller than the data length)

**Issue**: Plots not displaying correctly
**Solution**: Ensure matplotlib is properly installed and update to the latest version

Getting Help
----------

If you encounter issues not covered in this guide:

1. Check the API documentation for detailed information about the underlying functions
2. Look for error messages in the Python console, which may provide more details
3. Visit the project repository for known issues and solutions
4. Contact the maintainers for support with persistent problems

Conclusion
=========

The ARMAX GUI provides an intuitive interface for time series modeling without requiring extensive coding. By leveraging PyQt6's modern UI capabilities and Python's asynchronous processing, it delivers a responsive and feature-rich experience for building, estimating, and diagnosing ARMAX models.

For more advanced time series modeling that goes beyond the GUI's capabilities, refer to the :doc:`time_series_analysis` guide, which covers the programmatic API in detail.
