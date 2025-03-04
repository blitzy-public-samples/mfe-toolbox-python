# mfe/ui/views/styles.py
"""
Styling Constants and Theme Configurations for MFE Toolbox UI

This module defines styling constants, theme configurations, and helper functions
for applying consistent visual styling across the MFE Toolbox's PyQt6 interface.
It provides color schemes, font definitions, layout constants, and stylesheet
templates to ensure a professional and cohesive appearance throughout the application.

The module supports both light and dark themes with accessibility considerations,
and includes helper functions for applying styles to widget hierarchies and
dynamically switching themes at runtime.
"""

import logging
import sys
from typing import Dict, Any, Optional, Union, List, Tuple, cast

from PyQt6.QtCore import Qt, QSize, QMargins
from PyQt6.QtGui import QColor, QFont, QPalette, QFontMetrics
from PyQt6.QtWidgets import (
    QWidget, QApplication, QPushButton, QLabel, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QGroupBox,
    QTabWidget, QTableWidget, QHeaderView, QScrollBar, QFrame,
    QMainWindow, QDialog, QMessageBox, QToolTip, QMenu, QToolBar,
    QStatusBar, QProgressBar, QSlider, QSplitter, QTabBar
)

# Set up module-level logger
logger = logging.getLogger("mfe.ui.views.styles")

# =============================================================================
# COLOR SCHEMES
# =============================================================================

# Light theme colors
LIGHT_THEME = {
    # Primary colors
    "primary": "#4a86e8",           # Primary blue
    "primary_light": "#7ba9f0",     # Lighter blue for hover states
    "primary_dark": "#2a66c8",      # Darker blue for pressed states
    
    # Secondary colors
    "secondary": "#6aa84f",         # Secondary green
    "secondary_light": "#8cc774",   # Lighter green for hover states
    "secondary_dark": "#4a882f",    # Darker green for pressed states
    
    # Accent colors
    "accent": "#e69138",            # Accent orange
    "accent_light": "#f0b06c",      # Lighter orange for hover states
    "accent_dark": "#c67118",       # Darker orange for pressed states
    
    # Neutral colors
    "background": "#ffffff",        # Background white
    "surface": "#f5f5f5",           # Surface light gray
    "text": "#333333",              # Text dark gray
    "text_secondary": "#666666",    # Secondary text medium gray
    "text_disabled": "#999999",     # Disabled text light gray
    "border": "#dddddd",            # Border light gray
    "divider": "#eeeeee",           # Divider very light gray
    
    # UI element colors
    "button": "#f0f0f0",            # Button background
    "button_text": "#333333",       # Button text
    "button_hover": "#e0e0e0",      # Button hover
    "button_pressed": "#d0d0d0",    # Button pressed
    "input_background": "#ffffff",  # Input background
    "input_border": "#cccccc",      # Input border
    "input_focus": "#4a86e8",       # Input focus border
    
    # Feedback colors
    "success": "#6aa84f",           # Success green
    "warning": "#e69138",           # Warning orange
    "error": "#cc0000",             # Error red
    "info": "#4a86e8",              # Info blue
    
    # Table colors
    "table_header": "#f0f0f0",      # Table header background
    "table_row_odd": "#ffffff",     # Table odd row
    "table_row_even": "#f9f9f9",    # Table even row
    "table_border": "#dddddd",      # Table border
    "table_selection": "#d0e0f7",   # Table selection
    
    # Tab colors
    "tab_background": "#f0f0f0",    # Tab background
    "tab_selected": "#ffffff",      # Selected tab
    "tab_hover": "#e0e0e0",         # Tab hover
    
    # Scrollbar colors
    "scrollbar_background": "#f0f0f0",  # Scrollbar background
    "scrollbar_handle": "#cccccc",      # Scrollbar handle
    "scrollbar_hover": "#bbbbbb",       # Scrollbar hover
    
    # Tooltip colors
    "tooltip_background": "#f5f5f5",    # Tooltip background
    "tooltip_text": "#333333",          # Tooltip text
    "tooltip_border": "#dddddd",        # Tooltip border
    
    # Progress bar colors
    "progress_background": "#f0f0f0",   # Progress bar background
    "progress_fill": "#4a86e8",         # Progress bar fill
    
    # Dialog colors
    "dialog_background": "#ffffff",     # Dialog background
    "dialog_border": "#dddddd",         # Dialog border
    
    # Additional UI colors
    "highlight": "#4a86e8",             # Highlight color
    "highlighted_text": "#ffffff",      # Highlighted text
    "link": "#0066cc",                  # Link color
    "visited_link": "#551a8b",          # Visited link
    "active_link": "#cc0000",           # Active link
    "base": "#ffffff",                  # Base color
    "alternate_base": "#f9f9f9",        # Alternate base color
    "window": "#f5f5f5",                # Window color
    "window_text": "#333333",           # Window text
    "button_text": "#333333",           # Button text
    "bright_text": "#ffffff",           # Bright text
    "light": "#f0f0f0",                 # Light color
    "midlight": "#e0e0e0",              # Midlight color
    "dark": "#a0a0a0",                  # Dark color
    "mid": "#b0b0b0",                   # Mid color
    "shadow": "#707070"                 # Shadow color
}

# Dark theme colors
DARK_THEME = {
    # Primary colors
    "primary": "#5c94f0",           # Primary blue
    "primary_light": "#80b0ff",     # Lighter blue for hover states
    "primary_dark": "#3a74d0",      # Darker blue for pressed states
    
    # Secondary colors
    "secondary": "#7ab85f",         # Secondary green
    "secondary_light": "#9cd884",   # Lighter green for hover states
    "secondary_dark": "#5a983f",    # Darker green for pressed states
    
    # Accent colors
    "accent": "#f0a148",            # Accent orange
    "accent_light": "#ffc078",      # Lighter orange for hover states
    "accent_dark": "#d08128",       # Darker orange for pressed states
    
    # Neutral colors
    "background": "#2d2d2d",        # Background dark gray
    "surface": "#3d3d3d",           # Surface medium gray
    "text": "#e0e0e0",              # Text light gray
    "text_secondary": "#b0b0b0",    # Secondary text medium gray
    "text_disabled": "#808080",     # Disabled text dark gray
    "border": "#555555",            # Border medium gray
    "divider": "#444444",           # Divider dark gray
    
    # UI element colors
    "button": "#3d3d3d",            # Button background
    "button_text": "#e0e0e0",       # Button text
    "button_hover": "#4d4d4d",      # Button hover
    "button_pressed": "#5d5d5d",    # Button pressed
    "input_background": "#2d2d2d",  # Input background
    "input_border": "#555555",      # Input border
    "input_focus": "#5c94f0",       # Input focus border
    
    # Feedback colors
    "success": "#7ab85f",           # Success green
    "warning": "#f0a148",           # Warning orange
    "error": "#e05050",             # Error red
    "info": "#5c94f0",              # Info blue
    
    # Table colors
    "table_header": "#3d3d3d",      # Table header background
    "table_row_odd": "#2d2d2d",     # Table odd row
    "table_row_even": "#353535",    # Table even row
    "table_border": "#555555",      # Table border
    "table_selection": "#4a6484",   # Table selection
    
    # Tab colors
    "tab_background": "#3d3d3d",    # Tab background
    "tab_selected": "#2d2d2d",      # Selected tab
    "tab_hover": "#4d4d4d",         # Tab hover
    
    # Scrollbar colors
    "scrollbar_background": "#3d3d3d",  # Scrollbar background
    "scrollbar_handle": "#555555",      # Scrollbar handle
    "scrollbar_hover": "#666666",       # Scrollbar hover
    
    # Tooltip colors
    "tooltip_background": "#3d3d3d",    # Tooltip background
    "tooltip_text": "#e0e0e0",          # Tooltip text
    "tooltip_border": "#555555",        # Tooltip border
    
    # Progress bar colors
    "progress_background": "#3d3d3d",   # Progress bar background
    "progress_fill": "#5c94f0",         # Progress bar fill
    
    # Dialog colors
    "dialog_background": "#2d2d2d",     # Dialog background
    "dialog_border": "#555555",         # Dialog border
    
    # Additional UI colors
    "highlight": "#5c94f0",             # Highlight color
    "highlighted_text": "#ffffff",      # Highlighted text
    "link": "#80b0ff",                  # Link color
    "visited_link": "#c080ff",          # Visited link
    "active_link": "#ff8080",           # Active link
    "base": "#2d2d2d",                  # Base color
    "alternate_base": "#353535",        # Alternate base color
    "window": "#2d2d2d",                # Window color
    "window_text": "#e0e0e0",           # Window text
    "button_text": "#e0e0e0",           # Button text
    "bright_text": "#ffffff",           # Bright text
    "light": "#3d3d3d",                 # Light color
    "midlight": "#4d4d4d",              # Midlight color
    "dark": "#1d1d1d",                  # Dark color
    "mid": "#2d2d2d",                   # Mid color
    "shadow": "#1a1a1a"                 # Shadow color
}

# High contrast theme for accessibility
HIGH_CONTRAST_THEME = {
    # Primary colors
    "primary": "#0066cc",           # Primary blue
    "primary_light": "#0088ff",     # Lighter blue for hover states
    "primary_dark": "#004499",      # Darker blue for pressed states
    
    # Secondary colors
    "secondary": "#008800",         # Secondary green
    "secondary_light": "#00aa00",   # Lighter green for hover states
    "secondary_dark": "#006600",    # Darker green for pressed states
    
    # Accent colors
    "accent": "#cc6600",            # Accent orange
    "accent_light": "#ff8800",      # Lighter orange for hover states
    "accent_dark": "#994400",       # Darker orange for pressed states
    
    # Neutral colors
    "background": "#ffffff",        # Background white
    "surface": "#f0f0f0",           # Surface light gray
    "text": "#000000",              # Text black
    "text_secondary": "#333333",    # Secondary text dark gray
    "text_disabled": "#666666",     # Disabled text medium gray
    "border": "#000000",            # Border black
    "divider": "#666666",           # Divider medium gray
    
    # UI element colors
    "button": "#e0e0e0",            # Button background
    "button_text": "#000000",       # Button text
    "button_hover": "#cccccc",      # Button hover
    "button_pressed": "#b0b0b0",    # Button pressed
    "input_background": "#ffffff",  # Input background
    "input_border": "#000000",      # Input border
    "input_focus": "#0066cc",       # Input focus border
    
    # Feedback colors
    "success": "#008800",           # Success green
    "warning": "#cc6600",           # Warning orange
    "error": "#cc0000",             # Error red
    "info": "#0066cc",              # Info blue
    
    # Table colors
    "table_header": "#e0e0e0",      # Table header background
    "table_row_odd": "#ffffff",     # Table odd row
    "table_row_even": "#f0f0f0",    # Table even row
    "table_border": "#000000",      # Table border
    "table_selection": "#0066cc",   # Table selection
    
    # Tab colors
    "tab_background": "#e0e0e0",    # Tab background
    "tab_selected": "#ffffff",      # Selected tab
    "tab_hover": "#cccccc",         # Tab hover
    
    # Scrollbar colors
    "scrollbar_background": "#e0e0e0",  # Scrollbar background
    "scrollbar_handle": "#666666",      # Scrollbar handle
    "scrollbar_hover": "#333333",       # Scrollbar hover
    
    # Tooltip colors
    "tooltip_background": "#ffffcc",    # Tooltip background
    "tooltip_text": "#000000",          # Tooltip text
    "tooltip_border": "#000000",        # Tooltip border
    
    # Progress bar colors
    "progress_background": "#e0e0e0",   # Progress bar background
    "progress_fill": "#0066cc",         # Progress bar fill
    
    # Dialog colors
    "dialog_background": "#ffffff",     # Dialog background
    "dialog_border": "#000000",         # Dialog border
    
    # Additional UI colors
    "highlight": "#0066cc",             # Highlight color
    "highlighted_text": "#ffffff",      # Highlighted text
    "link": "#0000ee",                  # Link color
    "visited_link": "#551a8b",          # Visited link
    "active_link": "#cc0000",           # Active link
    "base": "#ffffff",                  # Base color
    "alternate_base": "#f0f0f0",        # Alternate base color
    "window": "#f0f0f0",                # Window color
    "window_text": "#000000",           # Window text
    "button_text": "#000000",           # Button text
    "bright_text": "#ffffff",           # Bright text
    "light": "#e0e0e0",                 # Light color
    "midlight": "#cccccc",              # Midlight color
    "dark": "#666666",                  # Dark color
    "mid": "#999999",                   # Mid color
    "shadow": "#333333"                 # Shadow color
}

# Dictionary of available themes
THEMES = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
    "high_contrast": HIGH_CONTRAST_THEME
}

# Current theme (default to light)
CURRENT_THEME = "light"

# =============================================================================
# FONT DEFINITIONS
# =============================================================================

# Font families
FONT_FAMILY_SANS = "Segoe UI, Helvetica, Arial, sans-serif"
FONT_FAMILY_MONO = "Consolas, Courier New, monospace"
FONT_FAMILY_SERIF = "Times New Roman, serif"

# Font sizes
FONT_SIZE_TINY = 8
FONT_SIZE_SMALL = 10
FONT_SIZE_NORMAL = 11
FONT_SIZE_MEDIUM = 12
FONT_SIZE_LARGE = 14
FONT_SIZE_XLARGE = 16
FONT_SIZE_XXLARGE = 20

# Font weights
FONT_WEIGHT_NORMAL = QFont.Weight.Normal
FONT_WEIGHT_BOLD = QFont.Weight.Bold

# Font styles
FONT_STYLE_NORMAL = QFont.Style.StyleNormal
FONT_STYLE_ITALIC = QFont.Style.StyleItalic

# Default fonts
DEFAULT_FONT = QFont(FONT_FAMILY_SANS.split(',')[0].strip(), FONT_SIZE_NORMAL)
MONOSPACE_FONT = QFont(FONT_FAMILY_MONO.split(',')[0].strip(), FONT_SIZE_NORMAL)
HEADING_FONT = QFont(FONT_FAMILY_SANS.split(',')[0].strip(), FONT_SIZE_LARGE, FONT_WEIGHT_BOLD)
TITLE_FONT = QFont(FONT_FAMILY_SANS.split(',')[0].strip(), FONT_SIZE_XLARGE, FONT_WEIGHT_BOLD)

# =============================================================================
# LAYOUT CONSTANTS
# =============================================================================

# Spacing
SPACING_TINY = 2
SPACING_SMALL = 4
SPACING_NORMAL = 8
SPACING_MEDIUM = 12
SPACING_LARGE = 16
SPACING_XLARGE = 24

# Margins
MARGIN_NONE = QMargins(0, 0, 0, 0)
MARGIN_TINY = QMargins(SPACING_TINY, SPACING_TINY, SPACING_TINY, SPACING_TINY)
MARGIN_SMALL = QMargins(SPACING_SMALL, SPACING_SMALL, SPACING_SMALL, SPACING_SMALL)
MARGIN_NORMAL = QMargins(SPACING_NORMAL, SPACING_NORMAL, SPACING_NORMAL, SPACING_NORMAL)
MARGIN_MEDIUM = QMargins(SPACING_MEDIUM, SPACING_MEDIUM, SPACING_MEDIUM, SPACING_MEDIUM)
MARGIN_LARGE = QMargins(SPACING_LARGE, SPACING_LARGE, SPACING_LARGE, SPACING_LARGE)

# Icon sizes
ICON_SIZE_SMALL = QSize(16, 16)
ICON_SIZE_NORMAL = QSize(24, 24)
ICON_SIZE_MEDIUM = QSize(32, 32)
ICON_SIZE_LARGE = QSize(48, 48)

# Button sizes
BUTTON_SIZE_SMALL = QSize(80, 24)
BUTTON_SIZE_NORMAL = QSize(100, 30)
BUTTON_SIZE_MEDIUM = QSize(120, 36)
BUTTON_SIZE_LARGE = QSize(150, 42)

# Input field heights
INPUT_HEIGHT_SMALL = 24
INPUT_HEIGHT_NORMAL = 30
INPUT_HEIGHT_MEDIUM = 36
INPUT_HEIGHT_LARGE = 42

# Border radius
BORDER_RADIUS_NONE = 0
BORDER_RADIUS_SMALL = 2
BORDER_RADIUS_NORMAL = 4
BORDER_RADIUS_MEDIUM = 6
BORDER_RADIUS_LARGE = 8
BORDER_RADIUS_ROUND = 15

# Border widths
BORDER_WIDTH_NONE = 0
BORDER_WIDTH_THIN = 1
BORDER_WIDTH_NORMAL = 2
BORDER_WIDTH_THICK = 3

# =============================================================================
# STYLESHEET TEMPLATES
# =============================================================================

# Base stylesheet template
BASE_STYLESHEET = """
/* Global styles */
QWidget {
    font-family: "{font_family}";
    font-size: {font_size}pt;
    color: {text};
    background-color: {background};
}

/* Main window */
QMainWindow {
    background-color: {background};
}

/* Dialog */
QDialog {
    background-color: {dialog_background};
    border: {border_width_normal}px solid {dialog_border};
    border-radius: {border_radius_normal}px;
}

/* Group box */
QGroupBox {
    font-weight: bold;
    border: {border_width_thin}px solid {border};
    border-radius: {border_radius_normal}px;
    margin-top: 1.5ex;
    padding-top: 1.5ex;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
}

/* Push button */
QPushButton {{
    background-color: {button};
    color: {button_text};
    border: {border_width_thin}px solid {border};
    border-radius: {border_radius_normal}px;
    padding: 5px 10px;
    min-height: {input_height_normal}px;
}}

QPushButton:hover {{
    background-color: {button_hover};
}}

QPushButton:pressed {{
    background-color: {button_pressed};
}}

QPushButton:disabled {{
    background-color: {button};
    color: {text_disabled};
    border: {border_width_thin}px solid {border};
}}

QPushButton:default {{
    border: {border_width_normal}px solid {primary};
}}

/* Line edit */
QLineEdit {{
    background-color: {input_background};
    color: {text};
    border: {border_width_thin}px solid {input_border};
    border-radius: {border_radius_normal}px;
    padding: 2px 5px;
    min-height: {input_height_normal}px;
}}

QLineEdit:focus {{
    border: {border_width_thin}px solid {input_focus};
}}

QLineEdit:disabled {{
    background-color: {input_background};
    color: {text_disabled};
    border: {border_width_thin}px solid {border};
}}

/* Combo box */
QComboBox {{
    background-color: {input_background};
    color: {text};
    border: {border_width_thin}px solid {input_border};
    border-radius: {border_radius_normal}px;
    padding: 2px 5px;
    min-height: {input_height_normal}px;
}}

QComboBox:focus {{
    border: {border_width_thin}px solid {input_focus};
}}

QComboBox:disabled {{
    background-color: {input_background};
    color: {text_disabled};
    border: {border_width_thin}px solid {border};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: {border_width_thin}px solid {border};
}}

QComboBox QAbstractItemView {{
    background-color: {input_background};
    color: {text};
    border: {border_width_thin}px solid {input_border};
    selection-background-color: {highlight};
    selection-color: {highlighted_text};
}}

/* Spin box */
QSpinBox, QDoubleSpinBox {{
    background-color: {input_background};
    color: {text};
    border: {border_width_thin}px solid {input_border};
    border-radius: {border_radius_normal}px;
    padding: 2px 5px;
    min-height: {input_height_normal}px;
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border: {border_width_thin}px solid {input_focus};
}}

QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {input_background};
    color: {text_disabled};
    border: {border_width_thin}px solid {border};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border-left: {border_width_thin}px solid {border};
    border-bottom: {border_width_thin}px solid {border};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border-left: {border_width_thin}px solid {border};
    border-top: {border_width_thin}px solid {border};
}}

/* Check box */
QCheckBox {{
    spacing: 5px;
}}

QCheckBox:disabled {{
    color: {text_disabled};
}}

/* Radio button */
QRadioButton {{
    spacing: 5px;
}}

QRadioButton:disabled {{
    color: {text_disabled};
}}

/* Label */
QLabel {{
    color: {text};
}}

QLabel:disabled {{
    color: {text_disabled};
}}

/* Tab widget */
QTabWidget::pane {{
    border: {border_width_thin}px solid {border};
    border-radius: {border_radius_normal}px;
    top: -1px;
}}

QTabBar::tab {{
    background-color: {tab_background};
    color: {text};
    border: {border_width_thin}px solid {border};
    border-bottom: none;
    border-top-left-radius: {border_radius_normal}px;
    border-top-right-radius: {border_radius_normal}px;
    padding: 5px 10px;
    min-width: 80px;
}}

QTabBar::tab:selected {{
    background-color: {tab_selected};
}}

QTabBar::tab:hover {{
    background-color: {tab_hover};
}}

QTabBar::tab:disabled {{
    color: {text_disabled};
}}

/* Table widget */
QTableWidget {{
    background-color: {input_background};
    color: {text};
    gridline-color: {table_border};
    border: {border_width_thin}px solid {table_border};
    border-radius: {border_radius_normal}px;
}}

QTableWidget::item {{
    padding: 5px;
}}

QTableWidget::item:selected {{
    background-color: {table_selection};
    color: {text};
}}

QHeaderView::section {{
    background-color: {table_header};
    color: {text};
    padding: 5px;
    border: {border_width_thin}px solid {table_border};
}}

QTableWidget QTableCornerButton::section {{
    background-color: {table_header};
    border: {border_width_thin}px solid {table_border};
}}

/* Scroll bar */
QScrollBar:horizontal {{
    background-color: {scrollbar_background};
    height: 15px;
    margin: 0px 20px 0px 20px;
}}

QScrollBar:vertical {{
    background-color: {scrollbar_background};
    width: 15px;
    margin: 20px 0px 20px 0px;
}}

QScrollBar::handle:horizontal, QScrollBar::handle:vertical {{
    background-color: {scrollbar_handle};
    border-radius: {border_radius_small}px;
    min-width: 20px;
    min-height: 20px;
}}

QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {{
    background-color: {scrollbar_hover};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background-color: {scrollbar_background};
    border: {border_width_thin}px solid {border};
    subcontrol-origin: margin;
    width: 20px;
    height: 20px;
}}

QScrollBar::add-line:horizontal {{
    subcontrol-position: right;
}}

QScrollBar::sub-line:horizontal {{
    subcontrol-position: left;
}}

QScrollBar::add-line:vertical {{
    subcontrol-position: bottom;
}}

QScrollBar::sub-line:vertical {{
    subcontrol-position: top;
}}

/* Progress bar */
QProgressBar {{
    background-color: {progress_background};
    border: {border_width_thin}px solid {border};
    border-radius: {border_radius_normal}px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {progress_fill};
    width: 10px;
    margin: 0.5px;
}}

/* Slider */
QSlider::groove:horizontal {{
    border: {border_width_thin}px solid {border};
    height: 8px;
    background-color: {progress_background};
    margin: 2px 0;
    border-radius: {border_radius_normal}px;
}}

QSlider::handle:horizontal {{
    background-color: {scrollbar_handle};
    border: {border_width_thin}px solid {border};
    width: 18px;
    margin: -2px 0;
    border-radius: {border_radius_normal}px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {scrollbar_hover};
}}

/* Splitter */
QSplitter::handle {{
    background-color: {border};
}}

QSplitter::handle:horizontal {{
    width: 4px;
}}

QSplitter::handle:vertical {{
    height: 4px;
}}

/* Menu */
QMenu {{
    background-color: {input_background};
    color: {text};
    border: {border_width_thin}px solid {border};
    border-radius: {border_radius_normal}px;
}}

QMenu::item {{
    padding: 5px 20px 5px 20px;
}}

QMenu::item:selected {{
    background-color: {highlight};
    color: {highlighted_text};
}}

QMenu::separator {{
    height: 1px;
    background-color: {border};
    margin: 5px 0px 5px 0px;
}}

/* Tooltip */
QToolTip {{
    background-color: {tooltip_background};
    color: {tooltip_text};
    border: {border_width_thin}px solid {tooltip_border};
    border-radius: {border_radius_normal}px;
    padding: 5px;
}}

/* Message box */
QMessageBox {{
    background-color: {dialog_background};
}}

QMessageBox QLabel {{
    color: {text};
}}

/* Status bar */
QStatusBar {{
    background-color: {surface};
    color: {text};
    border-top: {border_width_thin}px solid {border};
}}

/* Tool bar */
QToolBar {{
    background-color: {surface};
    border: {border_width_thin}px solid {border};
    spacing: 3px;
}}

QToolBar::handle {{
    background-color: {border};
    width: 10px;
    height: 10px;
}}

/* Text edit */
QTextEdit {{
    background-color: {input_background};
    color: {text};
    border: {border_width_thin}px solid {input_border};
    border-radius: {border_radius_normal}px;
}}

QTextEdit:focus {{
    border: {border_width_thin}px solid {input_focus};
}}

QTextEdit:disabled {{
    background-color: {input_background};
    color: {text_disabled};
    border: {border_width_thin}px solid {border};
}}
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_theme_colors(theme_name: str = None) -> Dict[str, str]:
    """
    Get the color scheme for the specified theme.
    
    Args:
        theme_name: Name of the theme to get colors for, or None to use current theme
        
    Returns:
        Dictionary of color names and values
    """
    global CURRENT_THEME
    
    # Use current theme if none specified
    if theme_name is None:
        theme_name = CURRENT_THEME
    
    # Get theme colors
    if theme_name in THEMES:
        return THEMES[theme_name]
    else:
        logger.warning(f"Unknown theme '{theme_name}', using default light theme")
        return LIGHT_THEME


def set_current_theme(theme_name: str) -> bool:
    """
    Set the current theme.
    
    Args:
        theme_name: Name of the theme to set as current
        
    Returns:
        True if the theme was set successfully, False otherwise
    """
    global CURRENT_THEME
    
    if theme_name in THEMES:
        CURRENT_THEME = theme_name
        return True
    else:
        logger.warning(f"Unknown theme '{theme_name}', current theme not changed")
        return False


def get_default_style(theme_name: str = None) -> str:
    """
    Get the default stylesheet for the specified theme.
    
    Args:
        theme_name: Name of the theme to get stylesheet for, or None to use current theme
        
    Returns:
        Stylesheet string
    """
    # Get theme colors
    colors = get_theme_colors(theme_name)
    
    # Format stylesheet with theme colors and constants
    return BASE_STYLESHEET.format(
        # Colors
        **colors,
        
        # Fonts
        font_family=FONT_FAMILY_SANS,
        font_size=FONT_SIZE_NORMAL,
        
        # Layout constants
        border_radius_small=BORDER_RADIUS_SMALL,
        border_radius_normal=BORDER_RADIUS_NORMAL,
        border_radius_medium=BORDER_RADIUS_MEDIUM,
        border_radius_large=BORDER_RADIUS_LARGE,
        
        border_width_thin=BORDER_WIDTH_THIN,
        border_width_normal=BORDER_WIDTH_NORMAL,
        border_width_thick=BORDER_WIDTH_THICK,
        
        input_height_small=INPUT_HEIGHT_SMALL,
        input_height_normal=INPUT_HEIGHT_NORMAL,
        input_height_medium=INPUT_HEIGHT_MEDIUM,
        input_height_large=INPUT_HEIGHT_LARGE
    )


def apply_theme_to_application(theme_name: str = None) -> bool:
    """
    Apply the specified theme to the entire application.
    
    Args:
        theme_name: Name of the theme to apply, or None to use current theme
        
    Returns:
        True if the theme was applied successfully, False otherwise
    """
    # Get application instance
    app = QApplication.instance()
    if app is None:
        logger.warning("Cannot apply theme: No QApplication instance found")
        return False
    
    # Set current theme if specified
    if theme_name is not None:
        if not set_current_theme(theme_name):
            return False
    
    # Get theme colors
    colors = get_theme_colors()
    
    # Set application stylesheet
    app.setStyleSheet(get_default_style())
    
    # Set application palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["window_text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(colors["base"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["alternate_base"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors["tooltip_background"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(colors["tooltip_text"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["button"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["button_text"]))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(colors["bright_text"]))
    palette.setColor(QPalette.ColorRole.Link, QColor(colors["link"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["highlight"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors["highlighted_text"]))
    
    # Set disabled colors
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(colors["text_disabled"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(colors["text_disabled"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(colors["text_disabled"]))
    
    # Apply palette
    app.setPalette(palette)
    
    return True


def apply_style_to_widget(widget: QWidget, theme_name: str = None) -> None:
    """
    Apply appropriate styling to a widget based on its type.
    
    This function applies consistent styling to widgets based on their type,
    ensuring a cohesive appearance throughout the application.
    
    Args:
        widget: The widget to style
        theme_name: Name of the theme to use, or None to use current theme
    """
    # Get theme colors
    colors = get_theme_colors(theme_name)
    
    # Apply styling based on widget type
    if isinstance(widget, QPushButton):
        # Style button
        widget.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors["button"]};
                color: {colors["button_text"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
                padding: 5px 10px;
                min-height: {INPUT_HEIGHT_NORMAL}px;
            }}
            QPushButton:hover {{
                background-color: {colors["button_hover"]};
            }}
            QPushButton:pressed {{
                background-color: {colors["button_pressed"]};
            }}
            QPushButton:disabled {{
                color: {colors["text_disabled"]};
            }}
            QPushButton:default {{
                border: {BORDER_WIDTH_NORMAL}px solid {colors["primary"]};
            }}
        """)
    
    elif isinstance(widget, QLineEdit):
        # Style line edit
        widget.setStyleSheet(f"""
            QLineEdit {{
                background-color: {colors["input_background"]};
                color: {colors["text"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["input_border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
                padding: 2px 5px;
                min-height: {INPUT_HEIGHT_NORMAL}px;
            }}
            QLineEdit:focus {{
                border: {BORDER_WIDTH_THIN}px solid {colors["input_focus"]};
            }}
            QLineEdit:disabled {{
                color: {colors["text_disabled"]};
            }}
        """)
    
    elif isinstance(widget, QComboBox):
        # Style combo box
        widget.setStyleSheet(f"""
            QComboBox {{
                background-color: {colors["input_background"]};
                color: {colors["text"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["input_border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
                padding: 2px 5px;
                min-height: {INPUT_HEIGHT_NORMAL}px;
            }}
            QComboBox:focus {{
                border: {BORDER_WIDTH_THIN}px solid {colors["input_focus"]};
            }}
            QComboBox:disabled {{
                color: {colors["text_disabled"]};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: {BORDER_WIDTH_THIN}px solid {colors["border"]};
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors["input_background"]};
                color: {colors["text"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["input_border"]};
                selection-background-color: {colors["highlight"]};
                selection-color: {colors["highlighted_text"]};
            }}
        """)
    
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        # Style spin box
        widget.setStyleSheet(f"""
            QSpinBox, QDoubleSpinBox {{
                background-color: {colors["input_background"]};
                color: {colors["text"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["input_border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
                padding: 2px 5px;
                min-height: {INPUT_HEIGHT_NORMAL}px;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: {BORDER_WIDTH_THIN}px solid {colors["input_focus"]};
            }}
            QSpinBox:disabled, QDoubleSpinBox:disabled {{
                color: {colors["text_disabled"]};
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                border-bottom: {BORDER_WIDTH_THIN}px solid {colors["border"]};
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                border-top: {BORDER_WIDTH_THIN}px solid {colors["border"]};
            }}
        """)
    
    elif isinstance(widget, QCheckBox):
        # Style checkbox
        widget.setStyleSheet(f"""
            QCheckBox {{
                color: {colors["text"]};
                spacing: 5px;
            }}
            QCheckBox:disabled {{
                color: {colors["text_disabled"]};
            }}
        """)
    
    elif isinstance(widget, QRadioButton):
        # Style radio button
        widget.setStyleSheet(f"""
            QRadioButton {{
                color: {colors["text"]};
                spacing: 5px;
            }}
            QRadioButton:disabled {{
                color: {colors["text_disabled"]};
            }}
        """)
    
    elif isinstance(widget, QLabel):
        # Style label
        widget.setStyleSheet(f"""
            QLabel {{
                color: {colors["text"]};
            }}
            QLabel:disabled {{
                color: {colors["text_disabled"]};
            }}
        """)
    
    elif isinstance(widget, QGroupBox):
        # Style group box
        widget.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
                margin-top: 1.5ex;
                padding-top: 1.5ex;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: {colors["text"]};
            }}
        """)
    
    elif isinstance(widget, QTabWidget):
        # Style tab widget
        widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
                top: -1px;
            }}
            QTabBar::tab {{
                background-color: {colors["tab_background"]};
                color: {colors["text"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                border-bottom: none;
                border-top-left-radius: {BORDER_RADIUS_NORMAL}px;
                border-top-right-radius: {BORDER_RADIUS_NORMAL}px;
                padding: 5px 10px;
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                background-color: {colors["tab_selected"]};
            }}
            QTabBar::tab:hover {{
                background-color: {colors["tab_hover"]};
            }}
            QTabBar::tab:disabled {{
                color: {colors["text_disabled"]};
            }}
        """)
    
    elif isinstance(widget, QTableWidget):
        # Style table widget
        widget.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors["input_background"]};
                color: {colors["text"]};
                gridline-color: {colors["table_border"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["table_border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
            }}
            QTableWidget::item {{
                padding: 5px;
            }}
            QTableWidget::item:selected {{
                background-color: {colors["table_selection"]};
                color: {colors["text"]};
            }}
            QHeaderView::section {{
                background-color: {colors["table_header"]};
                color: {colors["text"]};
                padding: 5px;
                border: {BORDER_WIDTH_THIN}px solid {colors["table_border"]};
            }}
            QTableWidget QTableCornerButton::section {{
                background-color: {colors["table_header"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["table_border"]};
            }}
        """)
        
        # Set alternating row colors
        widget.setAlternatingRowColors(True)
        
        # Configure header
        if widget.horizontalHeader():
            widget.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            widget.horizontalHeader().setHighlightSections(False)
    
    elif isinstance(widget, QProgressBar):
        # Style progress bar
        widget.setStyleSheet(f"""
            QProgressBar {{
                background-color: {colors["progress_background"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {colors["progress_fill"]};
                width: 10px;
                margin: 0.5px;
            }}
        """)
    
    elif isinstance(widget, QSlider):
        # Style slider
        widget.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                height: 8px;
                background-color: {colors["progress_background"]};
                margin: 2px 0;
                border-radius: {BORDER_RADIUS_NORMAL}px;
            }}
            QSlider::handle:horizontal {{
                background-color: {colors["scrollbar_handle"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                width: 18px;
                margin: -2px 0;
                border-radius: {BORDER_RADIUS_NORMAL}px;
            }}
            QSlider::handle:horizontal:hover {{
                background-color: {colors["scrollbar_hover"]};
            }}
        """)
    
    elif isinstance(widget, QSplitter):
        # Style splitter
        widget.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {colors["border"]};
            }}
            QSplitter::handle:horizontal {{
                width: 4px;
            }}
            QSplitter::handle:vertical {{
                height: 4px;
            }}
        """)
    
    elif isinstance(widget, QFrame):
        # Style frame
        widget.setStyleSheet(f"""
            QFrame {{
                background-color: {colors["background"]};
                color: {colors["text"]};
            }}
        """)
    
    elif isinstance(widget, QMainWindow):
        # Style main window
        widget.setStyleSheet(f"""
            QMainWindow {{
                background-color: {colors["background"]};
            }}
            QStatusBar {{
                background-color: {colors["surface"]};
                color: {colors["text"]};
                border-top: {BORDER_WIDTH_THIN}px solid {colors["border"]};
            }}
            QToolBar {{
                background-color: {colors["surface"]};
                border: {BORDER_WIDTH_THIN}px solid {colors["border"]};
                spacing: 3px;
            }}
            QToolBar::handle {{
                background-color: {colors["border"]};
                width: 10px;
                height: 10px;
            }}
        """)
    
    elif isinstance(widget, QDialog):
        # Style dialog
        widget.setStyleSheet(f"""
            QDialog {{
                background-color: {colors["dialog_background"]};
                border: {BORDER_WIDTH_NORMAL}px solid {colors["dialog_border"]};
                border-radius: {BORDER_RADIUS_NORMAL}px;
            }}
        """)
    
    elif isinstance(widget, QMessageBox):
        # Style message box
        widget.setStyleSheet(f"""
            QMessageBox {{
                background-color: {colors["dialog_background"]};
            }}
            QMessageBox QLabel {{
                color: {colors["text"]};
            }}
        """)
    
    # Apply font to widget
    widget.setFont(DEFAULT_FONT)


def apply_style_recursively(widget: QWidget, theme_name: str = None) -> None:
    """
    Apply styling to a widget and all its children recursively.
    
    Args:
        widget: The root widget to style
        theme_name: Name of the theme to use, or None to use current theme
    """
    # Apply style to the widget itself
    apply_style_to_widget(widget, theme_name)
    
    # Apply style to all child widgets
    for child in widget.findChildren(QWidget):
        apply_style_to_widget(child, theme_name)


def get_font_metrics(font: Optional[QFont] = None) -> QFontMetrics:
    """
    Get font metrics for the specified font or the default font.
    
    Args:
        font: Font to get metrics for, or None to use default font
        
    Returns:
        Font metrics object
    """
    if font is None:
        font = DEFAULT_FONT
    
    return QFontMetrics(font)


def calculate_text_width(text: str, font: Optional[QFont] = None) -> int:
    """
    Calculate the width of text in pixels.
    
    Args:
        text: Text to measure
        font: Font to use for measurement, or None to use default font
        
    Returns:
        Width of text in pixels
    """
    metrics = get_font_metrics(font)
    return metrics.horizontalAdvance(text)


def calculate_text_height(text: str, font: Optional[QFont] = None) -> int:
    """
    Calculate the height of text in pixels.
    
    Args:
        text: Text to measure
        font: Font to use for measurement, or None to use default font
        
    Returns:
        Height of text in pixels
    """
    metrics = get_font_metrics(font)
    return metrics.height()


def create_font(family: str = FONT_FAMILY_SANS, 
               size: int = FONT_SIZE_NORMAL,
               weight: QFont.Weight = FONT_WEIGHT_NORMAL,
               style: QFont.Style = FONT_STYLE_NORMAL) -> QFont:
    """
    Create a font with the specified properties.
    
    Args:
        family: Font family
        size: Font size in points
        weight: Font weight
        style: Font style
        
    Returns:
        Created font
    """
    # Use first font in family list
    family_name = family.split(',')[0].strip()
    
    # Create font
    font = QFont(family_name, size)
    font.setWeight(weight)
    font.setStyle(style)
    
    return font


def is_dark_theme(theme_name: str = None) -> bool:
    """
    Check if the specified theme is a dark theme.
    
    Args:
        theme_name: Name of the theme to check, or None to use current theme
        
    Returns:
        True if the theme is dark, False otherwise
    """
    # Get theme colors
    colors = get_theme_colors(theme_name)
    
    # Check if background color is dark
    bg_color = QColor(colors["background"])
    
    # Calculate luminance (perceived brightness)
    # Formula: 0.299*R + 0.587*G + 0.114*B
    luminance = (0.299 * bg_color.red() + 0.587 * bg_color.green() + 0.114 * bg_color.blue()) / 255.0
    
    # Theme is dark if luminance is less than 0.5
    return luminance < 0.5


def get_contrasting_color(color: Union[str, QColor]) -> QColor:
    """
    Get a contrasting color (black or white) for the specified color.
    
    Args:
        color: Color to get contrast for (hex string or QColor)
        
    Returns:
        QColor with contrasting color (black or white)
    """
    # Convert to QColor if string
    if isinstance(color, str):
        qcolor = QColor(color)
    else:
        qcolor = color
    
    # Calculate luminance (perceived brightness)
    # Formula: 0.299*R + 0.587*G + 0.114*B
    luminance = (0.299 * qcolor.red() + 0.587 * qcolor.green() + 0.114 * qcolor.blue()) / 255.0
    
    # Use white text on dark backgrounds, black text on light backgrounds
    return QColor(Qt.GlobalColor.white) if luminance < 0.5 else QColor(Qt.GlobalColor.black)


def initialize_styles() -> None:
    """
    Initialize the styles module.
    
    This function sets up the styles module, detecting system theme preferences
    and configuring default styles accordingly.
    """
    global CURRENT_THEME
    
    # Log initialization
    logger.debug("Initializing styles module")
    
    # Detect system theme preference if possible
    try:
        # Check if running on Windows
        if sys.platform == 'win32':
            import winreg
            try:
                # Check Windows registry for dark mode setting
                registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
                key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                if value == 0:
                    CURRENT_THEME = "dark"
                    logger.debug("Detected Windows dark mode preference")
                else:
                    CURRENT_THEME = "light"
                    logger.debug("Detected Windows light mode preference")
            except Exception:
                # Default to light theme if registry access fails
                CURRENT_THEME = "light"
                logger.debug("Could not detect Windows theme preference, using light theme")
        
        # Check if running on macOS
        elif sys.platform == 'darwin':
            import subprocess
            try:
                # Check macOS dark mode setting
                result = subprocess.run(
                    ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                    capture_output=True, text=True, check=False
                )
                if result.stdout.strip() == 'Dark':
                    CURRENT_THEME = "dark"
                    logger.debug("Detected macOS dark mode preference")
                else:
                    CURRENT_THEME = "light"
                    logger.debug("Detected macOS light mode preference")
            except Exception:
                # Default to light theme if command fails
                CURRENT_THEME = "light"
                logger.debug("Could not detect macOS theme preference, using light theme")
        
        # Default to light theme for other platforms
        else:
            CURRENT_THEME = "light"
            logger.debug(f"No theme detection for platform {sys.platform}, using light theme")
    
    except Exception as e:
        # Default to light theme if any error occurs
        CURRENT_THEME = "light"
        logger.debug(f"Error detecting system theme: {e}, using light theme")
    
    logger.debug(f"Styles module initialized with {CURRENT_THEME} theme")


# Initialize styles module
initialize_styles()
