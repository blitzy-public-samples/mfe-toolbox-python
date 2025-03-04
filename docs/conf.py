# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Add the project root directory to the path so that autodoc can find the modules
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Import the project's version information
try:
    from mfe.version import (
        __version__, 
        VERSION_MAJOR, 
        VERSION_MINOR, 
        VERSION_PATCH,
        __author__,
        __copyright__,
        get_release_date
    )
except ImportError:
    # Fallback if version module can't be imported
    __version__ = "4.0.0"
    VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH = 4, 0, 0
    __author__ = "Kevin Sheppard"
    __copyright__ = f"Copyright (c) 2009-{datetime.datetime.now().year}, Kevin Sheppard"
    def get_release_date(version=None):
        return "2023-11-01"  # Placeholder date

# -- Project information -----------------------------------------------------
project = 'MFE Toolbox'
copyright = __copyright__
author = __author__

# The full version, including alpha/beta/rc tags
release = __version__
# The short X.Y version
version = f"{VERSION_MAJOR}.{VERSION_MINOR}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',           # Generate documentation from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables for modules
    'sphinx.ext.viewcode',          # Add links to view source code
    'sphinx.ext.napoleon',          # Support for NumPy and Google style docstrings
    'sphinx.ext.mathjax',           # Render math via MathJax
    'sphinx.ext.intersphinx',       # Link to other projects' documentation
    'sphinx.ext.todo',              # Support for TODO items
    'sphinx.ext.coverage',          # Documentation coverage checking
    'sphinx.ext.ifconfig',          # Conditional content based on config values
    'sphinx.ext.githubpages',       # GitHub pages support
    'sphinx_rtd_theme',             # Read the Docs theme
    'IPython.sphinxext.ipython_console_highlighting',  # IPython highlighting
    'IPython.sphinxext.ipython_directive',             # IPython directives
    'matplotlib.sphinxext.plot_directive',             # Matplotlib plot directive
    'nbsphinx',                     # Jupyter notebook support
    'sphinx_copybutton',            # Add copy button to code blocks
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = 'description'

# Only show class docstring and not __init__ docstring
autoclass_content = 'class'

# -- Options for napoleon ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'numba': ('https://numba.pydata.org/numba-doc/latest/', None),
    'pyqt6': ('https://www.riverbankcomputing.com/static/Docs/PyQt6/', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'css/custom.css',
]

# Custom JavaScript files
html_js_files = [
    'js/mfe-docs.js',
    'js/mermaid-init.js',
]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/favicon.ico'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'MFEToolboxdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '11pt',

    # Additional stuff for the LaTeX preamble.
    'preamble': r'''\
        \usepackage{charter}\
        \usepackage[defaultsans]{lato}\
        \usepackage{inconsolata}
    ''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', 'MFEToolbox.tex', 'MFE Toolbox Documentation',
     'Kevin Sheppard', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'mfetoolbox', 'MFE Toolbox Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'MFEToolbox', 'MFE Toolbox Documentation',
     author, 'MFEToolbox', 'Financial Econometrics Toolbox for Python.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_execute = 'auto'
nbsphinx_allow_errors = False
nbsphinx_timeout = 60  # seconds, default is 30

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Setup for sphinx-apidoc -------------------------------------------------
# This is needed for autodoc to find the modules
def setup(app):
    # Add custom CSS
    app.add_css_file('css/custom.css')
    
    # Add custom JavaScript
    app.add_js_file('js/mfe-docs.js')
    app.add_js_file('js/mermaid-init.js')
    
    # Add custom directives or configurations here if needed
    app.add_config_value('release_date', get_release_date(__version__), 'env')

# -- Additional configuration ------------------------------------------------

# Add the 'copybutton' JavaScript to the documentation.
copybutton_selector = "div.highlight pre"

# Enable mermaid diagrams
mermaid_version = "9.3.0"  # Use the appropriate version
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"

# Configure mathjax
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax_config = {
    'tex2jax': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': True,
        'processEnvironments': True,
    },
    'HTML-CSS': {
        'fonts': ['TeX'],
        'scale': 90,
        'linebreaks': {
            'automatic': True,
        },
    },
    'SVG': {
        'scale': 90,
        'linebreaks': {
            'automatic': True,
        },
    },
}

# -- Project-specific configurations -----------------------------------------

# Release date for the documentation
release_date = get_release_date(__version__)

# Project URLs
project_urls = {
    'Source Code': 'https://github.com/bashtage/arch',  # Update with actual repository
    'Issue Tracker': 'https://github.com/bashtage/arch/issues',  # Update with actual repository
    'Documentation': 'https://mfe-toolbox.readthedocs.io/',  # Update with actual documentation URL
}

# Sidebar links
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}
