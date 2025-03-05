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

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute.
sys.path.insert(0, os.path.abspath('..'))

# Import the project version
try:
    from mfe.version import __version__ as version
    from mfe.version import __author__, __copyright__, __title__, __description__
except ImportError:
    # If the package is not installed, try to get version from pyproject.toml
    import tomli
    with open('../pyproject.toml', 'rb') as f:
        pyproject = tomli.load(f)
    version = pyproject['project']['version']
    __author__ = pyproject['project']['authors'][0]['name']
    __title__ = pyproject['project']['name']
    __description__ = pyproject['project']['description']
    __copyright__ = f"2023, {__author__}"

# -- Project information -----------------------------------------------------

project = __title__
copyright = __copyright__
author = __author__

# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',           # Generate documentation from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables for modules
    'sphinx.ext.viewcode',          # Add links to view source code
    'sphinx.ext.napoleon',          # Support for NumPy and Google style docstrings
    'sphinx.ext.mathjax',           # Render math via MathJax
    'sphinx.ext.intersphinx',       # Link to other project's documentation
    'sphinx.ext.todo',              # Support for todo items
    'sphinx.ext.coverage',          # Check documentation coverage
    'sphinx.ext.ifconfig',          # Conditional content based on config values
    'sphinx.ext.githubpages',       # GitHub pages support
    'sphinx_rtd_theme',             # Read the Docs theme
    'nbsphinx',                     # Jupyter notebook support
    'IPython.sphinxext.ipython_console_highlighting',  # IPython highlighting
    'IPython.sphinxext.ipython_directive',             # IPython directive
    'matplotlib.sphinxext.plot_directive',             # Matplotlib plot directive
    'sphinx_copybutton',            # Add copy button to code blocks
    'sphinx.ext.graphviz',          # Support for Graphviz graphs
    'sphinxcontrib.mermaid',        # Support for Mermaid diagrams
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for autodoc -----------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = 'description'

# Only show class docstring and not __init__ docstring
autoclass_content = 'class'

# Sort members by type
autodoc_member_order = 'groupwise'

# Document special methods
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__, __call__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for intersphinx -------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
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

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

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
        \usepackage{inconsolata}\
    ''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', 'MFEToolbox.tex', 'MFE Toolbox Documentation',
     __author__, 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'mfetoolbox', 'MFE Toolbox Documentation',
     [__author__], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'MFEToolbox', 'MFE Toolbox Documentation',
     __author__, 'MFEToolbox', __description__,
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
epub_identifier = 'https://github.com/bashtage/arch'
epub_scheme = 'URL'
epub_uid = f"MFEToolbox-{version}"

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Extension configuration -------------------------------------------------

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = 'auto'
nbsphinx_allow_errors = False
nbsphinx_timeout = 60  # seconds, default is 30

# -- Napoleon settings -------------------------------------------------------
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
napoleon_use_keyword = True
napoleon_custom_sections = None

# -- Mermaid configuration ---------------------------------------------------
mermaid_params = [
    '--theme', 'default',
    '--width', '100%',
    '--backgroundColor', 'transparent'
]
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"

# -- Copybutton configuration ------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Setup for sphinx-apidoc -------------------------------------------------
# In case sphinx-apidoc is used to generate API docs
def setup(app):
    # Add custom CSS
    app.add_css_file('css/custom.css')
    
    # Add custom JavaScript
    app.add_js_file('js/mermaid-init.js')
    
    # Register a custom directive for version-specific content
    from sphinx.directives import other
    
    # Custom event handlers
    app.connect('autodoc-process-docstring', process_docstrings)
    
    # Custom configuration values
    app.add_config_value('package_version', version, 'env')
    app.add_config_value('is_release', 'dev' not in version, 'env')

def process_docstrings(app, what, name, obj, options, lines):
    """Process docstrings to add custom formatting or annotations."""
    # Example: Add a note about Numba acceleration to relevant functions
    if what in ('function', 'method') and hasattr(obj, '__module__'):
        module_name = obj.__module__
        if '_numba_core' in module_name or any(x in module_name for x in ['univariate', 'multivariate', 'bootstrap']):
            lines.append('')
            lines.append('.. note::')
            lines.append('   This function is accelerated using Numba JIT compilation for optimal performance.')
            lines.append('')
