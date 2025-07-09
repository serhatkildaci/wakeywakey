# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'WakeyWakey'
copyright = '2025, Serhat KILDACI'
author = 'Serhat KILDACI'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True 