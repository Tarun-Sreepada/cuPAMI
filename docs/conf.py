import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'cuPAMI'
author = 'cuPAMI Contributors'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
