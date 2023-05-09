# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../core"))
sys.path.insert(0, os.path.abspath("../scenarios"))

project = 'ECOMOD'
copyright = '2023, ChainedGenius'
author = 'ChainedGenius'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo",
              "sphinx.ext.viewcode",
              "sphinx.ext.autodoc"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
#latex_documents = [
#    ('index', 'yourdoc.tex', u'ECOMOD Tech', u'ECOMOD Team', 'Manual')
#]
