# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = 'nmrcryspy'
copyright = '2023, Maxwell Venetos'
author = 'Maxwell Venetos'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

autodoc_mock_imports = ['numpy', 'eigenn', 'e3nn', 'torch', 'pymatgen']

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    # "sphinxcontrib.bibtex",
    "breathe",
    "sphinxjp.themes.basicstrap",
    # "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "sphinx.ext.todo",
    "recommonmark",
    "versionwarning.extension",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# ---------------------------------------------------------------------------- #
#                                  HTML theme                                  #
# ---------------------------------------------------------------------------- #

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Some html_theme options are 'alabaster', 'bootstrap', 'sphinx_rtd_theme',
# 'classic', 'basicstrap'
html_theme = "basicstrap"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    # Set the lang attribute of the html tag. Defaults to 'en'
    "lang": "en",
    # Disable showing the sidebar. Defaults to 'false'
    "nosidebar": False,
    # Show header searchbox. Defaults to false. works only "nosidebar=True",
    "header_searchbox": False,
    # Put the sidebar on the right side. Defaults to false.
    "rightsidebar": False,
    # Set the width of the sidebar. Defaults to 3
    "sidebar_span": 3,
    # Fix navbar to top of screen. Defaults to true
    "nav_fixed_top": True,
    # Fix the width of the sidebar. Defaults to false
    "nav_fixed": True,
    # Set the width of the sidebar. Defaults to '900px'
    "nav_width": "300px",
    # Fix the width of the content area. Defaults to false
    "content_fixed": False,
    # Set the width of the content area. Defaults to '900px'
    "content_width": "900px",
    # Fix the width of the row. Defaults to false
    "row_fixed": False,
    # Disable the responsive design. Defaults to false
    "noresponsive": False,
    # Disable the responsive footer relbar. Defaults to false
    "noresponsiverelbar": False,
    # Disable flat design. Defaults to false.
    # Works only "bootstrap_version = 3"
    "noflatdesign": False,
    # Enable Google Web Font. Defaults to false
    # "googlewebfont": True,
    # Set the URL of Google Web Font's CSS.
    # Defaults to 'https://fonts.googleapis.com/css?family=Text+Me+One'
    # "googlewebfont_url": "https://fonts.googleapis.com/css?family=Roboto+Script+One",  # NOQA
    # "googlewebfont_url": "https://fonts.googleapis.com/css2?family=Inter",
    # Set the Style of Google Web Font's CSS.
    # Defaults to "font-family: 'Text Me One', sans-serif;"
    # "googlewebfont_style": "font-family: Helvetica",
    # "googlewebfont_style": "font-family: 'Inter', sans-serif;",
    # Set 'navbar-inverse' attribute to header navbar. Defaults to false.
    "header_inverse": True,
    # Set 'navbar-inverse' attribute to relbar navbar. Defaults to false.
    "relbar_inverse": False,
    # Enable inner theme by Bootswatch. Defaults to false
    "inner_theme": False,
    # Set the name of inner theme. Defaults to 'bootswatch-simplex'
    # "inner_theme_name": "bootswatch-Yeti",
    # Select Twitter bootstrap version 2 or 3. Defaults to '3'
    "bootstrap_version": "3",
    # Show "theme preview" button in header navbar. Defaults to false.
    "theme_preview": False,
}

html_style = "style.css"
html_title = f"nmrcryspy:docs v{release}"
html_logo = "_static/nmrcryspy_logo.png"
# html_favicon = "_static/favicon.ico"
html_last_updated_fmt = ""

html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html"],
    "using/windows": ["searchbox.html", "windowssidebar.html"],
}
# ---------------------------------------------------------------------------- #

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]