# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
from dataclasses import asdict

from sphinxawesome_theme import LinkIcon, ThemeOptions

# -- Project information

project = 'ONTraC'
copyright = '2024-2025, Guo-Cheng Yuan Lab'
author = 'Wen Wang, Shiwei Zheng, Crystal Shin, Joselyn Chavez, Guo-Cheng Yuan'

release = '1.2'
version = '1.2.2'

# -- General configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
]

# in conf.py, after your extensions list:
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",       # only register .md once
    ".rmd": "myst-nb",      # .md and .rmd are the same
    ".qmd": "myst-nb",      # .qmd is also supported
    ".ipynb": "myst-nb",     # we want notebook files too
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "colon_fence",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinxawesome_theme"


theme_options = ThemeOptions(
    logo_light="images/logo.ico",
    logo_dark="images/logo_dark.ico",
    show_scrolltop=True,
    show_breadcrumbs=True,
    awesome_external_links=True,
    main_nav_links={
        "Home": "index",
        "Installation": "installation",
        "Tutorials": "tutorials/index",
        "Examples": "examples/index",
        "How it works": "how_it_works/index",
        #"API": "api", # TODO: Add API page
        "Contributors": "contributors",
        "Citation": "citation",
        #"FAQ": "faq", # TODO: Add FAQ page
    },
    extra_header_link_icons={
        "GitHub": LinkIcon(
            link="https://github.com/gyuanlab/ONTraC",
            icon='<svg fill="currentColor" height="26px" style="margin-top:-2px;display:inline" viewBox="0 0 45 44" xmlns="http://www.w3.org/2000/svg"><path clip-rule="evenodd" d="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 17.736 14.853 20.608 1.087.2 1.483-.47 1.483-1.047 0-.516-.019-1.881-.03-3.693-6.04 1.312-7.315-2.912-7.315-2.912-.988-2.51-2.412-3.178-2.412-3.178-1.972-1.346.149-1.32.149-1.32 2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 1.803.197-1.403.759-2.36 1.379-2.903-4.823-.548-9.894-2.412-9.894-10.734 0-2.37.847-4.31 2.236-5.828-.224-.55-.969-2.759.214-5.748 0 0 1.822-.584 5.972 2.226 1.732-.482 3.59-.722 5.437-.732 1.845.01 3.703.25 5.437.732 4.147-2.81 5.967-2.226 5.967-2.226 1.185 2.99.44 5.198.217 5.748 1.392 1.517 2.232 3.457 2.232 5.828 0 8.344-5.078 10.18-9.916 10.717.779.67 1.474 1.996 1.474 4.021 0 2.904-.027 5.247-.027 5.96 0 .58.392 1.256 1.493 1.044C37.981 40.375 44.2 32.24 44.2 22.647c0-11.996-9.726-21.72-21.722-21.72" fill="currentColor" fill-rule="evenodd"></path></svg>',
        )
    },
)

html_theme_options = asdict(theme_options)

html_context = {"default_mode": "auto"}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

html_js_files = [
    "custom.js",
]

# -- conf.py additions --

# Configure myst-nb
nb_render_priority = {
    "html": (
        "outputs.HTMLOutput",
        "execute_result",
        "display_data",
        "text/plain",
        "text/html",
        "text/latex",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "application/json"
    )
}

# -- Options for Autosummary, Autodoc, type hints, and Napoleon docstring format

autosummary_generate = True
autodoc_docstring_signature = True
typehint_defaults = "braces"
typehints_use_signature_return = True
typehints_document_rtype = True
always_use_bar_union = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = True
numpydoc_show_class_members = False

html_title = "ONTraC"

# -- Options for extensions -------------------------------------------------------------------------------

nb_execution_mode = "off"

# -- Options for EPUB output
epub_show_urls = 'footnote'
