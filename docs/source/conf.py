"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html.
"""

import mspdetect

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mspdetect"
copyright = "2026, Noemi Bührer"  # noqa: A001
author = "Loris Keller, Yoel Pérez Haas, Noemi Bührer"
gh_username = "lycosystem"
version = mspdetect.__version__
release = mspdetect.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinxcontrib.programoutput",
]


templates_path = ["_templates"]
exclude_patterns = []

# document classes and their constructors
autoclass_content = "class"

# sort members by source
autodoc_member_order = "bysource"

# show type hints
autodoc_typehints = "description"

# create links to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# autodoc_pydantic settings
autodoc_use_legacy_class_based = True  # needed for autodoc_pydantic with Sphinx 9
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_show_field_summary = False
autodoc_pydantic_settings_member_order = "bysource"
autodoc_pydantic_field_show_constraints = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/{gh_username}/{project}",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_navbar_depth": 3,
    "home_page_in_toc": True,
}
