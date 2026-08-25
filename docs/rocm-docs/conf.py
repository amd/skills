# Configuration file for the Sphinx documentation builder.
# pylint: disable=invalid-name

# -- Project information ---------------------------------------------------

project = "AMD Skills"
author = "Advanced Micro Devices, Inc."
# pylint: disable=redefined-builtin
copyright = "Copyright (c) Advanced Micro Devices, Inc. All rights reserved."
# pylint: enable=redefined-builtin

# -- General configuration -------------------------------------------------

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = "AMD Skills documentation"
suppress_warnings = ["etoc.toctree"]
external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "amd-skills"
extensions = [
    "rocm_docs",
    "sphinx_design",
]
exclude_patterns = []

# -- Sphinx setup ----------------------------------------------------------


def setup(app):
    """Sphinx setup"""
    return {"parallel_read_safe": True, "parallel_write_safe": True}
