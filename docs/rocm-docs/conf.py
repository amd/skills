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
html_theme_options = {
    "flavor": "rocm-ai",
    "use_repository_button": True,
    "use_download_button": True,
    "show_navbar_depth": 2,
}
html_title = "AMD Skills documentation"
suppress_warnings = ["etoc.toctree"]
external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "amd-skills"
extensions = [
    "rocm_docs",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autosectionlabel",
]
# Render ```mermaid fences (as written for GitHub) as diagrams.
myst_fence_as_directive = ["mermaid"]
exclude_patterns = []
# Allow :ref:`page:Heading` links, checked at build time.
autosectionlabel_prefix_document = True

# -- Sphinx setup ----------------------------------------------------------


def setup(app):
    """Sphinx setup"""
    return {"parallel_read_safe": True, "parallel_write_safe": True}
