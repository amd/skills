.. meta::
   :description: AMD Skills give your coding agent the knowledge to work with AMD hardware and software.
   :keywords: AMD Skills, AMD, ROCm, GPU

========================
AMD Skills documentation
========================

AMD Skills give your coding agent the knowledge to work with AMD hardware and software.

Skills use the standardized `Agent Skills <https://github.com/anthropics/skills>`_
format and work with coding agents such as Cursor, Claude Code, OpenAI Codex,
and Gemini CLI. To learn how skills work, see
:doc:`What is a skill? <concepts/index>`.

Install a skill
===============

You need `Node.js <https://nodejs.org>`_ to run the installer. To install the
``local-ai-use`` skill for Claude Code, run:

.. code-block:: bash

   npx skills add amd/skills --skill local-ai-use --agent claude-code

For other agents and install options, see
:doc:`Installing AMD Skills <install/installation>`.

Find a skill
============

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Client-native
      :link: catalog:client-native
      :link-type: ref

      Run and optimize on Ryzen AI.

   .. grid-item-card:: Cross-stack
      :link: catalog:cross-stack
      :link-type: ref

      Work across AMD hardware, from client to cloud.

   .. grid-item-card:: Server-native
      :link: catalog:server-native
      :link-type: ref

      Run and optimize on AMD Instinct GPUs and EPYC CPUs.

   .. grid-item-card:: Diagnose a failure
      :link: https://github.com/amd/skills/blob/main/skills/rocm-doctor/SKILL.md
      :link-type: url

      Diagnose and fix common ROCm, HIP, PyTorch, and llama.cpp failures with
      ``rocm-doctor``.

To try a skill from start to finish, follow one of the
:doc:`walkthroughs <walkthroughs/index>`.

.. note::

   The catalog is built in the open and evolves frequently. Some skills are in
   Tech Preview. For status, see the underlying product.

The source is in the `amd/skills <https://github.com/amd/skills>`_ public
repository.
