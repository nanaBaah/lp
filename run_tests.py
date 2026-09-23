# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Run the solver test suite
# MAGIC %md
# MAGIC # Run the solver test suite
# MAGIC
# MAGIC For working in the Databricks UI.
# MAGIC
# MAGIC Working locally instead? Skip this notebook and run:
# MAGIC
# MAGIC ```
# MAGIC pytest solutions/innovations/2026_new_redispatch_mechanism -q
# MAGIC ```
# MAGIC
# MAGIC **Reading the result:** `passed` is what you want. `xfailed` means a
# MAGIC known defect is still present and is not a problem — see the "Known
# MAGIC defects" section of `tests/test_lp.py`. `failed` or `xpassed` both need
# MAGIC attention.

# COMMAND ----------

import os
import sys

import pytest

# Workspace files are read-only, so stop pytest writing .pyc and its cache there.
sys.dont_write_bytecode = True

# Databricks runs a notebook from its own folder, which is where the tests live.
if not os.path.exists("tests"):
    raise SystemExit(f"tests/ not found in {os.getcwd()}")

# COMMAND ----------

retcode = pytest.main(["-q", "-p", "no:cacheprovider", "tests"])
assert retcode == 0, f"test suite failed (exit code {retcode})"
