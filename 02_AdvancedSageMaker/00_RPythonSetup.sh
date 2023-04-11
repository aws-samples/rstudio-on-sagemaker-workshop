#!/bin/bash

# Reinstall numpy to workaround conflict between reticulate and RStudio v4.x due to BLAS library dependency
# The reinstall process takes 5 minutes.
# https://github.com/rstudio/reticulate/issues/1257

python_path=$(which python)
echo $python_path
sudo $python_path -m pip install --no-user --force-reinstall --no-binary numpy numpy