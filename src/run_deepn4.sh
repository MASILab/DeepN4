#!/bin/bash

echo within run_deepn4.sh
proj_path=/src

source $proj_path/venv/bin/activate
python $proj_path/inference.py $@
deactivate