#! /bin/sh
# This assumes you are already in a venv with correct requirements

PYTHONPATH=.. PYTHONUNBUFFERED=1 THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu,on_unused_input=warn \
time python small_config.py
