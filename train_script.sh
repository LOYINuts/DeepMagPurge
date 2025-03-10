#!/bin/bash
conda activate ~/gh/envs/bioenv/
nohup python Trainscript.py > train.log 2>&1 &