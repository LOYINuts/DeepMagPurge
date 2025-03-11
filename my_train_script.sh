#!/bin/bash
/home/lys/gh/envs/bioenv/bin/python ./other/art_simulate_genomes.py > train_log.log 2>&1;
/home/lys/gh/envs/bioenv/bin/python ./other/concat_all_fq.py >> train_log.log 2>&1;
/home/lys/gh/envs/bioenv/bin/python Trainscript.py >> train_log.log 2>&1;