#!/bin/bash
/home/lys/gh/envs/bioenv/bin/python ./other/art_simulate_genomes.py;
/home/lys/gh/envs/bioenv/bin/python ./other/concat_all_fq.py;
/home/lys/gh/envs/bioenv/bin/python Trainscript.py;