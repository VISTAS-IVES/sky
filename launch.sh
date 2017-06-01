#!/bin/bash
while read $options; do export $options; sbatch run.sh; done <conditions.txt

