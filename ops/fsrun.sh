#!/bin/bash
srun --job-name=cosmos \
     --time=0-02:00:00 \
     --partition=performance \
     --pty bash -l