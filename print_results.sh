#!/bin/bash

RESULT_DIR=/home/sdb2/yinxiaojie/osrci/32

echo "Baseline and OpenMax results (non-generative):"
cat `find ${RESULT_DIR}/evaluations/ | grep json | sort | head -1`

echo "Results with generated data:"
cat `find ${RESULT_DIR}/evaluations/ | grep json | sort | tail -1`
