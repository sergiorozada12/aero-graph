#!/bin/bash

cd ..

d=`date +%Y%m%d-%H%M`

mkdir ../output/$d

python3 model_rf.py 2>&1 | tee ../output/$d/model_rf.out
python3 model_dt.py 2>&1 | tee ../output/$d/model_dt.out
python3 model_gbt.py 2>&1 | tee ../output/$d/model_gbt.out
python3 model_knn.py 2>&1 | tee ../output/$d/model_knn.out
python3 model_lr.py 2>&1 | tee ../output/$d/model_lr.out
python3 model_mlp.py 2>&1 | tee ../output/$d/model_mlp.out
python3 model_mlp_sergio.py 2>&1 | tee ../output/$d/model_mlp_sergio.out