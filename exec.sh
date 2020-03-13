python3 etl.py 2>&1 | tee ../output/etl.out
python3 feature_engineering.py 2>&1 | tee ../output/feature_engineering.out
python3 feature_selection.py 2>&1 | tee ../output/feature_selection.out
