H=2
CLS_ITER=3
BATCH=72
DIM_VAL=96
DIM_ATTN=128
ATTN_TYPE=query_selector_0.85
LR=0.0002
set -x
python3.7 train_model.py  --cls_iterations $CLS_ITER --heads $H --batch_size $BATCH --dim_val $DIM_VAL --dim_attn $DIM_ATTN --attn_type $ATTN_TYPE --lr $LR --data data/bpi_12_w.csv
python3.7 evaluate_model.py --heads $H --dim_val $DIM_VAL --dim_attn $DIM_ATTN --attn_type $ATTN_TYPE --data data/bpi_12_w.csv