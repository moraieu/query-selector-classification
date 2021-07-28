H=2
CLS_ITER=5
BATCH=72
DIM_VAL=96
DIM_ATTN=256
ATTN_TYPE=query_selector_0.875
LR=0.0002
python3.7 train_model.py  --cls_iterations $CLS_ITER --heads $H --batch_size $BATCH --dim_val $DIM_VAL --dim_attn $DIM_ATTN --attn_type $ATTN_TYPE --lr $LR --data data/helpdesk.csv
python3.7 evaluate_model.py --heads $H --dim_val $DIM_VAL --dim_attn $DIM_ATTN --attn_type $ATTN_TYPE --data data/helpdesk.csv