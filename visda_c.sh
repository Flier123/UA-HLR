DATASET="visda" # imagenet_c domainnet126 officehome
METHOD="uahlr"          # shot nrc plue difo

echo DATASET: $DATASET
echo METHOD: $METHOD

# 如果用户未传入 GPU，则默认使用 1
GPU=${GPU:-1}

s_list=(0)
t_list=(1)
for s in ${s_list[*]}; do
    for t in ${t_list[*]}; do
    if [[ $s = $t ]]
        then
        continue
    fi
        CUDA_VISIBLE_DEVICES=$GPU python image_target_of_oh_vs.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
            SETTING.S "$s" SETTING.T "$t" &
        wait
    done
done 
