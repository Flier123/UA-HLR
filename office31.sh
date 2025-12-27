
DATASET="office31" # imagenet_c domainnet126 officehome
METHOD="uahlr"          # shot nrc plue difo

echo DATASET: $DATASET
echo METHOD: $METHOD

pairs=(
  "0 1"
  "0 2"
  "1 0"
  "1 2"
  "2 0"
  "2 1"
)

# 如果用户未传入 GPU，则默认使用 1
GPU=${GPU:-1}

for pair in "${pairs[@]}"; do
    set -- $pair      # 将字符串拆成两个参数
    s=$1
    t=$2
    CUDA_VISIBLE_DEVICES=$GPU python image_target_of_oh_vs.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
        SETTING.S "$s" SETTING.T "$t" &
    wait
done

# python avg_result.py --base_dir "/root/code/hlf/DIFO/output/target/uda/office" --method_name "difo"


