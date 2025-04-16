project=`echo "$( cd "$( dirname "$0" )" && pwd )" | cut -d'/' -f5`
v_dataset=`echo "${0##*/}" | cut -d'.' -f1`
s_dataset='imagenet'
device='cuda:1'
query_budget_list=(2000 4000 6000 8000)
cl_epoch_list=(100)
attack_list=(
             'swiftthief'
             )

exp_keyword='sup_con_exp1'

supcon_batch_size_list=(512)
supcon_lambda_list=(1.0)
total_cnt=6
cnt=0
for query_budget in ${query_budget_list[@]}
do
  for attack in ${attack_list[@]}
  do
    for supcon_batch_size in ${supcon_batch_size_list[@]}
    do
      for supcon_lambda in ${supcon_lambda_list[@]}
      do
        for cl_epoch in ${cl_epoch_list[@]}
        do

            exp_name=${project}_${v_dataset}_${attack}_${query_budget}
            save_path=save/attack/${exp_keyword}/${v_dataset}_${s_dataset}/${attack}/budget_${query_budget}/model.pt
            python ${attack}.py \
                  --victim_dataset ${v_dataset} \
                  --attack_dataset ${s_dataset} \
                  --query_budget ${query_budget} \
                  --device ${device} \
                  --sl_lr 1e-2 \
                  --sl_epoch 500 \
                  --sl_aug_interval 50 \
                  --save_path $save_path \
                  --victim_path save/victim/${v_dataset}/model.pt \
                  --progress ${cnt}/${total_cnt} \
                  2>&1 | \
                  tee log/${exp_name}.log
            cnt=$((cnt+1))

        done
      done
    done
  done
done


