v_dataset=`echo "${0##*/}" | cut -d'.' -f1`
s_dataset='imagenet'
device='cuda:0'
query_budget_list=(2000 4000 6000 8000)
cl_epoch_list=(100)
attack_list=(
             'swiftthief'
             )

supcon_batch_size_list=(512)
supcon_lambda_list=(1.0)
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
            save_path=save/attack/${v_dataset}_${s_dataset}/${attack}/budget_${query_budget}/model.pt
            python ${attack}.py \
                  --victim_dataset ${v_dataset} \
                  --attack_dataset ${s_dataset} \
                  --query_budget ${query_budget} \
                  --device ${device} \
                  --sl_lr 1e-2 \
                  --sl_epoch 500 \
                  --sl_aug_interval 50 \
                  --save_path $save_path \
                  --victim_path save/victim/${v_dataset}/model.pt 
        done
      done
    done
  done
done


