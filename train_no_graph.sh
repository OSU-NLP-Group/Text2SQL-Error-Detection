cuda=6
exp_num=3

dir=/data/ED4SP 
batch=16
lr=3e-5
epoch=20
warmup_factor=10
gnn_model=none

test_pp=top1

# exp_group=SmBop
# exp_group=Bridge
# exp_group=NatSQL
# exp_group=NatSQL_spider_tr
# exp_group=NatSQL_ori
exp_group=ResdNatSQL


seed=${exp_num}00

expr_name=CodeBERT


### Main settings.
# train_dat=preprocessing/datasets/bridge/ed_bridge_full_beam_train_sim2.dat
# dev_dat=preprocessing/datasets/bridge/ed_bridge_full_beam_dev_sim2.dat
# test_dat=preprocessing/datasets/bridge/ed_bridge_beam_test_sim2.dat


# train_dat=preprocessing/datasets/smbop/ed_smbop_full_beam_train_sim2.dat
# dev_dat=preprocessing/datasets/smbop/ed_smbop_full_beam_dev_sim2.dat
# test_dat=preprocessing/datasets/smbop/ed_smbop_beam_test_sim2.dat


# train_dat=preprocessing/datasets/natsql/ed_natsql_full_beam_train_sim2.dat
# dev_dat=preprocessing/datasets/natsql/ed_natsql_full_beam_dev_sim2.dat
# test_dat=preprocessing/datasets/natsql/ed_natsql_beam_test_sim2.dat


train_dat=preprocessing/datasets/resdnatsql/ed_resdnatsql_full_beam_train_sim2.dat
dev_dat=preprocessing/datasets/resdnatsql/ed_resdnatsql_full_beam_dev_sim2.dat
test_dat=preprocessing/datasets/resdnatsql/ed_resdnatsql_beam_test_sim2.dat

### In-domain data collection
# train_dat=preprocessing/datasets/natsql_beam_spider_tr_train_sim2.dat
# dev_dat=preprocessing/datasets/natsql_beam_spider_tr_dev_sim2.dat
# test_dat=preprocessing/datasets/ed_natsql_beam_test_sim2_tr.dat

### w/o graph simplification
# train_dat=preprocessing/datasets/ed_natsql_full_beam_train_ori.dat
# dev_dat=preprocessing/datasets/ed_natsql_full_beam_dev_ori.dat
# test_dat=preprocessing/datasets/ed_natsql_beam_test_ori.dat


name=${exp_group}/${expr_name}_${exp_num}
# name=${expr_name}_ron_test
mkdir -p ${dir}/experiments/$name
touch ${dir}/experiments/$name/log.txt
cat ${dir}/train_no_graph.sh >> ${dir}/experiments/$name/config

# nvidia-smi
# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=${cuda} \
python3 ${dir}/model.py \
	--expr_name ${name} \
	--batch_size ${batch} \
	--lr ${lr} \
	--dir=${dir} \
	--exp_name ${name} \
	--epoch ${epoch} \
	--warmup_factor ${warmup_factor} \
	--dev_dat ${dev_dat} \
	--train_dat ${train_dat} \
	--test_dat ${test_dat} \
	--gnn_model ${gnn_model} \
	--seed ${seed} \
	--use_beam \
	
