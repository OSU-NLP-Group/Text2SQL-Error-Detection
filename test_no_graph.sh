cuda=0

dir=DIR/ED4SP # for use on strawberry1
batch=16
lr=3e-5
epoch=10
warmup_factor=10
gnn_model=none


seed=101

train_dat=dummy.dat # doesn't matter for testing
dev_dat=dummy.dat # doesn't matter for testing



exp_group=SmBop
# exp_group=NatSQL
# exp_group=Bridge
# exp_group=ResdNatSQL

expr_name=CodeBERT



# test_dat=preprocessing/datasets/ed_smbop_beam_test_sim2.dat
# test_dat=preprocessing/datasets/ed_smbop_beam_test_sim2_ntsq.dat
# test_dat=preprocessing/datasets/ed_smbop_beam_test_sim2_brg.dat
# test_dat=preprocessing/datasets/ed_smbop_full_beam_dev_sim2.dat
# test_dat=preprocessing/datasets/ed_smbop_full_beam_dev_sim2_brg.dat
# test_dat=preprocessing/datasets/ed_smbop_full_beam_dev_sim2_ntsq.dat

# test_dat=preprocessing/datasets/ed_smbop_beam_test_ori.dat
# test_dat=preprocessing/datasets/ed_smbop_beam_test_ori_brg.dat
# test_dat=preprocessing/datasets/ed_smbop_beam_test_ori_ntsq.dat
# test_dat=preprocessing/datasets/ed_smbop_full_beam_dev_ori.dat
# test_dat=preprocessing/datasets/ed_smbop_full_beam_dev_ori_brg.dat
# test_dat=preprocessing/datasets/ed_smbop_full_beam_dev_ori_ntsq.dat

# test_dat=preprocessing/datasets/ed_bridge_beam_test_sim2.dat
# test_dat=preprocessing/datasets/ed_bridge_beam_test_sim2_smbp.dat
# test_dat=preprocessing/datasets/ed_bridge_beam_test_sim2_ntsq.dat
# test_dat=preprocessing/datasets/ed_bridge_full_beam_dev_sim2.dat
# test_dat=preprocessing/datasets/ed_bridge_full_beam_dev_sim2_smbp.dat
# test_dat=preprocessing/datasets/ed_bridge_full_beam_dev_sim2_ntsq.dat
# test_dat=preprocessing/datasets/ed_bridge_beam_test_ori.dat
# test_dat=preprocessing/datasets/ed_bridge_beam_test_ori_smbp.dat
# test_dat=preprocessing/datasets/ed_bridge_beam_test_ori_ntsq.dat
# test_dat=preprocessing/datasets/ed_bridge_full_beam_dev_ori.dat
# test_dat=preprocessing/datasets/ed_bridge_full_beam_dev_ori_smbp.dat
# test_dat=preprocessing/datasets/ed_bridge_full_beam_dev_ori_ntsq.dat


# test_dat=preprocessing/datasets/ed_natsql_beam_test_sim2.dat
# test_dat=preprocessing/datasets/ed_natsql_beam_test_sim2_smbp.dat
# test_dat=preprocessing/datasets/ed_natsql_beam_test_sim2_brg.dat
# test_dat=preprocessing/datasets/ed_natsql_full_beam_dev_sim2.dat
# test_dat=preprocessing/datasets/ed_natsql_full_beam_dev_sim2_smbp.dat
# test_dat=preprocessing/datasets/ed_natsql_full_beam_dev_sim2_brg.dat
# test_dat=preprocessing/datasets/ed_natsql_beam_test_ori.dat
# test_dat=preprocessing/datasets/ed_natsql_beam_test_ori_smbp.dat
# test_dat=preprocessing/datasets/ed_natsql_beam_test_ori_brg.dat
# test_dat=preprocessing/datasets/ed_natsql_full_beam_dev_ori.dat
# test_dat=preprocessing/datasets/ed_natsql_full_beam_dev_ori_smbp.dat
# test_dat=preprocessing/datasets/ed_natsql_full_beam_dev_ori_brg.dat
# test_dat=preprocessing/datasets/ed_bridge_beam_kaggle_sim2.dat

test_dat=preprocessing/datasets/natsql/ed_natsql_beam_test_sim2.dat
# test_dat=preprocessing/datasets/resdnatsql/ed_resdnatsql_beam_test_sim2_ntsq.dat
# test_dat=preprocessing/datasets/resdnatsql/ed_resdnatsql_beam_test_sim2_smbp.dat
# test_dat=preprocessing/datasets/smbop/ed_smbop_beam_test_sim2_rsdntsq.dat
# test_dat=preprocessing/datasets/natsql/ed_natsql_beam_test_sim2_rsdntsq.dat

for exp_num in 1 2 
do
name=${exp_group}/${expr_name}_${exp_num}
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
	--test \
	--use_beam \
	--seed ${seed}
done