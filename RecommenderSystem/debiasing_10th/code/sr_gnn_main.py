"""
Author:
    Xinzhou Dong, Taofeng Xue
References:
[1] Wu S, Tang Y, Zhu Y, et al. Session-based recommendation with graph neural networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 346-353.
[2] Gupta P, Garg D, Malhotra P, et al. NISER: Normalized Item and Session Representations with Graph Neural Networks[J]. arXiv preprint arXiv:1909.04276, 2019.
"""

from code.recall import *
import re


def find_checkpoint_path(phase, checkpoint_prefix='session_id', version='v2'):
    checkpoint_dir = 'tmp/model_saved/{}/{}/{}'.format(version, mode, phase)
    step_max = 0
    re_cp = re.compile("{}-(\d+)\.".format(checkpoint_prefix))
    for file in os.listdir(checkpoint_dir):
        so = re_cp.search(file)
        if so:
            step = int(so.group(1))
            step_max = step if step > step_max else step_max
    checkpoint_path = '{}/{}-{}'.format(checkpoint_dir, checkpoint_prefix, step_max)
    print('CheckPoint: {}'.format(checkpoint_path))
    return checkpoint_path


def sr_nn_version_1(phase, item_cnt):
    print('version 1 start...')
    model_path = 'tmp/model_saved/v1/{}/{}'.format(mode, phase)
    if not os.path.exists(model_path): os.makedirs(model_path)

    file_path = '{}/{}/data'.format(sr_gnn_root_dir, phase)
    sr_gnn_lib_path = 'code/recall/sr_gnn/lib'
    if os.path.exists(model_path):
        print('model_path={} exists, delete'.format(model_path))
        os.system("rm -rf {}".format(model_path))
    os.system("python3 {sr_gnn_lib_path}/main.py --task train --node_count {item_cnt} "
              "--checkpoint_path {model_path}/session_id --train_input {file_path}/train_item_seq_enhanced.txt "
              "--test_input {file_path}/test_item_seq.txt --gru_step 2 --epochs 10 "
              "--lr 0.001 --lr_dc 2 --dc_rate 0.1 --early_stop_epoch 3 "
              "--hidden_size 256 --batch_size 256 --max_len 20 --has_uid True "
              "--feature_init {file_path}/item_embed_mat.npy --sigma 8 ".format(sr_gnn_lib_path=sr_gnn_lib_path,
                                                                                item_cnt=item_cnt,
                                                                                model_path=model_path,
                                                                                file_path=file_path))
    # generate rec
    checkpoint_path = find_checkpoint_path(phase, version='v1')
    prefix = 'standard_'

    rec_path = '{}/{}rec.txt'.format(file_path, prefix)
    os.system("python3 {sr_gnn_lib_path}/main.py --task recommend --node_count {item_cnt} "
              "--checkpoint_path {checkpoint_path} --item_lookup {file_path}/item_lookup.txt "
              "--recommend_output {rec_path} --session_input {file_path}/test_user_sess.txt "
              "--gru_step 2 --hidden_size 256 --batch_size 256 --rec_extra_count 50 --has_uid True "
              "--feature_init {file_path}/item_embed_mat.npy "
              "--max_len 10 --sigma 8".format(sr_gnn_lib_path=sr_gnn_lib_path,
                                              item_cnt=item_cnt, checkpoint_path=checkpoint_path,
                                              file_path=file_path, rec_path=rec_path))


def sr_nn_version_2(phase, item_cnt):
    '''
    version 2: the improved SR-GNN model
    :param phase: target_phase
    :param item_cnt: item node number
    :return: output recommendation results to file
    '''
    print('version 2 start...')

    model_path = 'tmp/model_saved/v2/{}/{}'.format(mode, phase)
    if not os.path.exists(model_path): os.makedirs(model_path)

    file_path = '{}/{}/data'.format(sr_gnn_root_dir, phase)
    sr_gnn_lib_path = 'code/recall/sr_gnn/lib'
    if os.path.exists(model_path):
        print('model_path={} exists, delete'.format(model_path))
        os.system("rm -rf {}".format(model_path))
    os.system("python3 {sr_gnn_lib_path}/main.py --task train --node_count {item_cnt} "
              "--checkpoint_path {model_path}/session_id --train_input {file_path}/train_item_seq_enhanced.txt "
              "--test_input {file_path}/test_item_seq.txt --gru_step 2 --epochs 10 "
              "--lr 0.001 --lr_dc 2 --dc_rate 0.1 --early_stop_epoch 3 --hidden_size 256 --batch_size 256 "
              "--max_len 20 --has_uid True --feature_init {file_path}/item_embed_mat.npy --sigma 10 "
              "--sq_max_len 5 --node_weight True  --node_weight_trainable True".format(
                                                        sr_gnn_lib_path=sr_gnn_lib_path,
                                                        item_cnt=item_cnt,
                                                        model_path=model_path,
                                                        file_path=file_path))
    # generate rec
    checkpoint_path = find_checkpoint_path(phase, version='v2')
    prefix = 'pos_node_weight_'

    rec_path = '{}/{}rec.txt'.format(file_path, prefix)
    sr_gnn_lib_path = 'code/recall/sr_gnn/lib'
    os.system("python3 {sr_gnn_lib_path}/main.py --task recommend --node_count {item_cnt} "
              "--checkpoint_path {checkpoint_path} --item_lookup {file_path}/item_lookup.txt "
              "--recommend_output {rec_path} --session_input {file_path}/test_user_sess.txt --gru_step 2 "
              "--hidden_size 256 --batch_size 256 --rec_extra_count 50 --has_uid True "
              "--feature_init {file_path}/item_embed_mat.npy "
              "--max_len 10 --sigma 10 --sq_max_len 5 --node_weight True "
              "--node_weight_trainable True".format(sr_gnn_lib_path=sr_gnn_lib_path,
                                                    item_cnt=item_cnt, checkpoint_path=checkpoint_path,
                                                    file_path=file_path, rec_path=rec_path, ))


if __name__ == '__main__':
    processed_item_feat_df, item_content_vec_dict = obtain_entire_item_feat_df()
    phase_item_cnt_dict = {}  # 7: 45194, 8: 44979, 9: 44365
    for phase in range(start_phase, now_phase+1):
        item_cnt = construct_sr_gnn_train_data(phase, item_content_vec_dict, is_use_whole_click=True)
        phase_item_cnt_dict[phase] = item_cnt
    print('construct train data done...')

    # running model
    for phase in range(start_phase, now_phase+1):
        print('phase={}'.format(phase))
        sr_nn_version_1(phase, phase_item_cnt_dict[phase])
        sr_nn_version_2(phase, phase_item_cnt_dict[phase])
