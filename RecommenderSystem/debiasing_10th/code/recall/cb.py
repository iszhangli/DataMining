"""
Author:
    Taofeng Xue, xuetfchn@foxmail.com, xtf615.com
Reference:
    Faiss, A library for efficient similarity search and clustering of dense vectors: https://github.com/facebookresearch/faiss
"""

import collections
import pickle
import faiss
import numpy as np
from ..conf import *


def get_content_sim_item(item_feat_df, topk=100, is_use_filled_feat=False, is_load_from_file=True):
    if not is_use_filled_feat:
        sim_path = os.path.join(user_data_dir, 'item_content_sim_dict.pkl')
    else:
        sim_path = os.path.join(user_data_dir, 'item_content_sim_dict_fill.pkl')

    if is_load_from_file and os.path.exists(sim_path):
        with open(sim_path, 'rb') as f:
            return pickle.load(f)
    print('begin compute similarity using faiss...')

    # {0: 42844} => dataframe.index: item_id  n_dim = 108916
    item_idx_2_rawid_dict = dict(zip(item_feat_df.index, item_feat_df['item_id']))
    txt_item_feat_df = item_feat_df.filter(regex="txt*")  # filter txt vec feature
    img_item_feat_df = item_feat_df.filter(regex="img*")  # filter img vec feature

    txt_item_feat_np = np.ascontiguousarray(txt_item_feat_df.values, dtype=np.float32)  # Return a contiguous array (ndim >= 1) in memory (C order)
    img_item_feat_np = np.ascontiguousarray(img_item_feat_df.values, dtype=np.float32)

    # norm
    txt_item_feat_np = txt_item_feat_np / np.linalg.norm(txt_item_feat_np, axis=1, keepdims=True)
    img_item_feat_np = img_item_feat_np / np.linalg.norm(img_item_feat_np, axis=1, keepdims=True)

    txt_index = faiss.IndexFlatIP(128)  # 向量内积计算相似度， 这里的index是numpy的index，需要通过字典item_idx_2_rawid_dict 找到对应关系
    txt_index.add(txt_item_feat_np)

    img_index = faiss.IndexFlatIP(128)
    img_index.add(img_item_feat_np)

    item_sim_dict = collections.defaultdict(dict)

    def search(feat_index, feat_np):
        sim, idx = feat_index.search(feat_np, topk)  # sim 计算最相近的相似的值 idex-top个相似度的item值
        for target_idx, sim_value_list, rele_idx_list in zip(range(len(feat_np)), sim, idx):
            target_raw_id = item_idx_2_rawid_dict[target_idx]  #  得到实际的item值的坐标
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                rele_raw_id = item_idx_2_rawid_dict[rele_idx]
                item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                     0) + sim_value

    search(txt_index, txt_item_feat_np)
    search(img_index, img_item_feat_np)

    if is_load_from_file:
        with open(sim_path, 'wb') as f:
            pickle.dump(item_sim_dict, f)

    return item_sim_dict

# 对于索引的处理
