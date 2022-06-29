# Solution to the Debiasing Track of KDD CUP 2020 (Team Rush)

We propose a CF + GNN recall approach, as well as an ensemble ranking approach of GBDT and DIN.
Please refer to the summary of our solution in my blog or zhihu, 

- http://xtf615.com/2020/06/17/KDD-CUP-2020-Debiasing-Rush/
- https://zhuanlan.zhihu.com/p/149061129


## Final Results

Track B,
- ndcg-full: **rank 3rd**（7.277)
- ndcg-half: **rank 10th** (7.226)

## Introduction 

https://tianchi.aliyun.com/competition/entrance/231785/information

## Data

Downloading Link: https://pan.baidu.com/s/1P2h7APGUEscdnGlYQpgh-g  password: kilm

Just download the data and override the data directory

Data Fold format
```
|-- data
	|-- underexpose_train
		|-- underexpose_item_feat.csv
		|-- underexpose_train_click-0.csv
		|-- underexpose_train_click-1.csv
		|-- ...
		|-- underexpose_train_click-9.csv
		
	|-- underexpose_test
		|-- underexpose_test_click-0
			|-- underexpose_test_qtime-0.csv
			|-- underexpose_test_click-0.csv
		|-- underexpose_test_click-1
			|-- underexpose_test_qtime-1.csv
			|-- underexpose_test_click-1.csv
		|-- ...
		|-- underexpose_test_click-9
			|-- underexpose_test_qtime-9.csv
			|-- underexpose_test_click-9.csv
```

## Key Points of Our Solution 
- **The construction of the recall training set**, how to use the ENTIRE data for training is important. we need to avoid data crossing from both the user side and the item side. This improvement is very significant, indicating that the data have a great impact on the results.

- **Improvements in CF methods** can effectively debias the data, including interaction time, direction, content similarity, item popularity, user activity, etc. This improvement is also very significant and suits the topic of the game, i.e., Debiasing.

- **SR-GNN** is designed for sequential recommendation based on the graph neural networks, which perfectly suits the scenario of this competition. SR-GNN captures the high-order proximity between items and takes into account the user's long-term and short-term preferences. In addition, we improve SR-GNN in many ways, i.e., we use content features for embedding initialization, introducing node weights according to frequency (for debiasing), position embedding (enhance short-term interaction importance), embedding L2 normalization, residual connection, the sequence-level embedding construction. The improvement of the SR-GNN reaches more than 0.05+ in terms of full-ndcg.

- **The coarse-grained ranking** considers the frequency of items and motivate the exposure of low-frequency items to effectively debias, which significantly improves the half metrics.

- **Construction of ranking features**, including recall features, content features, historical behavior-related features, ID features, etc.

- **Ensemble ranking model**: the fusion of Tree-based model, i.e., GBDT and Deep-Learning-based model, i.e., DIN can effectively improve the ranking results

## Components of the solution
- Recall
    - Item-CF
    - User-CF
    - Swing
    - Bi-Graph
    - SR-GNN
    
- Ranking
    - GBDT
    - DIN

## Code Preview
Our code is well organized. The code package is organized as follows:

```
|-- code
    |-- process/
        |-- feat_process.py  (item feature processing, read/pre-process/fill in missing items)
        |-- load_data.py (load original data or training data for recall)
        |-- recommend_process.py  (generate recommendation results. item-based/user-based/re-rank)
        |-- convert_data.py  (tools of converting data format)
        |-- tr_val_split.py (train/validation split)
		
    |-- recall/
        |-- item_cf.py
        |-- swing.py
        |-- bi_graph.py
        |-- user_cf.py
        |-- cb.py (the content-based similarity computing module using faiss)
        |-- do_recall.py (wrapp the multi-source recall procedure, single processor)
        |-- do_recall_multi_processing.py (wrap the multi-source recall procedure, multiple processors)
        |-- sr_gnn/    (sr_gnn recall package)
            |-- construct_sr_gnn_train_data.py (construct sr-gnn training data, get train/validate/test sequences / enhance data)
            |-- read_sr_gnn_results.py (read sr-gnn results from the trained sr-gnn model)
            |-- lib/   (sr-gnn training/recommending library)
		        
    |-- rank/
        |-- construct_ranking_data.py (construct ranking data for DIN and GBDT)
        |-- organize_ranking_data_label.py (obtain data label, e.g., negative-sampling)
        |-- organize_ranking_data_info_feat.py (extract info feat)
        |-- organize_ranking_data_recall_feat.py (extract recall feat)
        |-- din.py (DIN model)
        |-- lgb.py (GBDT model)
        |-- ensemble.py (ensemble of din and gbdt)
        
    |-- conf.py (setup the global settings, e.g., mode (online/offline), data directory (online_data/offline_data))
    |-- global_variables.py (global variables shared across multi-modules)
    |-- recall_main.py (wrapper of the multi-source recall pipeline)
    |-- rank_main.py (wrapper of the ranking pipeline)
    |-- sr_gnn_main.py (wrapper of the sr-gnn recall pipeline)
```

## Environment
python 3.6, tensorflow 1.14.0

## Running
- pipeline: ```./run.sh```

If you just want to run some parts of our solution: 
- sr-gnn recall (single model recall): ```python3 code/sr_gnn_main.py```
- recall pipeline (CF + read SR-GNN): ```python3 code/recall_main.py```
- ranking pipeline (GBDT + DIN): ```python3 code/rank_main.py``` 

Or if you want to run step by step in Notebook:
- notebook/Rush_0615.ipynb

Note:
The recall code and sr-gnn code are well checked. you can run both in py or notebook.

~~The ranking code is not well checked now, we will double-check the code as soon as possible.~~

The ranking code is checked now, you can run both in py or notebook. (2020.06.19)

## Evaluation

We do the offline evaluation for each phase respectively. That is, we use the phase-specific data to do recall/rank for each phase, while we submit the online results using the full click data of all phases.
In this way, we have observed that there is almost a FIXED gap of ~0.08 between online ndcg-full and offline ndcg-full, ~0.02 between online ndcg-half and offline ndcg-half (we eval and judge in this way). 
If you use the full click to do offline evaluation, you need to delete the click data occurring in offline answer files from train/test data to avoid leak (we don't eval in this way).

A offline evaluation example to show the effectiveness of SR-GNN：
- Full CF, recall evaluation:
```
current_phase: 9
phase_id=7, score=[0.05876107 0.06376237 0.133125   0.12718205]
phase_id=8, score=[0.05402379 0.0609233  0.11375    0.11425062]
phase_id=9, score=[0.05611179 0.06346961 0.126875   0.13161132]
score=7.168896675109863,
hitrate_50_full=7.373749732971191,
ndcg_50_full=7.168896675109863,
hitrate_50_half=7.373044013977051,
ndcg_50_half=7.188154697418213
```

- SR-GNN, recall evaluation (Better than CF in terms of hitrate-full metrics):
```
current_phase: 9
phase_id=7, score=[0.05882552 0.04997362 0.140625   0.11471321]
phase_id=8, score=[0.05348391 0.05319532 0.130625   0.11547912]
phase_id=9, score=[0.05377048 0.0488597  0.134375   0.11562116]
score=7.166079998016357,
hitrate_50_full=7.405624866485596,
ndcg_50_full=7.166079998016357,
hitrate_50_half=7.345813274383545,
ndcg_50_half=7.152028560638428
```

- Full CF + SR-GNN, recall evaluation:
```
phase_id=7, score=[0.06840383 0.06921597 0.16125    0.15336658]
phase_id=8, score=[0.0635776  0.06901148 0.145      0.14250614]
phase_id=9, score=[0.06315026 0.06603911 0.155      0.15252152]
score=7.195131778717041,
hitrate_50_full=7.461250305175781,
ndcg_50_full=7.195131778717041,
hitrate_50_half=7.448394298553467,
ndcg_50_half=7.204266548156738
```

From the above evaluation, we can see the significant effectiveness of SR-GNN. In fact,
The online submission of single SR-GNN result outperforms Full-CF (i.e., the full-click data enhance SR-GNN more than Full-CF).

You can reproduce these results by checkout the 'offline' branch by ```git checkout offline```
and run ```python3 code/sr_gnn_main.py``` and ```python3 code/recall_main.py```  in sequence.

 
## References

[1]  Wu S, Tang Y, Zhu Y, et al. Session-based recommendation with graph neural networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 346-353.

[2]  Gupta P, Garg D, Malhotra P, et al. NISER: Normalized Item and Session Representations with Graph Neural Networks[J]. arXiv preprint arXiv:1909.04276, 2019.

[3]  Zhou T, Ren J, Medo M, et al. Bipartite network projection and personal recommendation[J]. Physical review E, 2007, 76(4): 046115.

[4] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018: 1059-1068.

[5] Ke G, Meng Q, Finley T, et al. Lightgbm: A highly efficient gradient boosting decision tree[C]//Advances in neural information processing systems. 2017: 3146-3154.

[6] DeepCTR, Easy-to-use,Modular and Extendible package of deep-learning based CTR models, https://github.com/shenweichen/DeepCTR

[7] A simple itemCF Baseline, score:0.1169(phase0-2), https://tianchi.aliyun.com/forum/postDetail?postId=103530

[8] 改进青禹小生baseline，phase3线上0.2, https://tianchi.aliyun.com/forum/postDetail?postId=105787

[9] 推荐系统算法调研, http://xtf615.com/2018/05/03/recommender-system-survey/

[10] A Simple Recall Method based on Network-based Inference, score:0.18 (phase0-3), https://tianchi.aliyun.com/forum/postDetail?postId=104936

[11] A library for efficient similarity search and clustering of dense vectors, https://github.com/facebookresearch/faiss

[12] CIKM 2019 tutorial: Learning and Reasoning on Graph for Recommendation, https://next-nus.github.io/

[13] Source code and datasets for the paper "Session-based Recommendation with Graph Neural Networks" (AAAI-19), https://github.com/CRIPAC-DIG/SR-GNN

[14] 2019-CIKM挑战赛，超大规模推荐之用户兴趣高效检索赛道, 冠军解决方案: https://github.com/ChuanyuXue/CIKM-2019-AnalytiCup



## 方案
1. item-cf
1.1 聚合用户组的点击item
1.2 计算单用户组item的相似度
1.3 累计每个用户组item的相似度得到所有item的相似度

