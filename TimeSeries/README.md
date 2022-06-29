### 1. 安装cpu版本的paddle
python -m pip install paddlepaddle==2.2.2 -i https://mirror.baidu.com/pypi/simple

### 2. 参数

------------------------------------  filename -----------------
checkpoints='./checkpoints/',
data_path='C:/ZhangLI/Codes/DataSet/kdd-cup/',
filename='sdwpf_baidukddcup2022_full.csv',
pred_file='./predict.py',
gpu=0,
is_debug=False,

capacity=134,  -- 134 turbines
turbine_id=0,  -- 涡轮机

------------------------------------- model -------------------
in_var=10,
out_var=1,
dropout=0.05,
lstm_layer=2,

------------------------------------- data ---------------------
input_len=144,
output_len=288,
task='MS',  -- ?
target='Patv', -- ?
start_col=3,
day_len=144,
train_size=153,
test_size=15,
total_size=184,

batch_size=32,
num_workers=5,

lr=0.0001,
lr_adjust='type1',
train_epochs=10,


patience=3,


use_gpu=False,
val_size=16
--------------------------------------------------------------  predict -----------------------
预测数据的格式
1. 小于0的值为0
2. 空值
3. 异常值 Ndir is [-720°, 720°] Wdir is [-180°, 180°]
