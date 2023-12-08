# python run.py --num_layers 4 --dropout_rate 0.3 > num_layer_4_dropout_rate_0.3.txt 2>&1
# python run.py --num_layers 4 --dropout_rate 0.8 > num_layer_4_dropout_rate_0.8.txt 2>&1
# python run.py --num_layers 2 --dropout_rate 0.3 > num_layer_2_dropout_rate_0.3.txt 2>&1
# python run.py --num_layers 2 --dropout_rate 0.5 > num_layer_2_dropout_rate_0.5.txt 2>&1
# python run.py --num_layers 2 --dropout_rate 0.8 > num_layer_2_dropout_rate_0.8.txt 2>&1
# python run.py  --lr 0.005 > num_layer_1_lr_0.005_SGD.txt 2>&1
# python run.py  --lr 0.05 > num_layer_1_lr_0.05_SGD.txt 2>&1
# python run.py  --lr 0.1 > num_layer_1_lr_0.1_SGD.txt 2>&1
# python run.py  --lr 0.2 > num_layer_1_lr_0.2_SGD.txt 2>&1
# python run.py  --lr 1 > num_layer_1_lr_1_SGD.txt 2>&1
# python run.py  --lr 0.8 > num_layer_1_lr_0.8_SGD.txt 2>&1
# python run.py  --lr 0.3 > num_layer_1_lr_0.3_SGD.txt 2>&1
# python run.py  --lr 0.4 > num_layer_1_lr_0.4_SGD.txt 2>&1
# python run.py  --lr 0.6 > num_layer_1_lr_0.6_SGD.txt 2>&1
# python run.py  --lr 0.7 > num_layer_1_lr_0.7_SGD.txt 2>&1
# python run.py  --lr 0.9 > num_layer_1_lr_0.9_SGD.txt 2>&1
# python run.py   --model_dims 256 > num_layer_1_model-dim_256.txt 2>&1
# python run.py   --model_dims 512 > num_layer_1_model-dim_512.txt 2>&1
python run.py --num_ff_layers 2 --num_layers 2 > num_layer_2_num_ff_layer2.txt 2>&1
python run.py --num_ff_layers 2 --num_layers 1 > num_layer_1_num_ff_layer2.txt 2>&1
python run.py --num_ff_layers 3 --num_layers 1 > num_layer_1_num_ff_layer3.txt 2>&1
python run.py --num_ff_layers 3 --num_layers 2 > num_layer_2_num_ff_layer3.txt 2>&1
python run.py --num_ff_layers 3 --num_layers 3 > num_layer_3_num_ff_layer3.txt 2>&1
python run.py --num_ff_layers 4 --num_layers 4 > num_layer_4_num_ff_layer4.txt 2>&1