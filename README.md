# Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Models
This is an implementation of the [`Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Models`](https://arxiv.org/abs/2209.06053), which is accpete by CIKM2022. The codes are developed with TensorFlow 1.4.

## Prepare data
You can get the data and process it using the script
```
sh prepare_data.sh
```
## Get the statistics of the data set
```
python script/cal_occurrence.py
```
## Experiments of the analysis of the one epoch phenomenon
```
python script/train.py --model_type DNN --epochs 10 
```
You can change the default parameters to get the results of different experiments:
* model structure
    * --model_type [DNN,LR]
* corruption percent, which varies from 0.0 (no corruption) to 1.0 (complete random labels).
    * --corruption_percent
* number of parameters
    * --embed_dim
    * --neuron
    * --nlayers
* batch size
    * --batch_size 
* activation function
    * --activation [dice,relu,prelu,sigmoid]
* optimizer
    * --optimizer [Adam,sgd,rmsprop]
* techniques to alleviate overfitting
    * --weight_decay
    * --dropout
    
## Experiments of the hypothesis
calculate the A-distance
```
python script/train_hypothesis.py --model_type DNN --epochs 3
```
obtain the parameter changes of embedding and MLP layers 
```
python script/train.py --model_type DNN --epochs 3 --print_grad 1
```

