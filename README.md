# FedLearning
Federated learning with DP, sparsification, RFL, Attack &amp; Defense, etc. of TNI lab use.
    
# environment: envfl.yml
# Datasets:
- Fashion-MNIST is a relatively small dataset, use it for testing and debugging your algorithm will save you time a lot. 

# Use tensorboard for training logging

Using VSCode, it will automatically map the port and visit from local browser.

or

Tricks for remote accessing Tensorboard

(1) Open local terminal and type following command

```
ssh -L 16006:127.0.0.1:6006 username@server_address
```

16006 is a local port to map the server ip and port
6006 is a port for tensorboard, change it if you use another specific port for tensorboard

(2) After you type password and log into the remote server with ssh, go to the path for tensorboard log, and type the command

```
tensorboard --logdir='.'
```

it will show: TensorBoard 2.12.2 at http://localhost:6006/ (Press CTRL+C to quit)

(3) Open the local browser, and visit this link

```
http://127.0.0.1:16006/
```

# Use 'screen' to keep your remote processes running

```
sudo apt install screen
```

Useful Command Example:

(1) Start a new window

```
screen
```

(2) Start a new window with a name

```
screen -S EXPERIMENT_NAME
```

(3) Display the currently opened screens

```
screen -ls
```

(4) When you are in a window, you want to leave the window

```
screen -d or Ctrl-a + d
```

(5) When you outside a window, and you want to reattach a screen window

Firstly, get the window id or window name with `screen -ls`

Then

```
screen -r WINDOW_ID/WINDOW_NAME
```

# Use MMEngine to set up configurations

```

pip install mmengine

```

For detailed usages, please refer to this [link](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html)

# Use Optuna for Hyperparameter Search 

```

pip install optuna

```

For detailed usages, please refer to this [link](optuna.readthedocs.io), and this [video](https://youtu.be/P6NwZVl8ttc)

# Benchmark

`Defend on FMNIST`<sup>0</sup>

|Attack\Defense|   config   |    bulyan     |   median    | multi_krum |   tr_mean   |   flame   |   sparsefed  |dp<sup>2</sup>|Ours<sup>1</sup>| None |
| :---        |    :----:   |    :----:     |   :----:    |    :----:  |   :----:    |  :----:   |     :----:   |  :----:      | :---:    |     ---:   |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |  ----        | ---      |            |
|fang_trmean|fang_trmean_image_fmnist_DEFENSE.yaml|74.40|68.42| 75.09  | 63.45/58.35 |  75.12    |    15.19     |              | 74.06    |            |
|DBA  | dba_image_fmnist_DEFENSE.yaml |       |           |            |             |           |              |              |          |            |
|edge |edge_image_fmnist_DEFENSE.yaml |       |           |            |             |           |              |              |          |            |

0. Attack Number = 20/100
1. We use following settings for comparison: NM=1.4, CB=0.5, CR=0.2.
2. We use following settings for comparison: NM=1.4, CB=0.5. 

`Defend on SVHN`

| Method      |    config   |    bulyan     |   median    | multi_krum |   tr_mean   |   flame   |   sparsefed  |     Ours     | 
| :---        |    :----:   |    :----:     |   :----:    |    :----:  |   :----:    |  :----:   |     :----:   |     ---:     |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |     ---      |
|None |fedavg_image_svhn_DEFENSE.yaml|      |             |            |             |           |              |              |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |     ---      |
|fang_trmean|fang_trmean_image_svhn_DEFENSE.yaml|         |     |      |             |           |              |              | 
|fang_median|fang_median_image_svhn_DEFENSE.yaml|         |     |      |             |           |              |              |
| DBA | dba_image_svhn_DEFENSE.yaml |       |             |            |             |           |              |              |  
| edge|edge_image_svhn_DEFENSE.yaml |       |             |            |             |           |              |              |  

<!-- `Defend on Cifar10`

| Method      |    config   |    bulyan     |   median    | multi_krum |   tr_mean   |   flame   |   sparsefed  |     Ours     | 
| :---        |    :----:   |    :----:     |   :----:    |    :----:  |   :----:    |  :----:   |     :----:   |     ---:     |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |     ---      |
|None |fedavg_image_cifar_DEFENSE.yaml|     |             |            |             |           |              |              |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |     ---      |
|fang_trmean|fang_trmean_image_cifar_DEFENSE.yaml|       |     |       |             |           |              |              | 
|fang_median|fang_median_image_cifar_DEFENSE.yaml|       |     |       |             |           |              |              |
| DBA | dba_image_cifar_DEFENSE.yaml |      |             |            |             |           |              |              | 
| edge|edge_image_cifar_DEFENSE.yaml |      |             |            |             |           |              |              |    -->

`Defend on Reddit`

| Method      |    config   |    bulyan     |   median    | multi_krum |   tr_mean   |   flame   |   sparsefed  |     Ours     | 
| :---        |    :----:   |    :----:     |   :----:    |    :----:  |   :----:    |  :----:   |     :----:   |     ---:     |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |     ---      |
|None |fedavg_image_reddit_DEFENSE.yaml|    |             |            |             |           |              |              |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |     ---      |
|fang_trmean|fang_trmean_image_reddit_DEFENSE.yaml|      |     |       |             |           |              |              | 
|fang_median|fang_median_image_reddit_DEFENSE.yaml|      |     |       |             |           |              |              |
| DBA | dba_image_reddit_DEFENSE.yaml |     |             |            |             |           |              |              | 
| edge|edge_image_reddit_DEFENSE.yaml |     |             |            |             |           |              |              |  

`Defend on Shakespeare`

| Method      |    config   |    bulyan     |   median     |  multi_krum |    tr_mean   |   flame   |   sparsefed  |     Ours     | 
| :---        |    :----:   |    :----:     |   :----:     |     :----:  |    :----:    |  :----:   |     :----:   |     ---:     |
|  ---        |     ----    |     ----      |    ----      |     ----    |    ----      |   ----    |      ----    |     ---      |
|None |fedavg_image_shakespeare_DEFENSE.yaml|      |       |             |              |           |              |              |
|  ---        |     ----    |     ----      |    ----      |     ----    |    ----      |   ----    |      ----    |     ---      |
|fang_trmean|fang_trmean_image_shakespeare_DEFENSE.yaml|      |     |     |             |           |              |              | 
|fang_median|fang_median_image_shakespeare_DEFENSE.yaml|      |     |     |             |           |              |              |
| DBA | dba_image_shakespeare_DEFENSE.yaml |     |         |             |              |           |              |              | 
| edge|edge_image_shakespeare_DEFENSE.yaml |     |         |             |              |           |              |              |  

# Acknowledges

We express our gratitude to the contributors of the following excellent open-source projects:

- [NDSS21-Model-Poisoning](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)
- [DBA](https://github.com/AI-secure/DBA)
- [backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [OOD_Federated_Learning](https://github.com/ksreenivasan/OOD_Federated_Learning)
- [FLAME](https://github.com/zhmzm/FLAME)
- [sparsefed](https://github.com/sparsefed/sparsefed)
