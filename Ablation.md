### Ablation Studies

* Noise Multiplier -> NM
* Compression Rate -> CR
* Clip Bound -> CB
* Momentum -> MO
* \* -> diverge

`Attack Number = 10/100`
* FedAvg with CB=1.0 = 86.53 (same as Fed-SMP)

| CR or CB\Method      |Ours-NM1.0| Ours-NM1.4| Ours-NM2.0 | Ours-NM1.0-MO5|
| :---                 |  :---:   |  :---:    |  :---:     |        ---:   |
| CR=1.0 CB=1.0        |     *    |   *       |  40.78*    |               |
| CR=0.8 CB=1.0        |          |           |  52.60*    |               |
| CR=0.4 CB=1.0        |          |           |  63.52*    |               |
| CR=0.2 CB=1.0        |          |           |  75.83     |               |
| CR=0.1 CB=1.0        |          |           |  77.02     |               |
| CR=0.05 CB=1.0       |          |           |  74.97     |               |
| CR=0.01 CB=1.0       |          |           |  70.98     |               |
| CR=0.005 CB=1.0      |   72.86  |   *       |  63.20     |               |
| ---                  |   ---    |  ---      |     ---    |     ---       |
| CR=1.0 CB=0.5        |   83.08  |  81.07    |    74.55*  |     79.34     |
| CR=0.8 CB=0.5        |   83.03  |  81.58    |    69.90*  |               |
| CR=0.4 CB=0.5        |   82.54  |  81.36    |    78.51   |               |
| CR=0.2 CB=0.5        |   82.01  |  81.43    |    79.00   |               |
| CR=0.1 CB=0.5        |   80.95  |  80.31    |    79.04   |               |
| CR=0.05 CB=0.5       |   79.34  |  79.05    |    78.04   |               |
| CR=0.01 CB=0.5       |   75.59  |  75.11    |    73.74   |               |
| CR=0.005 CB=0.5      |   73.59  |  73.23    |    71.29   |               |

`Attack Number = 20/100`
* FedAvg with CB=1.0 =  (same as Fed-SMP)

| CR or CB\Method      |Ours-NM1.0| Ours-NM1.4| Ours-NM2.0 | Ours-NM1.0-MO5|
| :---                 |  :---:   |  :---:    |  :---:     |        ---:   |
| CR=1.0 CB=1.0        |         |          |       |               |
| CR=0.8 CB=1.0        |          |           |       |               |
| CR=0.4 CB=1.0        |          |           |       |               |
| CR=0.2 CB=1.0        |          |           |       |               |
| CR=0.1 CB=1.0        |          |           |       |               |
| CR=0.05 CB=1.0       |          |           |       |               |
| CR=0.01 CB=1.0       |          |           |       |               |
| CR=0.005 CB=1.0      |     |          |       |               |
| ---                  |   ---    |  ---      |     ---    |     ---       |
| CR=1.0 CB=0.5        |     |      |       |          |
| CR=0.8 CB=0.5        |     |      |       |               |
| CR=0.4 CB=0.5        |     |      |       |               |
| CR=0.2 CB=0.5        |     |      |       |               |
| CR=0.1 CB=0.5        |     |      |       |               |
| CR=0.05 CB=0.5       |     |      |       |               |
| CR=0.01 CB=0.5       |     |      |       |               |
| CR=0.005 CB=0.5      |     |      |       |               |

### Performance after adding defense method

`Ours Defend method on FMNIST against Fang_trmean attack`

`Attack Number = 10/100`

| HyperParams    | NM=1.4 & CB=0.5 |
|:---            |           ---:  |
| CR=1.0         |                 |
| CR=0.8         |   79.16         |
| CR=0.4         |   79.29         |
| CR=0.2         |   79.47         |
| CR=0.1         |   77.71         |
| CR=0.05        |   77.01         |
| CR=0.01        |   71.03         |
| CR=0.005       |   68.93         |

`Attack Number = 20/100`

| HyperParams    | NM=1.4 & CB=0.5 |
|:---            |           ---:  |
| CR=1.0         |                 |
| CR=0.8         |            |
| CR=0.4         |            |
| CR=0.2         |            |
| CR=0.1         |            |
| CR=0.05        |            |
| CR=0.01        |            |
| CR=0.005       |            |

`Defend on FMNIST`

`Attack Number = 10/100`

|Attack\Defense|   config   |    bulyan     |   median    | multi_krum |   tr_mean   |   flame   |   sparsefed  |dp<sup>2</sup>|Ours<sup>1</sup>| None |
| :---        |    :----:   |    :----:     |   :----:    |    :----:  |   :----:    |  :----:   |     :----:   |  :----:      | :---:    |     ---:   |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |  ----        | ---      |            |
|None |fedavg_image_fmnist_DEFENSE.yaml|    |             |            |             |           |    89.75     |  81.07       | 81.43    |            |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |  ----        | ---      |            |
|fang_trmean|fang_trmean_image_fmnist_DEFENSE.yaml|78.39  |76.74|74.62 |79.36/70.71  |   81.88   |    74.24     |  63.74*      | 79.47    |            |
|DBA  | dba_image_fmnist_DEFENSE.yaml |       |           |            |             |           |              |              |          |            |
|edge |edge_image_fmnist_DEFENSE.yaml |       |           |            |             |           |              |              |          |            |

1. We use following settings for comparison: NM=1.4, CB=0.5, CR=0.2.
2. We use following settings for comparison: NM=1.4, CB=0.5. 

`Attack Number = 15/100`

|Attack\Defense|   config   |    bulyan     |   median    | multi_krum |   tr_mean   |   flame   |   sparsefed  |dp<sup>2</sup>|Ours<sup>1</sup>| None |
| :---        |    :----:   |    :----:     |   :----:    |    :----:  |   :----:    |  :----:   |     :----:   |  :----:      | :---:    |     ---:   |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |  ----        | ---      |            |
|None |fedavg_image_fmnist_DEFENSE.yaml|    |             |            |             |           |              |              |          |            |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |  ----        | ---      |            |
|fang_trmean|fang_trmean_image_fmnist_DEFENSE.yaml|       |     |      |             |           |   38.91      |              |          |            |
|DBA  | dba_image_fmnist_DEFENSE.yaml |       |           |            |             |           |              |              |          |            |
|edge |edge_image_fmnist_DEFENSE.yaml |       |           |            |             |           |              |              |          |            |

1. We use following settings for comparison: NM=1.4, CB=0.5, CR=0.2.
2. We use following settings for comparison: NM=1.4, CB=0.5. 

`Attack Number = 20/100`

|Attack\Defense|   config   |    bulyan     |   median    | multi_krum |   tr_mean   |   flame   |   sparsefed  |dp<sup>2</sup>|Ours<sup>1</sup>| None |
| :---        |    :----:   |    :----:     |   :----:    |    :----:  |   :----:    |  :----:   |     :----:   |  :----:      | :---:    |     ---:   |
|  ---        |     ----    |     ----      |    ----     |     ----   |    ----     |   ----    |      ----    |  ----        | ---      |            |
|fang_trmean|fang_trmean_image_fmnist_DEFENSE.yaml|74.40|68.42| 75.09  | 63.45/58.35 |  75.12    |    15.19     |              | 74.06    |            |
|DBA  | dba_image_fmnist_DEFENSE.yaml |       |           |            |             |           |              |              |          |            |
|edge |edge_image_fmnist_DEFENSE.yaml |       |           |            |             |           |              |              |          |            |

0. Attack Number = 20/100
1. We use following settings for comparison: NM=1.4, CB=0.5, CR=0.2.
2. We use following settings for comparison: NM=1.4, CB=0.5. 