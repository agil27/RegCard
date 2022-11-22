# RegCard
Learned Cardinalities Estimation with Regularities

## How to use

1. Groundtruth and Featurize

    First we generate bitmaps for the train data, and cardinalities + bitmaps for our created test datasets for evaluate the monotonicity.

    ```bash
    python3 reg/data_gen.py -d data -f train -b
    python3 reg/data_gen.py -d workloads -f job-cmp-mini
    ```

    Alternatively we can also try `job-cmp-light` and `job-cmp`, which are much larger.
    
    ```bash
    python3 reg/data_gen.py -d workloads -f job-cmp-light
    python3 reg/data_gen.py -d workloads -f job-cmp
    ```

    This will also create the sampled materialized table view for each of the tables.

2. Training and Evaluation

    Then we run the training with the bitmaps. This part is credited to [Thomas Kipf, et al.](https://github.com/andreaskipf/learnedcardinalities)

    ```bash
    python3 train.py job-cmp-mini-card --cmp
    ```

    We added extra evaluation for the **monotonicity** by introducing the relative partial order labels in `job-cmp-mini-card-pairs.csv`. We will utilize it to calculate the obeying rate.

## Designs

1. Test dataset generation

    - We provided 3 versions of data: ...
    - We provide the compare pairs
    
2. Featurize and modeling

    bitmaps / model ...

3. Evaluate the monotonicity with MonoM score


## Results
We trained on the training data provided by Kipf, et al. for 100 epochs with 10000 queries of the training data, and then tested on our `job-cmp-light` dataset to the Q-score and MonoM score.

Train / Eval
```
Q-Error training set:
Median: 2.76472872178293
90th percentile: 17.0
95th percentile: 32.69356061737146
99th percentile: 115.25250000000005
Max: 1421.34375
Mean: 9.74698230501782

Q-Error validation set:
Median: 2.669427808000976
90th percentile: 17.00664223919475
95th percentile: 35.71871720512936
99th percentile: 150.92579164796643
Max: 18868.0
Mean: 45.83238834596302
```

```
Q-Error job-cmp-light-card:
Median: 2.259790362051729
90th percentile: 31.930000000000007
95th percentile: 123.63857710240028
99th percentile: 4012.0200000000013
Max: 7285.909090909091
Mean: 99.3319545865635

MonoM job-cmp-light-card:
Median: 1.0
90th percentile: 1.0
95th percentile: 1.0
99th percentile: 1.0
Max: 1
Mean: 0.7289156626506024
```

We can see that the Q-error is low but the MonoM score is not high, indicating a violation of monotonicity.

## TODOs

- [x] fix `python data_gen.py`, will throw error on 59it.
- [x] generate files in `multiprocess`, which would be faster utilizing 8 cores of the CPU
- [x] generate a list of queries for monotonicity evaluation, and generate a list of comparisons that look like `i > j` or `i = 0` to compare row $i$ and row $j$.
- [x] serialize / persist the table view used to generate bitmaps
