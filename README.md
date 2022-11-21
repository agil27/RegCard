# RegCard
Learned Cardinalities Estimation with Regularities

## How to use

1. Groundtruth and Featurize

    First we generate bitmaps for the train data, and cardinalities + bitmaps for our created test datasets for evaluate the monotonicity.

    ```bash
    python3 reg/data_gen.py -d data -f train -b
    python3 reg/data_gen.py -d workloads -f job-cmp-mini
    ```

    This will also create the sampled materialized table view for each of the tables.

2. Training and Evaluation

    Then we run the training with the bitmaps. This part is credited to [Thomas Kipf, et al.](https://github.com/andreaskipf/learnedcardinalities)

    ```bash
    python3 train.py job-cmp-mini-card --cmp workloads/job-cmp-mini-card-pairs.csv
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

On `job-cmp-mini` dataset we created:

```
Q-Error training set:
Median: 9.035106910996845
90th percentile: 112.84683343500987
95th percentile: 245.0
99th percentile: 950.9500492483508
Max: 28807.0
Mean: 73.84785764410925

Q-Error validation set:
Median: 8.102225041004711
90th percentile: 94.80605604405748
95th percentile: 208.28043650561827
99th percentile: 1081.6639247096732
Max: 103861.0
Mean: 179.64540903871588

Loaded queries
Loaded bitmaps
Number of test samples: 186
Prediction time per test sample: 0.013295040335706486

Q-Error job-cmp-mini-card:
Median: 395.0625
90th percentile: 2046.3333333333333
95th percentile: 6141.75
99th percentile: 6176.3
Max: 6189.0
Mean: 1068.9164904381933

MonoM job-cmp-mini-card:
 Median: 1.0
 90th percentile: 1.0
 95th percentile: 1.0
 99th percentile: 1.0
 Max: 1
 Mean: 0.9470588235294117
```

## TODOs

- [x] fix `python data_gen.py`, will throw error on 59it.
- [ ] generate files in multithreads, which would be faster
- [x] generate a list of queries for monotonicity evaluation, and generate a list of comparisons that look like `i > j` or `i = 0` to compare row $i$ and row $j$.
- [x] serialize / persist the table view used to generate bitmaps
