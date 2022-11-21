# RegCard
Learned Cardinalities Estimation with regularities

## How to use

1. Groundtruth and Featurize

    First we generate cardinalities and bitmaps for our created test datasets for evaluate the monotonicity.

    ```bash
    python3 reg/data_gen.py -d workloads -f job-cmp-mini
    ```

    This will also create the sampled materialized table view for each of the tables.

2. Training and Evaluation

    Then we run the training with the bitmaps. This part is credited to [Thomas Kipf, et al.](https://github.com/andreaskipf/learnedcardinalities)

    ```bash
    python3 train.py job-cmp-mini-card
    ```

    We added extra evaluation for the **monotonicity** by introducing the relative partial order labels in `job-cmp-mini-card-pairs.csv`. We will utilize it to calculate the obeying rate.

# TODOs

- [x] fix `python data_gen.py`, will throw error on 59it.
- [ ] generate files in multithreads, which would be faster
- [x] generate a list of queries for monotonicity evaluation, and generate a list of comparisons that look like `i > j` or `i = 0` to compare row $i$ and row $j$.
- [x] serialize / persist the table view used to generate bitmaps
