import pickle
import os
from train import train_and_predict


def load_counter():
    if not os.path.isfile('logs/grid_search/counter.pkl'):
        return 0

    with open('logs/grid_search/counter.pkl', 'rb') as fin:
        counter = pickle.load(fin)
    return counter


def save_counter(counter):
    with open('logs/grid_search/counter.pkl', 'wb') as fout:
        pickle.dump(counter, fout)


def wrapper_train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda, cmp,
    lbda, regbatch, dist, soften, log_dir):

    cmd = "python3 train.py {} --queries {} --epochs {} --batch {} --hid {} --cuda --cmp --lbda {} --log {}".format(
        workload_name, num_queries, num_epochs, batch_size, hid_units, lbda, log_dir
    )

    if lbda != 0.0:
        cmd += " --regbatch {} --dist {} --soften {}".format(regbatch, dist, soften)
    
    print(cmd)


if __name__ == "__main__":

    queries = 50000
    epochs = 50
    batch = 1024
    regbatch = 1024

    hids = [128, 256, 512]
    lbdas = [0, 0.1, 0.5, 1, 3, 10]
    dists = ['jaccard', 'diff']
    softens = [10, 100, 1000, 10000]

    log = 'grid_search'
    testset = 'job-cmp-card'

    os.system('mkdir -p logs/grid_search')

    counter = load_counter()

    curr_counter = 0
    for hid in hids:
        for lbda in lbdas:

            if lbda == 0.0:
                if curr_counter >= counter:
                    wrapper_train_and_predict(testset, queries, epochs, batch, hid, True, True,
                        lbda, None, None, None, log)
                curr_counter += 1
                if curr_counter > counter:
                    save_counter(curr_counter)

            else:
                for dist in dists:

                    for soften in softens:
                        if curr_counter >= counter:
                            wrapper_train_and_predict(testset, queries, epochs, batch, hid, True, True,
                                lbda, regbatch, dist, soften, log)
                        curr_counter += 1
                        if curr_counter > counter:
                            save_counter(curr_counter)

