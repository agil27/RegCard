import argparse
import time
import os
import random
import numpy as np
import json
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, load_data, make_dataset, load_monotonic_regularization
from mscn.model import SetConv
from reg.cmp import print_monom

from sklearn.metrics import mean_squared_error


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def jaccard_distance(range1, range2):
    if type(range1) == tuple and type(range2) == tuple:
        # we know one range is no less than the other
        lo1, hi1 = range1
        lo2, hi2 = range2
        size1 = hi1 - lo1 + 1
        size2 = hi2 - lo2 + 1
        return (size1 - size2) / max(size1, size2)
    else:  # ranges are numerical
        return (range1 - range2) / max(range1, range2)


def diff_distance(range1, range2):
    if type(range1) == tuple and type(range2) == tuple:
        # we know one range is no less than the other
        lo1, hi1 = range1
        lo2, hi2 = range2
        size1 = hi1 - lo1 + 1
        size2 = hi2 - lo2 + 1
        return size1 - size2
    else:  # ranges are numerical
        return range1 - range2


# https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
def stable_soften_sign(num, soften):
    prod = num * soften
    if prod >= 0:
        return 1/(1 + np.exp(-1 * prod))
    else:
        return np.exp(prod)/(1 + np.exp(prod))


def monotonic_regularization(mono_preds, predicate_ranges, mono_constraints, lbda, dist, soften):
    regs = []
    for constraint in mono_constraints:
        left, right = constraint
        if 0 <= left < len(mono_preds) and 0 <= right <= len(mono_preds):
            if dist == 'jaccard':
                true_dist = jaccard_distance(predicate_ranges[left], predicate_ranges[right])
                pred_dist = jaccard_distance(mono_preds[left], mono_preds[right])
            elif dist == 'diff':
                true_dist = diff_distance(predicate_ranges[left], predicate_ranges[right])
                pred_dist = diff_distance(mono_preds[left], mono_preds[right])
            if true_dist == 0:
                regs.append(0)
            else:
                regs.append(lbda*(stable_soften_sign(pred_dist, soften) - stable_soften_sign(true_dist, soften))**2)
    return torch.mean(torch.FloatTensor(regs))


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
            targets)
        sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
            join_masks)

        t = time.time()
        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror_df = list()
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
        qerror_df.append({"idx": i, "pred": preds_unnorm[i], "label": float(labels_unnorm[i]), "qerror": qerror[-1]})

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

    return pd.DataFrame(qerror_df)


def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda, cmp=False,
    lbda=0.0, regbatch=1024, dist='jaccard', soften=1.0, log_dir=None):
    random.seed(10)
    np.random.seed(10)

    # Load training and validation data
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    # load workload for monotonic regularization
    if lbda != 0.0:
        print('Using lambda = {} and c = {} for monotonic regularization'.format(lbda, soften))
        monotonic_data_loader, monotonic_constraints, predicate_ranges = load_monotonic_regularization(
            table2vec, column2vec, op2vec, join2vec, min_val, max_val, column_min_max_vals,
            num_materialized_samples, batch_size
        )

    model.train()

    # save per-epoch loss
    if log_dir:
        train_loss = list()

    for epoch in range(num_epochs):
        loss_total = 0.
        if lbda != 0.0:
            loss_total_q, loss_total_reg = 0.0, 0.0
        
        epoch_start_time = time.time()

        for batch_idx, data_batch in enumerate(train_data_loader):

            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

            if cuda:
                samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
                targets)
            sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
                join_masks)

            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            if lbda == 0.0:
                loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            else:
                monotonic_pred, _ = predict(model, monotonic_data_loader, cuda)
                mono_pred_unnorm = unnormalize_labels(monotonic_pred, min_val, max_val)
                qerror = qerror_loss(outputs, targets.float(), min_val, max_val)
                constraint_batch = random.sample(monotonic_constraints, k=regbatch)
                mono_reg = monotonic_regularization(
                    mono_pred_unnorm, predicate_ranges, constraint_batch, lbda, dist, soften)
                loss = qerror + mono_reg
                loss_total_q += qerror.item()
                loss_total_reg += mono_reg.item()
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
        
        epoch_end_time = time.time()
        epoch_train_time = epoch_end_time - epoch_start_time
    
        # compute validation loss
        val_loss_total = 0.
        if lbda != 0.0:
            val_loss_total_q, val_loss_total_reg = 0.0, 0.0
        for val_batch_idx, val_data_batch in enumerate(test_data_loader):
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = val_data_batch
            if cuda:
                samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
                targets)
            sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
                join_masks)
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            if lbda == 0.0:
                loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            else:
                monotonic_pred, _ = predict(model, monotonic_data_loader, cuda)
                mono_pred_unnorm = unnormalize_labels(monotonic_pred, min_val, max_val)
                qerror = qerror_loss(outputs, targets.float(), min_val, max_val)
                constraint_batch = random.sample(monotonic_constraints, k=regbatch)
                mono_reg = monotonic_regularization(
                    mono_pred_unnorm, predicate_ranges, constraint_batch, lbda, dist, soften)
                loss = qerror + mono_reg
                val_loss_total_q += qerror.item()
                val_loss_total_reg += mono_reg.item()
            val_loss_total += loss.item()

        train_loss_total = loss_total / len(train_data_loader)
        val_loss_total = val_loss_total / len(test_data_loader)
        if lbda != 0.0:
            loss_total_q = loss_total_q / len(train_data_loader)
            loss_total_reg = loss_total_reg / len(train_data_loader)
            val_loss_total_q = val_loss_total_q / len(test_data_loader)
            val_loss_total_reg = val_loss_total_reg / len(test_data_loader)

        # record logs for analysis
        if log_dir:
            info = {'epoch': epoch, "time": epoch_train_time, "train_loss": train_loss_total, "val_loss": val_loss_total}
            if lbda != 0.0:
                info['train_loss_q'] = loss_total_q
                info['train_loss_reg'] = loss_total_reg
                info['val_loss_q'] = val_loss_total_q
                info['val_loss_reg'] = val_loss_total_reg
            train_loss.append(info)

        if lbda == 0.0:
            print("Epoch {}, train loss: {}, val loss: {}".format(epoch, train_loss_total, val_loss_total))
        else:
            print("Epoch {}, train loss: {} (qerror {} reg {}), val loss: {} (qerror {} reg {})".format(
                epoch, train_loss_total, loss_total_q, loss_total_reg,
                val_loss_total, val_loss_total_q, val_loss_total_reg
            ))
    
    # cache per-iteration train loss
    if log_dir:
        tldf = pd.DataFrame(train_loss)
        tldf.to_csv(os.path.join(log_dir, 'train_log.csv'), index=False)

    # Get final training and validation set predictions
    preds_train, t_total = predict(model, train_data_loader, cuda)
    print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)
    print("")

    # Load test data
    file_name = "workloads/" + workload_name
    joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

    # Get feature encoding and proper normalization
    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    max_num_predicates = max([len(p) for p in predicates_test])
    max_num_joins = max([len(j) for j in joins_test])

    # Get test set predictions
    test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error " + workload_name + ":")
    qerror_df = print_qerror(preds_test_unnorm, label)

    # Print MonoM score
    if cmp:
        print("\nMonoM " + workload_name + ":")
        monom_df = print_monom(preds_test_unnorm, os.path.join("workloads", workload_name + '.cmp'))
    
    # Compute overall error
    overall_error = dict()
    overall_error['qerror_median'] = np.median(qerror_df.qerror.values)
    overall_error['qerror_mean'] = np.mean(qerror_df.qerror.values)
    overall_error['qerror_10'], overall_error['qerror_25'], overall_error['qerror_75'], overall_error['qerror_90'] = \
        np.percentile(a=qerror_df.qerror.values, q=[10, 25, 75, 90])
    
    # Maybe compute MSE and other metrics for more complete error analysis?
    y_test_pred, y_true = qerror_df.pred.values, qerror_df.label.values
    overall_error['mse'] = mean_squared_error(y_true, y_test_pred)
    overall_error['rmsd'] = np.sqrt(overall_error['mse'])
    overall_error['nrmsd'] = overall_error['rmsd'] / (np.amax(y_true) - np.amin(y_true))
    
    if cmp:
        overall_error['monom_median'] = np.median(monom_df.MonoM.values)
        overall_error['monom_mean'] = np.mean(monom_df.MonoM.values)

    # Write predictions
    if log_dir is None:
        file_name = "results/predictions_" + workload_name + ".csv"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            for i in range(len(preds_test_unnorm)):
                f.write(str(preds_test_unnorm[i]) + "," + label[i] + "\n")
    else:
        qerror_df.to_csv(os.path.join(log_dir, "test_qerror.csv"), index=False)
        if cmp:
            monom_df.to_csv(os.path.join(log_dir, "test_monom.csv"), index=False)
        with open(os.path.join(log_dir, 'eval.json'), "w") as fout:
            json.dump(overall_error, fout, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic, scale, or job-light")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--cmp", help="whether to perform MonoM evaluation", action="store_true")
    parser.add_argument("--lbda", help="monotonicity regularization strength (default: 0)", type=float, default=0.0)
    parser.add_argument('--regbatch', help='number of monotonic pairs used for regularization per batch', type=int, default=1024)
    parser.add_argument("--dist", help="distance between two cardinalities", type=str, default='jaccard', choices=['jaccard', 'diff'])
    parser.add_argument("--soften", help="constant for soften sign function (default: 100)", type=float, default=100)
    parser.add_argument("--log", help="path to completely log this experiment", type=str, default=None)
    args = parser.parse_args()

    if args.log is not None:
        log_dir = os.path.join('logs', args.log, str(int(time.time())))
        os.system('mkdir -p {}'.format(log_dir))

        config_dict = {
            "testset": args.testset,
            "num_train_queries": args.queries,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "num_hidden_units": args.hid,
            "cuda": args.cuda,
            "lbda": args.lbda
        }
        if args.lbda > 0.0:  # add regularization info
            config_dict['regbatch'] = args.regbatch
            config_dict['reg_dist_metric'] = args.dist
            config_dict['soften'] = args.soften
        
        with open(os.path.join(log_dir, 'config.json'), 'w') as fout:
            json.dump(config_dict, fout, indent=2)
    else:
        log_dir = None

    train_and_predict(
        args.testset, args.queries, args.epochs, args.batch, args.hid, args.cuda, args.cmp,
        args.lbda, args.regbatch, args.dist, args.soften, log_dir
    )


if __name__ == "__main__":
    main()