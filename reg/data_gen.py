import psycopg2 as pg
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import os
import multiprocess
import time
from tqdm import tqdm


def connect_psql(user='ceb', db_host='localhost', port='5432', pwd='password', db_name='imdb'):
    con = pg.connect(user=user, host=db_host, port=port, password=pwd, database=db_name)
    cursor = con.cursor()
    return cursor


def get_bitmap(row, cursor):
    tables = row[0].split(',')
    table_abbrs = [t.split(' ')[1] for t in tables]
    predicates = row[2]
    all_bitmaps = np.zeros((len(table_abbrs), 1000), dtype=int)

    # if no predicates on table
    if not isinstance(predicates, list):
        return np.packbits(all_bitmaps, axis=1)

    predicates = predicates.split(',')
    num_predicates = len(predicates) // 3
    for i in range(num_predicates):
        p = ''.join(predicates[3 * i : 3 * i + 3])
        table_abbr = predicates[3 * i].split('.')[0]
        get_bitmap = 'SELECT CASE WHEN %s THEN 1 else 0 END AS bitmap FROM %s_view as %s;' % (p, table_abbr, table_abbr)
        cursor.execute(get_bitmap)
        record = np.array([i[0] for i in list(cursor)])
        all_bitmaps[table_abbrs.index(table_abbr)] = record
    return np.packbits(all_bitmaps, axis=1)


def get_cardinality(row, cursor):
    tables = row[0]
    join_predicates = row[1]
    join_predicates = ' AND '.join(join_predicates.split(','))
    predicates = row[2].split(',')
    num_predicates = len(predicates) // 3
    predicates = ' AND '.join([''.join(predicates[3 * i: 3 * i + 3]) for i in range(num_predicates)])
    get_card = 'SELECT COUNT(*) FROM %s WHERE %s AND %s;' % (tables, join_predicates, predicates)
    # print(get_card)
    cursor.execute(get_card)
    card = cursor.fetchone()
    card = int(card[0])
    return card


def gen_bitmaps_and_cardinalities(filename, bitmap_only=False, directory='workloads'):
    df = pd.read_csv(os.path.join(directory, filename + '.csv'), sep='#', header=None)

    all_tables = []
    for tables in df[0]:
        all_tables.extend(tables.split(','))
    all_tables = [t.split(' ') for t in list(set(all_tables))]
    all_tables

    # generate materialized views for all of the tables
    cursor = connect_psql()
    for table_name, table_abbr in all_tables:
        gen_view = 'CREATE MATERIALIZED VIEW %s_view AS SELECT * FROM %s AS t ORDER BY RANDOM() LIMIT 1000;' % (table_abbr, table_name)
        try:
            cursor.execute(gen_view)
        except Exception as e:
            print(e)
            cursor.execute('ABORT;')
    cursor.close()

    if not bitmap_only:
        cmp_pairs = np.array(pd.read_csv(os.path.join(directory, filename + '-pairs.csv'), header=None)[0])

    num_queries = len(df)
    manager = multiprocess.Manager()
    workload_bitmaps = manager.list([None for _ in range(num_queries)])
    workload_queries = manager.list([None for _ in range(num_queries)])
    workload_pairs = manager.list([None for _ in range(num_queries)])

    num_threads = 8
    pool = multiprocess.Pool(num_threads)

    # a worker process
    def thread_func(param):
        i, row, workload_bitmaps, workload_queries, workload_pairs = param
        # print('row', i, 'started')
        cursor = connect_psql()        
        try:
            row_bitmap = get_bitmap(row, cursor)
            if not bitmap_only:
                row_card = get_cardinality(row, cursor)
        except Exception as e:
            print(e)
            cursor.execute('ABORT;')
            return
        else:
            if bitmap_only or row_card > 0:
                workload_bitmaps[i] = row_bitmap
                if not bitmap_only:
                    workload_pairs[i] = cmp_pairs[i]
                    workload_queries[i] = list(row) + [row_card]
            cursor.execute('COMMIT;')
        # print('row', i, 'finished')

    # add the row generation jobs to the process pool
    params = list(df.iterrows())
    params = [(p[0], p[1], workload_bitmaps, workload_queries, workload_pairs) for p in params]
    for _ in tqdm(pool.imap_unordered(thread_func, params), total=len(df)):
        pass

    # filter the non-NIL rows
    if not bitmap_only:
        non_nil_idx = [i for i in range(num_queries) if workload_pairs[i] is not None]
        idx_map = {j : i for i, j in enumerate(non_nil_idx)}
        workload_bitmaps = [workload_bitmaps[i] for i in non_nil_idx]
        workload_queries = [workload_queries[i] for i in non_nil_idx]
        workload_pairs = [workload_pairs[i] for i in non_nil_idx]
        workload_cmps = []
        for cmp in workload_pairs:
            op = '>' if '>' in cmp else '='
            left, right = cmp.split(op)
            if left in idx_map and right in idx_map:
                left = idx_map[int(left)]
                right = idx_map[int(right)]
            workload_cmps.append(left + op + right)
    else:
        workload_bitmaps = [wb for wb in workload_bitmaps]
        
    # save it
    with open(os.path.join(directory, '%s-card.bitmaps' % (filename, )), 'wb') as f:
        pickle.dump(workload_bitmaps, f)
    if not bitmap_only:     
        pd.DataFrame(workload_queries).to_csv(os.path.join(directory, '%s-card.csv' % (filename, )), sep='#', header=None, index=False)
        pd.Series(workload_cmps).to_csv(os.path.join(directory, '%s-card.cmp' % (filename, )), header=None, index=False)   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='workloads', help='directory to store the data')
    parser.add_argument('-f', '--file', type=str, default='job-light', help='path to the csv file')
    parser.add_argument('-b', '--bitmap_only', action='store_true', help='whether to only generate the bitmap (without the cardinality)')
    args = parser.parse_args()
    gen_bitmaps_and_cardinalities(args.file, args.bitmap_only, args.dir)
    

if __name__ == '__main__':
    main()
