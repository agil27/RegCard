import psycopg2 as pg
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import os


user = 'ceb'
db_host = 'localhost'
port = 5432
pwd = 'password'
db_name = 'imdb'
con = pg.connect(user=user, host=db_host, port=port, password=pwd, database=db_name)
cursor = con.cursor()
cursor


def get_bitmap(row):
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


def get_cardinality(row):
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
    for table_name, table_abbr in all_tables:
        gen_view = 'CREATE MATERIALIZED VIEW %s_view AS SELECT * FROM %s AS t ORDER BY RANDOM() LIMIT 1000;' % (table_abbr, table_name)
        try:
            cursor.execute(gen_view)
        except Exception as e:
            print(e)
            cursor.execute('ABORT;')

    workload_bitmaps = []
    workload_queries = []

    for _, row in tqdm(df.iterrows(), desc='iterating rows...'):
        # generate bitmaps
        try:
            row_bitmap = get_bitmap(row)
            if not bitmap_only:
                row_card = get_cardinality(row)
        except Exception as e:
            print(e)
            cursor.execute('ABORT;')
            continue
        else:
            workload_bitmaps.append(row_bitmap)
            if not bitmap_only:
                workload_queries.append(list(row) + [row_card])

    # save it
    with open(os.path.join(directory, '%s.bitmaps' % (filename, )), 'wb') as f:
        pickle.dump(workload_bitmaps, f)
    if not bitmap_only:     
        pd.DataFrame(workload_queries).to_csv(os.path.join(directory, '%s_card.csv' % (filename, )), sep='#', header=None, index=False)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='workloads', help='directory to store the data')
    parser.add_argument('-f', '--file', type=str, default='job-light', help='path to the csv file')
    parser.add_argument('-b', '--bitmap_only', action='store_true', help='whether to only generate the bitmap (without the cardinality)')
    args = parser.parse_args()
    gen_bitmaps_and_cardinalities(args.file, args.bitmap_only, args.dir)
    

if __name__ == '__main__':
    main()
