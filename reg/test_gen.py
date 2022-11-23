import psycopg2 as pg
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import csv
import argparse

def generate_new_line(row, base_predicates):
    new_rows = []
    new_rows_with_years = []
    predicates = row["predicates"]
    if "production_year" in predicates:
        if ">" in predicates and "<" in predicates:
            for year1 in range(1870, 2022):
                sub_base_predicates = str(base_predicates) + "_"+str(year1-1870)
                relative_order = 0 
                for year2 in range(year1+2, 2022):
                    gindex, sindex = predicates.index(">"), predicates.index("<")
                    new_predicates = predicates.replace(predicates[gindex:gindex+6], ">,"+str(min(year1, year2)))
                    new_predicates = new_predicates.replace(predicates[sindex:sindex+6], "<,"+str(max(year1, year2)))
                    new_rows.append([row["tables"], row["joins"], new_predicates, None])        
                    new_rows_with_years.append([row["tables"], row["joins"], new_predicates, sub_base_predicates, "range", year1, year2, relative_order])
                    relative_order += 1
        
        elif ">" in predicates:
            index = predicates.index(">")
            for year in range(1870, 2022):
                new_predicates = predicates.replace(predicates[index:index+6], ">,"+str(year))
                new_rows.append([row["tables"], row["joins"], new_predicates, None])
                new_rows_with_years.append([row["tables"], row["joins"], new_predicates, base_predicates, ">",  year, 10000, 2022-year])
        elif "<" in predicates:
            index = predicates.index("<")
            for year in range(1870, 2022):
                new_predicates = predicates.replace(predicates[index:index+6], "<,"+str(year))
                new_rows.append([row["tables"], row["joins"], new_predicates, None])
                
                new_rows_with_years.append([row["tables"], row["joins"], new_predicates, base_predicates, "<", 0, year , year-1870+1])
        elif "=" in predicates:
            index = predicates.index("=")
            for year in range(1870, 2022):
                new_predicates = predicates.replace(predicates[index:index+6], "=,"+str(year))
                new_rows.append([row["tables"], row["joins"], new_predicates, None])
                new_rows_with_years.append([row["tables"], row["joins"], new_predicates, base_predicates,"=", year, year, year-1870+1])
        
    return new_rows, new_rows_with_years


def get_cmp(df):
    # df is well ordered
    cmp = []
    df = df.sort_values(by=['base_predicates', "relative_order"])
    df["relative_order"] = df["relative_order"].astype(str)
    gp_by_base_predicates = df.groupby(["base_predicates","type"])
    for (base_predicate, ptype), group in gp_by_base_predicates:
        if ptype == "range" or ptype == "<" or ptype == ">": # range selection
            for i in range(len(group.index)):
                if i == 0:
                    cmp.append(str(group.index[i])+"="+str(group.index[i]))
                    df.at[group.index[i], "relative_order"] = str(group.index[i])+"="+str(group.index[i])
                else:
                    cmp.append(str(group.index[i])+">"+str(group.index[i-1]))
                    df.at[group.index[i], "relative_order"] = str(group.index[i])+">"+str(group.index[i-1])
        if ptype == "=": # range selection
            for i in range(len(group.index)-1, -1, -1):
                cmp.append(str(group.index[i])+"="+str(group.index[i]))
                df.at[group.index[i], "relative_order"] = str(group.index[i])+"="+str(group.index[i])
    return df, cmp



def generate_new_queries(file_name, generated_file_name):
    train_df = pd.read_csv("workloads/"+file_name+".csv",
                           sep='#',
                           names=["tables", "joins", "predicates", "count"])
    new_rows = []
    new_rows_with_years = []
    for i in range(train_df.shape[0]):
        rows, rows_with_years = generate_new_line(train_df.iloc[i], base_predicates=i)
        new_rows += rows
        new_rows_with_years += rows_with_years

    # save all generated queries
    new_df = pd.DataFrame(new_rows).iloc[:,0:-1]
    new_df.to_csv("workloads/" + generated_file_name +"-card.csv",
                               sep='#', index=False, header=False) # no cardinality column
    new_df_with_years = pd.DataFrame(new_rows_with_years,
                                     columns=["tables", "joins", "predicates", "base_predicates", "type","year1", "year2", "relative_order"])
    _, cmp = get_cmp(new_df_with_years.copy())
    pd.DataFrame(cmp).to_csv("workloads/" + generated_file_name +"-pairs.csv", index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='job-light', help='path to the csv file')
    parser.add_argument('-o', '--output_file', help='name of output file')
    args = parser.parse_args()
    generate_new_queries(args.file, args.output_file)


if __name__ == '__main__':
    main()