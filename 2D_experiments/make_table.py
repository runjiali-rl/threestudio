import pandas as pd
import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Make table of results')
    parser.add_argument('--results_dir', default="2D_experiments/results", type=str, help='Path to directory containing results')
    parser.add_argument('--output', default="2D_experiments/results/results_table.json", type=str, help='Path to output file')
    return parser.parse_args()

def main():
    args = parse_args()
    results_dir = args.results_dir
    subdirs = os.listdir(results_dir)
    table_results = {}

    subdirs = [os.path.join(results_dir, subdir) for subdir in subdirs]
    for subdir in subdirs:
        if subdir.endswith('.json'):
            continue
        subdir_key = subdir.split('/')[-1]
        table_results[subdir_key] = {}
        model_results = os.listdir(subdir)
        for model_result in model_results:
            model_result_key = model_result.split('.')[0]
            table_results[subdir_key][model_result_key] = None
            model_result_path = os.path.join(subdir, model_result)
            model_result_df = pd.read_csv(model_result_path)
            # count how many Yes's there are in the 'Result' column
            table_results[subdir_key][model_result_key] = model_result_df['Result'].value_counts()['Yes']/len(model_result_df['Result'])


    with open(args.output, 'w') as f:
        json.dump(table_results, f, indent=4)

if __name__ == '__main__':
    main()