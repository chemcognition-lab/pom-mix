import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )

from pommix_utils import bootstrap_ci
from chemix.utils import TORCH_METRIC_FUNCTIONS
from dataloader import DatasetLoader, SplitLoader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import numpy as np

from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument("--split", action="store", type=str, default="ablate_components", choices=["ablate_molecules", "ablate_components"])
# 
# FLAGS = parser.parse_args()

if __name__ == '__main__':

    all_df = []
    samples_df = []
    model_order = [
        'Cosine similarity\nbaseline',
        'XGBoost\nrdkit', 
        'XGBoost\nPOM embed',
        'CheMix\nPOM embed',
        'POM-Mix']

    for filename, tag in zip(
        [   
            base_dir / 'scripts_baseline/snitz_cosine_similarity',
            base_dir / 'scripts_baseline/xgb_rdkit2d',
            base_dir / 'scripts_baseline/xgb_pom_embeddings',
            base_dir / 'scripts_chemix/results/random_cv/model',
            base_dir / 'scripts_pommix/results/random_cv/model',
        ], 
        model_order
    ):
        
        results = []
        samples = {}
        for key, metric_fn in TORCH_METRIC_FUNCTIONS.items():
            tmp = []
            for i in range(5):
                pred_fname = filename / f'cv{i}_test_predictions.csv'

                df = pd.read_csv(pred_fname)
                y_true = torch.from_numpy(df['Predicted_Experimental_Values'].to_numpy(np.float32))
                y_pred = torch.from_numpy(df['Ground_Truth'].to_numpy(np.float32))

                met = metric_fn(y_pred, y_true)
                tmp.append(met.item())

            mu = np.mean(tmp)
            std = np.std(tmp)
            info = {'metric': key, 'low_ci': mu - std, 'upper_ci': mu + std, 'mean': mu}
            results.append(info)

        # samples = pd.DataFrame(samples)
        # samples['fname'] = tag
        # samples_df.append(samples)

        results_df = pd.DataFrame(results) #.set_index('metric')
        results_df['fname'] = tag
        all_df.append(results_df)

    all_df = pd.concat(all_df)
    # samples_df = pd.concat(samples_df)
    
    all_df.to_csv('compiled_metrics.csv', index=False)
    # samples_df.to_csv('compiled_metrics_significance.csv', index=False)
    metrics = all_df['metric'].unique()

    # # loop through and test significance
    # comparisons = []
    # for i, gdf in samples_df.groupby('fname'):
    #     for j, gdf2 in samples_df.groupby('fname'):
    #         if i == j:
    #             continue
    #         for m in metrics:
    #             t_ind, p_val = wilcoxon(gdf[m], gdf2[m])
    #             comparisons.append({'metric': m, 'A': i, 'B': j, 't_ind': t_ind, 'p_val': p_val})
            
    # comparisons = pd.DataFrame(comparisons)
    # print('The following are NOT statistically significantly different...')
    # print(comparisons[comparisons['p_val'] > 0.05].head())

    # print()
    # print('Here are the statistics... ')
    # print(all_df)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(30, 10), sharey=True)
    axs = axs.flatten()

    # List of metrics
    metric_name = {
        'pearson':'Pearson (↑)',
        'kendall':'Kendall (↑)',
        'rmse':'RMSE (↓)',
    }

    # Plot each metric
    for i, (metric, c) in enumerate(zip(metrics, ['Blues', "pink_r", 'Oranges'])):
        # Filter data for the current metric
        metric_data = all_df[all_df['metric'] == metric]
        palette = sns.color_palette(c, n_colors=len(model_order))

        # Plot the data
        sns.barplot(y='fname', x='mean', hue='fname', data=metric_data, ax=axs[i], order=model_order, orient='h', palette=palette, legend=False)
        (_, caps, _) = axs[i].errorbar(y=range(len(metric_data)), x=metric_data['mean'], 
                        xerr=[metric_data['mean'] - metric_data['low_ci'], 
                            metric_data['upper_ci'] - metric_data['mean']],
                        fmt='none', c='black')
        for cap in caps:
            cap.set_markeredgewidth(5)
        
        # axs[i].locator_params(axis='x', nbins=7)
        axs[i].set_ylabel('Model')
        axs[i].set_xlabel(f'{metric_name[metric]}')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig('performance_models.svg', format='svg')
    plt.savefig('performance_models.png')



    ###### 

    all_df = []
    samples_df = []
    model_order = [
        'CheMix CV',
        'CheMix LMO',
        'POM-Mix CV',
        'POM-Mix LMO']

    for split, filename, tag in zip(
        ['cv', 'lso', 'cv', 'lso'],
        [   
            base_dir / 'scripts_chemix/results/random_cv/model',
            base_dir / 'scripts_chemix/results/lso_molecules/model',
            base_dir / 'scripts_pommix/results/random_cv/model',
            base_dir / 'scripts_pommix/results/lso_molecules/model',
        ], 
        model_order
    ):
        results = []
        samples = {}
        for key, metric_fn in TORCH_METRIC_FUNCTIONS.items():
            tmp = []
            for i in range(5):
                pred_fname = filename / f'{split}{i}_test_predictions.csv'

                df = pd.read_csv(pred_fname)
                y_true = torch.from_numpy(df['Predicted_Experimental_Values'].to_numpy(np.float32))
                y_pred = torch.from_numpy(df['Ground_Truth'].to_numpy(np.float32))

                met = metric_fn(y_pred, y_true)
                tmp.append(met.item())

            # mu = np.mean(tmp)
            # std = np.std(tmp)
            info = {'metric': [key]*len(tmp), 'value': tmp, 'split': [split]*len(tmp)}
            results.append(pd.DataFrame(info))

        # samples = pd.DataFrame(samples)
        # samples['fname'] = tag
        # samples_df.append(samples)
        results = pd.concat(results)
        results['fname'] = tag
        all_df.append(results)

    all_df = pd.concat(all_df)
    # samples_df = pd.concat(samples_df)
    
    all_df.to_csv('compiled_metrics.csv', index=False)
    # samples_df.to_csv('compiled_metrics_significance.csv', index=False)


    metrics = all_df['metric'].unique()

    # # loop through and test significance
    # comparisons = []
    # for i, gdf in samples_df.groupby('fname'):
    #     for j, gdf2 in samples_df.groupby('fname'):
    #         if i == j:
    #             continue
    #         for m in metrics:
    #             t_ind, p_val = wilcoxon(gdf[m], gdf2[m])
    #             comparisons.append({'metric': m, 'A': i, 'B': j, 't_ind': t_ind, 'p_val': p_val})
            
    # comparisons = pd.DataFrame(comparisons)
    # print('The following are NOT statistically significantly different...')
    # print(comparisons[comparisons['p_val'] > 0.05].head())

    # print()
    # print('Here are the statistics... ')
    # print(all_df)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(30, 10), sharey=True)
    axs = axs.flatten()

    # Plot each metric
    for i, (metric, c) in enumerate(zip(metrics, ['Blues', "pink_r", 'Oranges'])):
        # Filter data for the current metric
        metric_data = all_df[all_df['metric'] == metric]
        # palette = sns.color_palette(c, n_colors=len(model_order))

        # Plot the data
        legend = i == 0
        sns.boxplot(data=metric_data, x='fname', y='value', hue='fname', ax=axs[i], order=model_order, legend=legend)
        
        # axs[i].locator_params(axis='x', nbins=7)
        axs[i].set_ylabel('Model')
        axs[i].set_xlabel(f'{metric_name[metric]}')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig('performance_lso_models.svg', format='svg')
    plt.savefig('performance_lso_models.png')

