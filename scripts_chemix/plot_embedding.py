import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )


import numpy as np
import seaborn as sns
import pandas as pd
from dataloader import DreamLoader
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchmetrics
from scipy.stats import wilcoxon
from ast import literal_eval

import umap

sns.set_context('talk', font_scale=1.7)

all_df = []
for filename, tag in zip(
    ['single_mixtures_train.pkl',
     'single_mixtures_leaderboard.pkl',
     'single_mixtures_test.pkl'], 
    ['Train', 
     'Leaderboard',
     'Test']
):
    df = pd.read_pickle(filename)
    df['Set'] = tag

    all_df.append(df)

all_df = pd.concat(all_df)

all_df['dataset'] = all_df['dataset'].str.replace(' 1', '').str.replace(' 2', '')
all_df['dataset'] = all_df['dataset'].str.replace('Test', 'DREAM')

reducer = umap.UMAP(metric='cosine')
red_dim = reducer.fit_transform(np.array(all_df.mixture_embedding.tolist()))

all_df['UMAP 1'] = list(red_dim[:,0])
all_df['UMAP 2'] = list(red_dim[:,1])

# o_s = len(all_df[all_df['Set'] == 'Train']) * [1]
# x_s = len(all_df[all_df['Set'] == 'Leaderboard']) * [10]
# p_s = len(all_df[all_df['Set'] == 'Test']) * [10]


fig, ax = plt.subplots(1, 1, figsize=(13, 13))
sns.scatterplot(all_df, x='UMAP 1', y='UMAP 2',  hue='dataset', style='Set', 
                markers=['o', 'X', 'P'], palette=sns.color_palette('colorblind'),
                alpha=0.7, ax=ax)
plt.savefig('embedding_space.png', bbox_inches='tight')
plt.savefig('embedding_space.svg', bbox_inches='tight', format='svg')
import pdb; pdb.set_trace()

