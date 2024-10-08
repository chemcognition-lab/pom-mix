import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )

import copy
import torch
import numpy as np

from chemix.model import build_chemix, compute_key_padding_mask
from chemix.data import dataset_to_torch
from dataloader import DatasetLoader
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from argparse import ArgumentParser

from pommix_utils import set_visualization_style


def main(
    config,
    checkpoint_path,
    interesting_idx = None,
):

    # Plot style
    set_visualization_style()
    green_color = sns.color_palette("Dark2")[4]
    cmap = mcolors.LinearSegmentedColormap.from_list("white_to_green", [(1, 1, 1), green_color])

    config = copy.deepcopy(config)

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    print(f'Running on: {device}')

    root_dir = config.root_dir
    os.makedirs(root_dir, exist_ok=True)

    # Data
    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize("mix_smiles")

    if interesting_idx is not None:
        good_size_mix_idx = [interesting_idx]
        smiles_good_size = [dl.features[interesting_idx]]
    else:
        good_size_mix_idx = [i for i, mix in enumerate(dl.features) if len(mix[0]) <= 10 and len(mix[1]) <= 10 and len(mix[0]) >= 4 and len(mix[1]) >= 4]
        smiles_good_size = [mix for i, mix in enumerate(dl.features) if i in good_size_mix_idx]

    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize("mix_pom_embeddings")

    _, test_loader = dataset_to_torch(
        X=dl.features[good_size_mix_idx],
        y=dl.labels[good_size_mix_idx],
        batch_size=1000,
        num_workers=config.num_workers,
    )

    # Model
    model = build_chemix(config=config.chemix).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            features, _ = batch

            # project
            key_padding_all =[]
            x_all = []
            attn_weights_all = []
            for mix in torch.unbind(features, dim=-1):
                key_padding_mask = compute_key_padding_mask(mix)
                mix = model.input_net(mix.to(device))

                emb_mix, attn_weights = model.mixture_net.mol_attn_layers[0].forward(
                    mix.to(device),
                    key_padding_mask.to(device),
                )

                key_padding_all.append(key_padding_mask)
                x_all.append(emb_mix)
                attn_weights_all.append(attn_weights)

            final_kp = torch.stack(key_padding_all, dim=-1)
            final_emb = torch.stack(x_all, dim=-1)
            final_attn_weights = torch.stack(attn_weights_all, dim=-1)

    smiles_mix = []
    for i, mix_idx in enumerate(good_size_mix_idx):

        # Mix 1
        print("mix1")

        pad_start = np.argmax(final_kp[i,:,0].cpu().numpy())

        attn_weights = final_attn_weights.cpu().numpy()[i, 0, :pad_start, :pad_start, 0]

        smiles_list = smiles_good_size[i][0].tolist()

        if smiles_list in smiles_mix:
            print("mix already in list")
        else:
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn_weights, cmap=cmap, vmax=1, vmin=0)

            plt.title('Attention Heatmap', fontsize=20)
            plt.ylabel("Queries", fontsize=18)
            plt.xlabel("Keys", fontsize=18)
            plt.savefig(f"./attention_heatmap_mix1_{mix_idx}.svg")
            plt.savefig(f"./attention_heatmap_mix1_{mix_idx}.png")
            plt.close()

            smiles_mix.append(smiles_list)

        # Mix 2
        print("mix2")
        pad_start = np.argmax(final_kp[i,:,1].cpu().numpy())

        attn_weights = final_attn_weights.cpu().numpy()[i, 0, :pad_start, :pad_start, 1]

        smiles_list = smiles_good_size[i][1].tolist()

        if smiles_list in smiles_mix:
            print("mix already in list")
        else:
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn_weights, cmap=cmap, vmax=1, vmin=0)

            plt.title('Attention Heatmap')
            plt.ylabel("Queries")
            plt.xlabel("Keys")
            plt.savefig(f"./attention_heatmap_mix2_{mix_idx}.svg")
            plt.savefig(f"./attention_heatmap_mix2_{mix_idx}.png")
            plt.close()
            smiles_mix.append(smiles_list)

if __name__ == "__main__":
    parser = ArgumentParser(description='Process some model parameters.')
    parser.add_argument('--run_name', help='The name of the run to identify model files.')
    parser.add_argument('--model_dir', help='Directory containing the model files.')
    parser.add_argument('--interesting_idx', action="store_true", default=86, help='Pick a mixture pair.')

    args = parser.parse_args()

    model_checkpoint_path = f"{args.model_dir}best_model_dict_{args.run_name}.pt"
    model_hparams_path = f"{args.model_dir}hparams_chemix_{args.run_name}.json"
    config = OmegaConf.load(model_hparams_path)

    main(
        config=config,
        checkpoint_path=model_checkpoint_path,
        interesting_idx=args.interesting_idx,
    )