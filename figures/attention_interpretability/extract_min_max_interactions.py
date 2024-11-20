import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-2])
sys.path.append(str(base_dir / "src/"))

import copy
import torch
import numpy as np
import pandas as pd

from chemix.model import build_chemix, compute_key_padding_mask
from chemix.data import dataset_to_torch
from dataloader import DatasetLoader
from omegaconf import OmegaConf

import rdkit
from rdkit import Chem
from argparse import ArgumentParser


def process_mixture(
    final_kp,
    final_attn_weights,
    all_mix_stats,
    mixture_smiles,
    smiles_mix,
    pair_id,
    mix_id,
):

    pad_start = np.argmax(final_kp[pair_id, :, mix_id].cpu().numpy())

    # Handling largest mixture padding
    if pad_start == 0:
        pad_start = 43

    attn_weights = final_attn_weights.cpu().numpy()[pair_id, 0, :pad_start, :pad_start, mix_id]

    smiles_list = mixture_smiles[pair_id][mix_id]

    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    for mol in molecules:
        Chem.RemoveStereochemistry(mol)

    smiles_list = [Chem.MolToSmiles(mol) for mol in molecules]

    if smiles_list not in smiles_mix:

        # iterating over queries
        num_row = attn_weights.shape[0]

        for r in range(num_row):
            row =np.expand_dims(attn_weights[r, :], axis=0)

            max_index = np.argmax(row)
            max_row, max_col = np.unravel_index(max_index, row.shape)

            min_index = np.argmin(row)
            min_row, min_col = np.unravel_index(min_index, row.shape)

            all_mix_stats["matrix_index"].append(pair_id)
            all_mix_stats["mixture_num"].append(mix_id)

            all_mix_stats["max_query_mol"].append(mixture_smiles[pair_id][mix_id][r])
            all_mix_stats["max_key_mol"].append(mixture_smiles[pair_id][mix_id][max_col])
            all_mix_stats["max_key_idx"].append(max_col)
            all_mix_stats["max_weight_val"].append(row[max_row, max_col])

            all_mix_stats["min_query_mol"].append(mixture_smiles[pair_id][mix_id][r])
            all_mix_stats["min_key_mol"].append(mixture_smiles[pair_id][mix_id][min_col])
            all_mix_stats["min_key_idx"].append(min_col)
            all_mix_stats["min_weight_val"].append(row[min_row, min_col])

        smiles_mix.append(smiles_list)

    return all_mix_stats, smiles_mix

def main(
    config,
    checkpoint_path,
):

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

    mixture_smiles = [mix for mix in dl.features]

    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize("mix_pom_embeddings")

    print(dl.features.shape)

    _, test_loader = dataset_to_torch(
        X=dl.features,
        y=dl.labels,
        batch_size=1000,
        num_workers=config.num_workers,
    )

    # Model
    model = build_chemix(config=config.chemix).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

    model.eval()

    with torch.no_grad():

        # only one batch
        for batch in test_loader:
            features, _ = batch

            # project
            key_padding_all =[]
            x_all_input = []
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
                x_all_input.append(mix)
                x_all.append(emb_mix)
                attn_weights_all.append(attn_weights)

            final_kp = torch.stack(key_padding_all, dim=-1)
            final_input_emb = torch.stack(x_all_input, dim=-1)
            final_emb = torch.stack(x_all, dim=-1)
            final_attn_weights = torch.stack(attn_weights_all, dim=-1)
    
    # Save for UMAP
    torch.save(final_input_emb, "final_input_emb.pt")
    torch.save(final_emb, "final_emb.pt")

    all_mix_stats = {
        "matrix_index": [],
        "mixture_num": [],
        "max_key_mol": [],
        "max_key_idx": [],
        "max_query_mol": [],
        "max_weight_val": [],
        "min_key_mol": [],
        "min_key_idx": [],
        "min_query_mol": [],
        "min_weight_val": [],
    }

    smiles_mix = []

    for pair_id, _ in enumerate(mixture_smiles):

        # Mix 1
        all_mix_stats, smiles_mix = process_mixture(
            final_kp=final_kp,
            final_attn_weights=final_attn_weights,
            all_mix_stats=all_mix_stats,
            mixture_smiles=mixture_smiles,
            smiles_mix=smiles_mix,
            pair_id=pair_id,
            mix_id=0,
        )

        # Mix 2
        all_mix_stats, smiles_mix = process_mixture(
            final_kp=final_kp,
            final_attn_weights=final_attn_weights,
            all_mix_stats=all_mix_stats,
            mixture_smiles=mixture_smiles,
            smiles_mix=smiles_mix,
            pair_id=pair_id,
            mix_id=1,
        )

    df = pd.DataFrame.from_dict(all_mix_stats)
    df.to_csv("min_max_keys_across_rows.csv", index=False)

if __name__ == "__main__":

    parser = ArgumentParser(description="Process some model parameters.")
    parser.add_argument(
        "--run_name",
        default= "interpretable_model",
        help="The name of the run to identify model files.",
    )
    parser.add_argument(
        "--model_dir",
        default= str(base_dir / "scripts" / "chemix" / "results" / "interpretability_model"),
        help="Directory containing the model files.",
    )

    args = parser.parse_args()

    model_checkpoint_path = f"{args.model_dir}/best_model_dict_{args.run_name}.pt"
    model_hparams_path = f"{args.model_dir}/hparams_chemix_{args.run_name}.json"

    config = OmegaConf.load(model_hparams_path)

    main(
        config=config,
        checkpoint_path=model_checkpoint_path,
    )
