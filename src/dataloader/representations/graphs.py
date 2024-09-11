"""
Contains methods to generate graph representations 
of molecules, chemical reactions and proteins.

# Inspired by gauche DataLoader
# https://github.com/leojklarner/gauche
"""

from typing import List, Optional

import networkx as nx
from torch_geometric.data import Data


def graphein_molecular_graphs(
    smiles: List[str], graphein_config: Optional[bool] = None
) -> List[nx.Graph]:
    """
    Convers a list of SMILES strings into molecular graphs
    using the feautrisation utilities of graphein.

    :param smiles: list of molecular SMILES
    :type smiles: list
    :param graphein_config: graphein configuration object
    :type graphein_config: graphein/config/graphein_config
    :return: list of molecular graphs
    """

    import graphein.molecule as gm

    return [
        gm.construct_graph(smiles=i, config=graphein_config) for i in smiles
    ]


def graphein_molecular_graphs(
    smiles: List[str], graphein_config: Optional[bool] = None
) -> List[nx.Graph]:
    """
    Convers a list of SMILES strings into molecular graphs
    using the feautrisation utilities of graphein.

    :param smiles: list of molecular SMILES
    :type smiles: list
    :param graphein_config: graphein configuration object
    :type graphein_config: graphein/config/graphein_config
    :return: list of molecular graphs
    """

    import graphein.molecule as gm

    return [
        gm.construct_graph(smiles=i, config=graphein_config) for i in smiles
    ]

def pyg_molecular_graphs(
    smiles: List[str], 
) -> List[Data]:
    """
    Convers a list of SMILES strings into PyGeometric molecular graphs.

    :param smiles: list of molecular SMILES
    :type smiles: list
    :return: list of PyGeometric molecular graphs
    """

    from torch_geometric.utils import from_smiles

    return [
        from_smiles(smiles=i) for i in smiles
    ]


def molecular_graphs(
    smiles: List[str],
    init_globals: Optional[bool] = False,
) -> List[Data]:
    """
    Converts a list of SMILES strings into a custom graph tuple
    """

    from .graph_utils import from_smiles

    return [
        from_smiles(smiles=i, init_globals=init_globals) for i in smiles
    ]
