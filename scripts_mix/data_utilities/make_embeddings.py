import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )

from pommix_utils import get_embeddings_from_smiles

if __name__ == '__main__':
    embeddings = get_embeddings_from_smiles(['c1ccccc1', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'], 'gs-lf_models/pretrained_pom/')
    print(embeddings)
