from typing import Union, List
import datamol as dm
import torch
import torch_geometric.transforms as T
from molfeat.trans.graph import PYGGraphTransformer
from molfeat.calc.bond import EdgeMatCalculator


class GPSTransformer(PYGGraphTransformer):
    def __init__(self, max_rw_steps: int = 10, num_eig: int = 6):
        super().__init__(
            self_loop=True,
            canonical_atom_order=True,
            bond_featurizer=EdgeMatCalculator(),
        )
        self.transform_op = T.Compose(
            [
                T.AddLaplacianEigenvectorPE(
                    k=num_eig, is_undirected=True, attr_name="lap_pe"
                ),
                T.AddRandomWalkPE(walk_length=max_rw_steps, attr_name="rwse"),
            ]
        )

    def transform(self, mols: List[Union[dm.Mol, str]], **kwargs) -> torch.Tensor:
        graphs = super().transform(mols, **kwargs)
        graphs = self.transform_op(graphs)
        return graphs
