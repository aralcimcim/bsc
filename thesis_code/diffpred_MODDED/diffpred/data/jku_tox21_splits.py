from pathlib import Path
import pandas as pd
import torch
import datamol as dm

from diffpred.data.mpp_datamodule import MPPDatamodule

TOX21_TASKS = [
    "NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


class CsvSplitDatamodule(MPPDatamodule):
    def __init__(
        self,
        dataset,
        seed,
        batch_size=128,
        featurizer=None,
        standardize_targets: bool = False,
        log_transform: bool = False,
        data_dir: str = "data/tox21_jku",
        smiles_col: str = "smiles",
        target_columns: list[str] | None = None,
        load_test: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            featurizer=featurizer,
            standardize_targets=standardize_targets,
            log_transform=log_transform,
            seed=seed,
            use_benchmark=False,
        )
        self.data_dir = Path(data_dir)
        self.smiles_col = smiles_col
        self.target_columns = list(target_columns or TOX21_TASKS)
        self.load_test = load_test

    def setup(self, stage=None) -> None:
        if self.is_setup:
            return
        self.is_setup = True

        self.train_df = self._drop_invalid_smiles(self._read_split("train"), "train")
        self.val_df = self._drop_invalid_smiles(self._read_split("valid"), "val")
        self.test_df = (
            self._drop_invalid_smiles(self._read_split("test"), "test")
            if self.load_test else None
        )

        train_targets = self.train_df.loc[:, self.target_columns].to_numpy()
        self.standardization_stats(train_targets)
        transform = self.get_transform(self.train_df[self.smiles_col].tolist())

        self.train_dt = self.df_to_dt(self.train_df, transform)
        self.val_dt = self.df_to_dt(self.val_df, transform)
        self.test_dt = self.df_to_dt(self.test_df, transform) if self.load_test else []

    def test_dataloader(self):
        if not self.load_test:
            raise RuntimeError("Hold-out loading is disabled for this tuning run")
        return super().test_dataloader()

    def _drop_invalid_smiles(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        keep = [dm.to_mol(s) is not None for s in df[self.smiles_col]]
        n_drop = int((~pd.Series(keep)).sum())
        if n_drop:
            print(f"Dropped {n_drop} invalid SMILES from {split}")
        return df.loc[keep].reset_index(drop=True)

    def _read_split(self, split: str) -> pd.DataFrame:
        candidates = [self.data_dir / f"{split}.csv"]
        if split == "valid":
            candidates.append(self.data_dir / "val.csv")
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing {split} csv in {self.data_dir}")

        df = pd.read_csv(path)
        missing = [c for c in [self.smiles_col, *self.target_columns] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        return df

    def _extract_smiles_targets(self, df):
        smiles = df[self.smiles_col].tolist()
        targets = torch.as_tensor(
            df.loc[:, self.target_columns].to_numpy(), dtype=torch.float32
        )
        return smiles, targets
