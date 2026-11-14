from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import yaml


class LetterPreprocessor:
    """Preprocesses ASLAD-96K static hand landmark data (wrist-centering + scale normalization)."""

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data_cfg = self.config["data"]
        self.prep_cfg = self.config["preprocessing"]

    def _load_config(self) -> dict:
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_data(self) -> pd.DataFrame:
        raw_path = Path(self.data_cfg["raw_csv_path"])
        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset not found: {raw_path}")
        return pd.read_csv(raw_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        if self.prep_cfg.get("drop_na", True):
            cleaned = cleaned.dropna()
        if self.prep_cfg.get("drop_duplicates", True):
            cleaned = cleaned.drop_duplicates()
        return cleaned.reset_index(drop=True)

    def _extract_landmarks_and_labels(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        label_col = str(self.data_cfg["label_col"])
        y = df[label_col].to_numpy(dtype=np.int64)

        feature_cols = [
            c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        raw_vals = df[feature_cols].to_numpy(dtype=np.float32)

        if raw_vals.shape[1] >= 63:
            raw_vals = raw_vals[:, :63]
        else:
            raise ValueError(f"Expected at least 63 numeric columns, got {raw_vals.shape[1]}")

        if self.prep_cfg.get("drop_all_zero_rows", True):
            valid_mask = ~np.all(np.isclose(raw_vals, 0.0, atol=1e-7), axis=1)
            raw_vals = raw_vals[valid_mask]
            y = y[valid_mask]

        landmarks = raw_vals.reshape(-1, 21, 3)
        return landmarks, y

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Applies wrist-centering (landmark 0) and scale normalization (distance 0 -> 9)."""
        wrist_idx = int(self.prep_cfg.get("wrist_index", 0))
        scale_idx = int(self.prep_cfg.get("scale_ref_index", 9))
        eps = float(self.prep_cfg.get("eps", 1e-6))

        wrist = landmarks[:, wrist_idx : wrist_idx + 1, :]
        centered = landmarks - wrist

        scale = np.linalg.norm(
            centered[:, scale_idx, :] - centered[:, wrist_idx, :],
            axis=1,
            keepdims=True,
        )[..., np.newaxis]

        normalized = centered / (scale + eps)
        return normalized.reshape(normalized.shape[0], -1).astype(np.float32)

    def run(self) -> Tuple[Path, Path]:
        df = self.load_data()
        df_clean = self.clean_data(df)
        landmarks, y = self._extract_landmarks_and_labels(df_clean)
        X_norm = self.normalize_landmarks(landmarks)

        out_dir = Path(self.data_cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        x_path = out_dir / self.prep_cfg.get("output_x_name", "X_letters_norm.npy")
        y_path = out_dir / self.prep_cfg.get("output_y_name", "y_letters.npy")

        np.save(x_path, X_norm)
        np.save(y_path, y)
        return x_path, y_path


if __name__ == "__main__":
    preprocessor = LetterPreprocessor("configs/letters.yaml")
    x_out, y_out = preprocessor.run()
    print("Saved X to:", x_out)
    print("Saved y to:", y_out)
