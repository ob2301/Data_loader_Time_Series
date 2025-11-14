from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from fmtk.datasets.base import TimeSeriesDataset
import os

class EpilepsyDataset(TimeSeriesDataset):
    def __init__(
        self,
        dataset_cfg,
        task_cfg,
        split,
        data_stride_len: int = 1,
        random_seed: int = 13,
    ):
        """
        Epilepsy dataset for 4-class time-series classification.
        Each row = multivariate signal (with multiple channels).
        Labels are {1, 2, 3, 4} → converted to {0, 1, 2, 3}.
        """
        super().__init__(dataset_cfg, task_cfg, split)
        self.data_stride_len = data_stride_len
        self.random_seed = random_seed
        self.task_name = self.task_cfg["task_type"]
        self.full_file_path_and_name = (
            f"{self.dataset_cfg['dataset_path']}/Epilepsy_{split.upper()}.txt"
        )
        self._read_data()

    def _read_data(self):
        """
        Reads, cleans, scales, and stores Epilepsy time-series data.
        Follows same structure as FordA but adapted for multichannel samples.
        """
        clean_lines = []
        with open(self.full_file_path_and_name, "r", encoding="utf-8") as f:
            for line in f:
                # skip metadata lines
                if line.startswith("@") or line.startswith("#") or line.startswith("%"):
                    continue
                clean_lines.append(line.strip())

        parsed_samples = []
        for line in clean_lines:
            # each ':' separates one channel
            sample_channels = []
            for ch_str in line.split(":"):
                ch_vals = []
                for token in ch_str.split(","):
                    token = token.strip()
                    if token == "":
                        continue
                    try:
                        ch_vals.append(float(token))
                    except ValueError:
                        continue
                if ch_vals:
                    sample_channels.append(ch_vals)

            # skip malformed samples
            if len(sample_channels) == 0:
                continue

            # first value of first channel = label
            label = int(sample_channels[0][0]) - 1
            label = max(label, 0)  # fix potential -1 issue
            sample_channels[0] = sample_channels[0][1:]
            parsed_samples.append((label, sample_channels))

        # convert to consistent array shapes
        max_len = max(len(ch) for _, chs in parsed_samples for ch in chs)
        n_channels = max(len(chs) for _, chs in parsed_samples)

        data = []
        labels = []
        for label, chs in parsed_samples:
            arr = np.zeros((n_channels, max_len), dtype=np.float32)
            for i, ch in enumerate(chs):
                arr[i, : len(ch)] = ch
            data.append(arr)
            labels.append(label)

        X = np.stack(data)
        y = np.array(labels, dtype=int)

        # flatten for StandardScaler (so it works channel-wise)
        X_reshaped = X.reshape(X.shape[0], -1)

        np.random.seed(self.random_seed)
        if self.split == "train":
            np.random.shuffle(X_reshaped)

        mean_path = os.path.join(self.dataset_cfg["dataset_path"], "epilepsy_mean.npy")
        scale_path = os.path.join(self.dataset_cfg["dataset_path"], "epilepsy_scale.npy")

        # scale features same way as FordA
        if self.split == "train":
            self.scaler = StandardScaler()
            self.scaler.fit(X_reshaped)
            np.save(mean_path, self.scaler.mean_)
            np.save(scale_path, self.scaler.scale_)
        else:
            mean = np.load(mean_path)
            scale = np.load(scale_path)
            self.scaler = StandardScaler()
            self.scaler.mean_ = mean
            self.scaler.scale_ = scale

        X_scaled = self.scaler.transform(X_reshaped).astype(np.float32)
        self.data = X_scaled.reshape(X.shape)
        self.labels = y
        self.seq_len = self.data.shape[-1]
        self.n_channels = self.data.shape[1]

    def __getitem__(self, index):
        """
        Each sample = multichannel time series tensor (C, L)
        Returns normalized signal and corresponding class label.
        """
        timeseries = self.data[index]
        # per-sample normalization (same idea as FordA)
        timeseries = (timeseries - timeseries.mean()) / (timeseries.std() + 1e-8)
        label = self.labels[index]
        return timeseries, label

    def __len__(self):
        return len(self.labels)

    def preprocess(self):
        pass
