#Mantis TSFM Wrapper for our toolkit

from fmtk.components.base import BaseModel
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model
from huggingface_hub.utils._errors import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError


class MantisModel(BaseModel):
    def __init__(self, device, model_name=None):
        super().__init__()
        #device (match MOMENT wrapper style is passed-in)
        self.device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")

        #model selection (Mantis provides 8M; accept "8M"/"2M", default to 8M)
        if model_name in ("8M", "2M"):
            repo_id = f"paris-noah/Mantis-{model_name}"
        else:
            repo_id = "paris-noah/Mantis-8M"

        # load network, fallback to random init if HF gated/unavailable
        try:
            self.network = Mantis8M(device=self.device)
            self.network = self.network.from_pretrained(repo_id, device=self.device)
        except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError):
            self.network = Mantis8M(device=self.device)

        #scikit-learn like wrapper (transform / fit / predict_proba)
        self.model = MantisTrainer(device=self.device, network=self.network)

        # PEFT toggle to mirror MOMENT wrapper API
        self.peft_enable = False

    def preprocess(self, batch):
        """
        Match the shape and preprocess before sending it to model.
        Args:
            batch: batch from dataloader (supports (x, y) or (x, mask, y))
        Returns:
            x: torch.FloatTensor on device
            mask: torch.BoolTensor or None
            y: labels
        """
        if len(batch) == 3:
            x, mask, y = batch
            mask = mask.to(self.device)
        else:
            x, y = batch
            mask = None

        x = x.float().to(self.device)
        self.B, self.S, self.L = x.shape  # keep same bookkeeping as MOMENT wrapper
        return x, mask, y

    def _to_mantis_layout_and_resize(self, x: torch.Tensor, target_len: int = 512) -> np.ndarray:
        """
        Convert (B, S, L) -> (B, C, T) and interpolate to T=target_len
        - If univariate, C=1.
        - Uses linear interpolation along T.
        Returns numpy array (B, C, target_len).
        """
        x_np = x.detach().cpu().numpy()

        # Ensure channel-first, time-last for Mantis: (B, C, T)
        # FMTK batches are typically (B, S, L) with S=time, L=features -> transpose to (B, L, S)
        if x_np.ndim == 3:
            B, S, L = x_np.shape
            if S > L:
                x_np = np.transpose(x_np, (0, 2, 1))  # (B, L, S)
        elif x_np.ndim == 2:
            # (B, T) -> (B, 1, T)
            x_np = x_np[:, np.newaxis, :]

        # interpolate to fixed length (matches pretraining: 512)
        xt = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        xt = F.interpolate(xt, size=target_len, mode="linear", align_corners=False)
        return xt.detach().cpu().numpy()

    def forward(self, batch):
        x, mask, y = self.preprocess(batch)

        # Prepare input for MantisTrainer.transform()
        x_np = self._to_mantis_layout_and_resize(x, target_len=512)  # official recommendation

        # Zero-shot feature extraction (embeddings)
        try:
            embeddings_np = self.model.transform(x_np)  # returns np.ndarray
        except Exception as e:
            raise RuntimeError(f"[Mantis] Forward transform failed: {e}")

        embeddings = torch.tensor(embeddings_np, dtype=torch.float32, device=self.device)
        return embeddings, y

    def postprocess(self, embedding):
        # Keep parity with MOMENT wrapper API
        pass

    @torch.no_grad()
    def predict(self, dataloader: DataLoader):
        """
        Compute embeddings (no grad) for decoder-only training/inference.
        Returns: embeddings_np [N,E], labels_np [N]
        """
        all_embeddings, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            output, y = self.forward(batch)
            all_embeddings.append(output.cpu().float().numpy())
            all_labels.append(y)
        embeddings_np = np.vstack(all_embeddings)
        labels_np = np.concatenate(all_labels)
        return embeddings_np, labels_np

    def enable_peft(self, peft_cfg):
        self.model = get_peft_model(self.model, peft_cfg)
        self.peft_enable = True

    def adapter_trainable_parameters(self):
        if not self.peft_enable:
            return []
        return (p for p in self.model.parameters() if p.requires_grad)
