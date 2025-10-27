from fmtk.components.base import BaseModel
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

class MantisModel(BaseModel):
    def __init__(self, device, model_name="8M"):
        super().__init__()
        self.device = device
        self.peft_enable = False

        #initialization step
        self.network = Mantis8M(device="cpu")
        self.network = self.network.from_pretrained(f"paris-noah/Mantis-{model_name}")

        #next step is wrapping it with trainer
        self.model = MantisTrainer(device="cpu", network=self.network)
        #adapters...?
        self.peft_enable=False

    #preprocess step almost the same as MOMENTs since both expect (batch, channels, length), and they both need float inputs
    def preprocess(self, batch):
        if len(batch) == 3:
            x, mask, y = batch
            mask = mask.to(self.device)
        else:
            x, y = batch
            mask = None

        x = x.float().to(self.device)
        self.B, self.S, self.L = x.shape

        #resizing via interpolation, as suggested
        if self.L != 512:
            x = torch.nn.functional.interpolate(x, size=512, mode='linear', align_corners=False)

        return x, mask, y

    def forward(self, batch):
        x, mask, y = self.preprocess(batch)
        x_np = x.cpu().numpy()
        emb_np = self.model.transform(x_np)
        embedding = torch.tensor(emb_np, dtype=torch.float32, device=self.device)
        return embedding, y

    @torch.no_grad()
    def predict(self, dataloader: DataLoader):
        all_embeddings, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader), desc="[Mantis] Embedding"):
            emb, y = self.forward(batch)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(y)
        embeddings_np = np.vstack(all_embeddings)
        labels_np = np.concatenate(all_labels)
        return embeddings_np, labels_np
    
    def enable_peft(self,peft_cfg):
        self.model = get_peft_model(self.model, peft_cfg)
        self.peft_enable=True
    
    def adapter_trainable_parameters(self):
        if not self.peft_enable:
            return []
        return (p for p in self.model.parameters() if p.requires_grad)
