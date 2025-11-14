from fmtk.pipeline import Pipeline
from fmtk.datasets.epilepsy import EpilepsyDataset 
from fmtk.components.backbones.mantis import MantisModel
from fmtk.components.decoders.classification.mlp import MLPDecoder
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader

device = 'cpu'

#same configs reused
task_cfg = {'task_type': 'classification', "num_classes": 4}
inference_config = {'batch_size': 50, 'shuffle': False}
train_config = {'batch_size': 50, 'shuffle': False, 'epochs': 50, 'lr': 1e-2}
dataset_cfg = {'dataset_path': '../dataset/Epilepsy'} 

print("Starting Epilepsy embedding extraction...")

dataloader_train = DataLoader(
    EpilepsyDataset(dataset_cfg, task_cfg, split='train'),
    batch_size=train_config['batch_size'],
    shuffle=train_config['shuffle']
)

dataloader_test = DataLoader(
    EpilepsyDataset(dataset_cfg, task_cfg, split='test'),
    batch_size=inference_config['batch_size'],
    shuffle=inference_config['shuffle']
)

P = Pipeline(MantisModel(device, '8M'))

#give MLPDecoder the smallest required info to use it
mlp_decoder = P.add_decoder(MLPDecoder(device, {'input_dim': 256, 'output_dim': 4, 'hidden_dim': 128}), load=True)

P.train(dataloader_train, parts_to_train=['decoder'], cfg=train_config)

y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
result = get_accuracy(y_test, y_pred)
print(result)
