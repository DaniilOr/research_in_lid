import torch
import torchaudio

import numpy as np
np_rng = np.random.default_rng(1)
import pandas as pd


import urllib.parse
from IPython.display import display, Markdown

import os

from lidbox.meta import (
    common_voice,
    generate_label2target,
    verify_integrity,
    read_audio_durations,
    random_oversampling_on_split
)


train = pd.read_csv("/tf/datasets/train.tsv", sep="\t")
test = pd.read_csv("/tf/datasets/test.tsv", sep="\t")
dev = pd.read_csv("/tf/datasets/dev.tsv", sep="\t")

train["path"] = train["path"].apply(lambda x: x[:-3] + "mp3")
test["path"] = test["path"].apply(lambda x: x[:-3] + "mp3")
dev["path"] = dev["path"].apply(lambda x: x[:-3] + "mp3")

train["split"] = "train"
test["split"] = "test"
dev["split"] = "dev"
#test = test.sample(30000, replace=False)
meta = pd.concat([train, test, dev])


meta.loc[meta["locale"] != "kz", "path"] = "/tf/datasets/data_untar/cv-corpus-6.1-2020-12-11/" +  meta.loc[meta["locale"] != "kz"]["locale"] + "/clips/" + meta.loc[meta["locale"] != "kz"]["path"]
targets = {"kz": 0, "ru": 1, "en":2, "other":3}
meta["target"] = meta["locale"]
meta.loc[(meta["locale"] != "kz") & (meta["locale"] != "ru") & (meta["locale"]!="en"), "target"] = "other"
meta = meta.loc[meta["path"] != "/tf/datasets/data_untar/cv-corpus-6.1-2020-12-11/kz/clips/5f590a130a73c.mp3"]
meta = meta.loc[meta["path"] != "/tf/datasets/data_untar/cv-corpus-6.1-2020-12-11/kz/clips/5ef9bd9ba7029.mp3"]

meta["id"] = meta["Unnamed: 0"].apply(str)
meta["target"] = meta["target"].map(targets)


from torch.utils.data import Dataset
import random
import math

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.append(["rate", f'{resample}'])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

SAMPLE_RIR_PATH = os.path.join(os.getcwd(), "rir.wav")

def get_rir_sample(*, resample=None, processed=False):
    rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate

class AudiosDataset(Dataset):
    def __init__(self, paths=None, targets=None, augment=False) -> None:
        self.paths = paths
        self.targets = targets
        self.augment = augment
        self.rir = get_rir_sample()[0]
        
    
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        y, sr = torchaudio.load(self.paths.iloc[idx], normalization=True)
        """
        if self.augment:
            effects = [
                    ["lowpass", "-1", "300"], 
                    ["speed", f"{random.uniform(0.7, 1.3)}"],  # change speed
                  ]
            y, sr = torchaudio.sox_effects.apply_effects_tensor(
                y, sr, effects)
        #
        """    
        if self.augment:
            # augment sound in order to imitate the room change
            rir = self.rir[:, int(16000*1.1):int(16000*1.3)]
            rir = rir / torch.norm(rir, p=2)
            rir = torch.flip(rir, [1])
            y = torch.nn.functional.conv1d(y[None, ...], rir[None, ...])[0]
        
        
        y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(y)
        #y = torchaudio.transforms.Vad(sample_rate = 16000)(y)

        
        if self.augment:
            y = (0.5)*torch.randn(y.shape)
        
        # convert to spectogram
        spectogram = torchaudio.transforms.MelSpectrogram()(y)
        #spectogram = torch.log(spectogram + 1e-5)
        melspectogram_db = torchaudio.transforms.AmplitudeToDB()(spectogram)
        
        #Make sure all spectrograms are the same size
        fixed_length = 12 * (16000//200)
        
        if melspectogram_db.shape[2] < fixed_length:
            melspectogram_db = torch.nn.functional.pad(
              melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
        else:
            melspectogram_db = melspectogram_db[:, :, :fixed_length]
        
        spectogram = melspectogram_db
        
        if self.augment:

            spectogram = torchaudio.transforms.FrequencyMasking(100)(spectogram)
            spectogram = torchaudio.transforms.TimeMasking(100)(spectogram)
        
        # returning result
        result = {"spec": spectogram, "target":self.targets.iloc[idx]}

        return result
    
ds = AudiosDataset(meta["path"], meta["target"])
from torch.utils.data import TensorDataset, DataLoader
train_ds = AudiosDataset(meta.loc[meta["split"]=="train"]["path"], meta.loc[meta["split"]=="train"]["target"], augment=True)
val_ds = AudiosDataset(meta.loc[meta["split"]=="dev"]["path"], meta.loc[meta["split"]=="dev"]["target"])
test_ds = AudiosDataset(meta.loc[meta["split"]=="test"]["path"], meta.loc[meta["split"]=="test"]["target"])

batch_size = 32
num_workers = 10
loaders = {
    "train": DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    ),
    "valid": DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    ),
    "test":DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    ),
}

from torchvision import  models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
model = models.resnet34(pretrained=True)
model.conv1=nn.Conv2d(1, model.conv1.out_channels, 
                      kernel_size=model.conv1.kernel_size[0], 
                      stride=model.conv1.stride[0], 
                      padding=model.conv1.padding[0])
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)


from tqdm import tqdm

from tqdm import tqdm

def train(model, opt, scheduler, loss_fn, epochs, data_tr, data_val, max_stable=5):
    best_val_loss = 1e9
    counter = 0
    for epoch in range(epochs):
        #tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for batch in tqdm(data_tr):
            loss = 0
            # data to device
            X_batch, Y_batch = batch["spec"], batch["target"]
            #print(X_batch.shape)
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            # set parameter gradients to zero
            opt.zero_grad()
            # forward
            Y_pred = model(X_batch)
            #print(Y_pred)
            loss = loss_fn(Y_pred, Y_batch)# forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights
            if not scheduler is None:
                scheduler.step()
            # calculate loss to show the user
            avg_loss += loss / len(data_tr)
      #  toc = time()
        print('loss: %f' % avg_loss)
        # show intermediate results
        model.eval()  # testing mode
        val_loss = 0
        print("start validation")
        for v_b in tqdm(data_val):
            X_val, Y_val = v_b["spec"], v_b["target"]
            Y_hat = model(X_val.to(DEVICE)).detach().cpu()# detach and put into cpu
            val_loss += loss_fn(Y_hat, Y_val)
        val_loss /= len(data_val)
        print( f"validation loss: {val_loss}")
        if val_loss <= best_val_loss and val_loss > 0:
            counter = 0
            print("Save new model!")
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model.h5')
            best_patn = f'{epoch}_{best_val_loss}.h5'
        else:
            counter += 1
        if counter == max_stable:
            break
            
            
            
DEVICE = 'cuda'
max_epochs = 100
model = model.to(DEVICE)
model.load_state_dict(torch.load("best_model.h5"))
#torch.cuda.empty_cache()
loss_fn =  nn.CrossEntropyLoss()
optimaizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimaizer, T_0=5, T_mult=1, eta_min=1e-8, last_epoch=-1)
train(model, optimaizer,scheduler, loss_fn, max_epochs, loaders["train"], loaders["valid"])

model.load_state_dict(torch.load("best_model.h5"))

true_labels = []
predicted_labels = []
for batch in loaders["test"]:
    model.eval()
    prediction = model(batch["spec"].to(DEVICE)).detach().cpu()
    predicted_labels.extend(torch.argmax(prediction, dim=1).tolist())
    true_labels.extend(batch["target"].tolist())
from sklearn.metrics import classification_report

report = classification_report(true_labels, predicted_labels, target_names=list(targets.keys()), labels=range(4))
print(report)

