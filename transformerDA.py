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
vox = pd.read_csv("/tf/datasets/new_dev.tsv", sep="\t")

#test = test.sample(30000, replace=False)


vox = pd.concat([vox] * 4)
vox = vox.iloc[:82449]

vox.loc[vox['locale'] == 'ru', 'path'] = vox.loc[vox['locale'] == 'ru', 'path'].apply(lambda x: f"/tf/datasets/vox/ru_dev/{x}")


vox.loc[vox['locale'] == 'en', 'path'] = vox.loc[vox['locale'] == 'en', 'path'].apply(lambda x: f"/tf/datasets/vox/en_dev/{x}")


vox.loc[vox['locale'] == 'kz', 'path'] = vox.loc[vox['locale'] == 'kz', 'path'].apply(lambda x: f"/tf/datasets/vox/kz_dev/{x}")


vox.loc[(vox['locale'] != "kz") & (vox['locale'] != "ru") & (vox['locale']  != "en"), "path"] = "/tf/datasets/data_untar/cv-corpus-6.1-2020-12-11/" + vox.loc[(vox['locale'] != "kz") & (vox['locale'] != "ru") & (vox['locale']  != "en")]["locale"]  + "/clips/" + vox.loc[(vox['locale'] != "kz") & (vox['locale'] != "ru") & (vox['locale']  != "en")]["path"]


vox = vox.reset_index()

train["target_domain"] = vox['path']

meta = pd.concat([train, test, dev])



meta.loc[meta["locale"] != "kz", "path"] = "/tf/datasets/data_untar/cv-corpus-6.1-2020-12-11/" +  meta.loc[meta["locale"] != "kz"]["locale"] + "/clips/" + meta.loc[meta["locale"] != "kz"]["path"]
targets = {"kz": 0, "ru": 1, "en":2, "other":3}
meta["target"] = meta["locale"]
meta.loc[(meta["locale"] != "kz") & (meta["locale"] != "ru") & (meta["locale"]!="en"), "target"] = "other"
meta = meta.loc[meta["path"] != "/tf/datasets/data_untar/cv-corpus-6.1-2020-12-11/kz/clips/5f590a130a73c.mp3"]
meta = meta.loc[meta["path"] != "/tf/datasets/data_untar/cv-corpus-6.1-2020-12-11/kz/clips/5ef9bd9ba7029.mp3"]

meta["id"] = meta["Unnamed: 0"].apply(str)
meta["target"] = meta["target"].map(targets)

meta


from torch.utils.data import Dataset
import random
import math
DEVICE = 'cuda'
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
    def __init__(self, paths=None, targets=None, tg=None, augment=False) -> None:
        self.paths = paths
        self.targets = targets
        self.augment = augment
        self.target_domain = tg
        self.rir = get_rir_sample()[0]
        
    
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        y, sr = torchaudio.load(self.paths.iloc[idx], normalization=True)
        
        y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(y)
        y = torchaudio.transforms.Vad(sample_rate = 16000)(y)

     
        fixed_length = 16000 * 12
        
        # returning result
        if y.shape[1] < fixed_length:
            y = torch.nn.functional.pad(
              y, (0, fixed_length - y.shape[1]))
        else:
            y = y[:, :fixed_length]
        representation = y
        
        
        if self.target_domain is None:
            y = []
        else:
            y, sr = torchaudio.load(self.target_domain.iloc[idx], normalization=True)


            y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(y)
            y = torchaudio.transforms.Vad(sample_rate = 16000)(y)


            fixed_length = 16000 * 12

            # returning result
            if y.shape[1] < fixed_length:
                y = torch.nn.functional.pad(
                  y, (0, fixed_length - y.shape[1]))
            else:
                y = y[:, :fixed_length]
        
        result = {"target":self.targets.iloc[idx], "representation":representation, "target_domain": y}

        return result
    
    
from torch.utils.data import TensorDataset, DataLoader
train_ds = AudiosDataset(meta.loc[meta["split"]=="train"]["path"], meta.loc[meta["split"]=="train"]["target"],  meta.loc[meta["split"]=="train"]["target_domain"], augment=True)
val_ds = AudiosDataset(meta.loc[meta["split"]=="dev"]["path"], meta.loc[meta["split"]=="dev"]["target"])
test_ds = AudiosDataset(meta.loc[meta["split"]=="test"]["path"], meta.loc[meta["split"]=="test"]["target"])


batch_size = 8
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



DEVICE = 'cuda'
m = torch.hub.load('s3prl/s3prl', 'audio_albert').to(DEVICE)


from torch import nn
import torch.nn.functional as F
class Extractor(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor

        self.rnn = nn.LSTM(input_size=768, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.7)
        
    def forward(self, x):
        features = self.extractor(torch.squeeze(x))
        features = torch.stack(features)
        res, _ = self.rnn(features)
        return res[:, -1, :]
    
    
    
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.5),
                    nn.Linear(128, 4),
                    )
        
    def forward(self, x):
        
        res = self.classifier(x)
        return F.log_softmax(res, dim=1)
    
    
G = Extractor(m)
C1 = Classifier()
C2 = Classifier()


import torch.optim as optim
opt_g = optim.Adam(G.parameters(),
                                    lr=1e-4, weight_decay=0.0005)

opt_c1 = optim.Adam(C1.parameters(),
                                     lr=5e-5, weight_decay=0.0005)
opt_c2 = optim.Adam(C2.parameters(),
                                     lr=5e-5, weight_decay=0.0005)




from tqdm import tqdm


def discrepancy(out1, out2):
        return torch.mean(torch.abs((out1) - (out2)))

def reset_grad():
    opt_g.zero_grad()
    opt_c1.zero_grad()
    opt_c2.zero_grad()
    
    
    

def train(criterion, epochs, data_tr, data_val, max_stable=5, num_K = 1):
    best_val_loss = 1e9
    counter = 0
    
    for epoch in range(epochs):
        #tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        G.train()
        C1.train()
        C2.train()  # train mode
        for batch in tqdm(data_tr):
            loss = 0
            # data to device
            X_batch, Y_batch, targ = batch["representation"], batch["target"], batch["target_domain"]
            #print(X_batch.shape)
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            targ = targ.to(DEVICE)
            # set parameter gradients to zero
            reset_grad()
            # forward
            features = G(X_batch)
            output_s1 = C1(features)
            output_s2 = C2(features)

            loss_s1 = criterion(output_s1, Y_batch)
            loss_s2 = criterion(output_s2, Y_batch)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            opt_g.step()
            opt_c1.step()
            opt_c2.step()
            reset_grad()
            
            
            feat_s = G(X_batch)
            output_s1 = C1(feat_s)
            output_s2 = C2(feat_s)
            feat_t = G(targ)
            output_t1 = C1(feat_t)
            output_t2 = C2(feat_t)

            loss_s1 = criterion(output_s1, Y_batch)
            loss_s2 = criterion(output_s2, Y_batch)
            loss_s = loss_s1 + loss_s2
            loss_dis = discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            opt_c1.step()
            opt_c2.step()
            reset_grad()
            for i in range(num_K):
                #
                feat_t = G(targ)
                output_t1 = C1(feat_t)
                output_t2 = C2(feat_t)
                loss_dis = discrepancy(output_t1, output_t2)
                loss_dis.backward()
                opt_g.step()
                reset_grad()
        # show intermediate results
        G.eval()
        C1.eval()
        C2.eval() 
        val_loss = 0
        print("start validation")
        for v_b in tqdm(data_val):
            X_val, Y_val = v_b["representation"], v_b["target"]
            feat = G(X_val.to(DEVICE))
            pred_1 = C1(feat).detach().cpu()
            pred_2 = C2(feat).detach().cpu()
            val_loss += (criterion(pred_1, Y_val) + criterion(pred_2, Y_val)) / 2
        val_loss /= len(data_val)
        print( f"validation loss: {val_loss}")
        if val_loss <= best_val_loss and val_loss > 0:
            counter = 0
            print("Save new model!")
            best_val_loss = val_loss
            torch.save(G.state_dict(), f'extractor.h5')
            torch.save(C1.state_dict(), f'clf1.h5')
            torch.save(C2.state_dict(), f'clf2.h5')

        else:
            counter += 1
        if counter == max_stable:
            break
            
            
            
DEVICE = 'cuda'
max_epochs = 100
G = G.to(DEVICE)
C1 = C1.to(DEVICE)
C2 = C2.to(DEVICE)


loss_fn =  nn.CrossEntropyLoss()
train(loss_fn, max_epochs, loaders["train"], loaders["valid"])
