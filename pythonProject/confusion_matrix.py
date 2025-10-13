import torch, torchaudio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd


model = torch.load('created_models/esc50model_final.pt', weights_only=False)
model.eval()

class ESC50Dataset(Dataset):
    def __init__(self, meta_path, folds=[1], esc10=False):
        self.meta_df = pd.read_csv(meta_path)
        if esc10:
            self.meta_df = self.meta_df[self.meta_df['esc10']==True]
        #filter out unwanted folds
        self.meta_df = self.meta_df[self.meta_df['fold'].isin(folds)].reset_index(drop=True)
        self.audio_dir = 'ESC-50/audio'

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, item):
        row = self.meta_df.iloc[item]
        waveform, _ = torchaudio.load(os.path.join(self.audio_dir,row['filename']))
        waveform = waveform.squeeze(0)
        label = row['target']

        return waveform, label

def generate_confusion_matrix(model, device):
    all_preds = []
    all_labels = []

    test_dataset = ESC50Dataset('ESC-50/meta/esc50.csv', folds=[1])

    worker_count = os.cpu_count() // 2  # relegate half of cpu count to multiprocess
    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             num_workers=worker_count,
                             shuffle=True,
                             pin_memory=True)

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)['clipwise_output']
            predictions = torch.argmax(outputs,dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

if __name__=="__main__":
    generate_confusion_matrix(model, "cuda")

