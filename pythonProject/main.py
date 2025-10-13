import torch, torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from models import Cnn10

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

meta_path = 'ESC-50/meta/esc50.csv'
audio_dir = 'ESC-50/audio'
device = "cuda" if torch.cuda.is_available() else "cpu"

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

if __name__=="__main__":
    print("Device:", device)

    model = Cnn10(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )
    checkpoint = torch.load("Cnn10_mAP=0.380.pth", map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.fc_audioset = nn.Linear(512, 50, bias=True)

    model = model.to(device)

    #training loop
    epochs = 100
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=1e-4)

    last_acc = 0
    saved = False
    test_fold = 0
    for epoch in range(epochs):
        #change folds every 20 epochs (100 epochs/5 folds)
        if (epoch+1)%20==0 or epoch==0:
            test_fold += 1
            # train on 80%
            train_folds = [1, 2, 3, 4, 5]
            train_folds.remove(test_fold)
            print("Training folds:",train_folds)
            train_dataset = ESC50Dataset(meta_path, folds=train_folds)
            # test on 20%
            test_dataset = ESC50Dataset(meta_path, folds=[test_fold])

            worker_count = os.cpu_count() // 2  # relegate half of cpu count to multiprocess
            train_loader = DataLoader(train_dataset,
                                      batch_size=32,
                                      num_workers=worker_count,
                                      shuffle=True,
                                      pin_memory=True)
            test_loader = DataLoader(test_dataset,
                                     batch_size=32,
                                     num_workers=worker_count,
                                     shuffle=True,
                                     pin_memory=True)
        model.train()
        running_correct = 0
        running_total = 0
        running_loss = 0

        for inputs, labels in train_loader:#runs (total_audio_samples/32 +-1)times
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimiser.zero_grad()
            output_logits = model(inputs)['clipwise_output']#raw calculation results (list of 50 probabilities)

            loss = loss_fn(output_logits, labels)
            running_loss += loss.item()
            loss.backward()

            optimiser.step()

            predictions = torch.argmax(output_logits,dim=1)
            running_correct += torch.eq(predictions, labels).sum().item()
            running_total += labels.size(0)

        acc = running_correct / running_total * 100
        print(f"Epoch {epoch+1} | Loss: {running_loss:.2f}, Accuracy: {acc:.2f}%")

        if (epoch+1)%10==0:#test every 10 epochs and save model
            with torch.inference_mode():
                running_test_correct = 0
                running_test_total = 0
                running_test_loss = 0

                for inputs, labels in test_loader:#runs (total_audio_samples/32 +-1)times
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    output_logits = model(inputs)['clipwise_output']#raw calculation results (list of 50 probabilities)

                    loss = loss_fn(output_logits, labels)
                    running_test_loss += loss.item()

                    predictions = torch.argmax(output_logits,dim=1)
                    running_test_correct += torch.eq(predictions, labels).sum().item()
                    running_test_total += labels.size(0)

                test_acc = running_test_correct/running_test_total * 100
                print(f"\n10th -> Epoch {epoch+1} | Test loss: {running_test_loss:.2f}, Test accuracy: {test_acc:.2f}%\n")

            #save checkpoint every 10 epochs
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': running_test_loss
            }

            # Save the checkpoint
            torch.save(checkpoint, f'created_models/checkpoint_epoch{epoch+1}.pth')

        if 0<acc-last_acc<0.5 and acc>95:
            print("Change in accuracy is negligible. Model training ended and saved.")
            saved = True
            break
        else:
            last_acc = acc

    torch.save(model, f='created_models/esc50model_final.pt')
    if not saved:
        print("100 epochs ran through, model saved")