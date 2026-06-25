import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# ========================================================
# 1. GLOBAL SETTINGS & DATA CONFIGURATION
# ========================================================
print("Verifying local Google Speech Commands dataset extraction...")
base_data_path = os.path.join(".", "data", "SpeechCommands", "speech_commands_v0.02")

if not os.path.exists(base_data_path):
    import torchaudio
    print("Downloading raw audio files (this may take a few minutes)...")
    torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=True)

LABELS = {"go": 0, "yes": 1, "no": 2}
REVERSE_LABELS = {0: "go", 1: "yes", 2: "no"}

mfcc_transform = T.MFCC(
    sample_rate=16000,
    n_mfcc=20,
    melkwargs={"n_fft": 1024, "hop_length": 160, "n_mels": 40}
)

# ========================================================
# 2. COMPETITION PIPELINE DATASET
# ========================================================
class AudioTrainDataset(Dataset):
    def __init__(self, base_dir, labels_dict):
        self.file_paths = []
        self.targets = []
        for word, label_idx in labels_dict.items():
            word_dir = os.path.join(base_dir, word)
            if os.path.exists(word_dir):
                files = sorted([f for f in os.listdir(word_dir) if f.endswith('.wav')])
                train_files = files[:-50]
                for f in train_files:
                    self.file_paths.append(os.path.join(word_dir, f))
                    self.targets.append(label_idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data, sample_rate = sf.read(self.file_paths[idx])
        waveform = torch.FloatTensor(data).unsqueeze(0)

        if waveform.shape[1] < 16000:
            waveform = nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]

        mfcc = mfcc_transform(waveform).squeeze(0)
        return mfcc.t(), self.targets[idx]  # Shape: [Time_Steps, 20]

# ========================================================
# 3. DEFINE VANILLA RNN MODEL
# ========================================================
class AudioRNN(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_classes=3):
        super().__init__()
        # nn.RNN processes the sequence one time-step at a time
        # batch_first=True means input shape is [batch, time_steps, features]
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # Final linear layer maps the last hidden state to one of 3 word classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape:      [batch_size, time_steps, 20]
        # output shape: [batch_size, time_steps, hidden_size]  (every time-step)
        # h_n shape:    [1, batch_size, hidden_size]           (just the final step)
        output, h_n = self.rnn(x)

        # We only care about the LAST time-step's hidden state
        # (it has "seen" the whole word by that point)
        last_hidden = h_n.squeeze(0)   # [batch_size, hidden_size]
        return self.fc(last_hidden)    # [batch_size, 3]

# ========================================================
# 4. TRAINING LOOP & STABILIZATION
# ========================================================
train_dataset = AudioTrainDataset(base_data_path, LABELS)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = AudioRNN(input_size=20, hidden_size=64, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}  (limit: 50,000)")

NUM_EPOCHS = 15
print("\nStarting training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_mfcc, batch_labels in train_loader:
        # 1. Clear old gradients from the previous step
        optimizer.zero_grad()

        # 2. Forward pass — model makes its predictions
        outputs = model(batch_mfcc)

        # 3. Calculate how wrong the predictions were
        loss = criterion(outputs, batch_labels)

        # 4. Backward pass — figure out which way to adjust each parameter
        loss.backward()

        # 5. Safety cap so gradients don't explode
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Actually update the model's parameters
        optimizer.step()

        # Track stats
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")

print("Training complete!")

# ========================================================
# 5. INFERENCE ON PACKAGED LEADERBOARD FILE
# ========================================================
print("\nRunning inference calculations on 'student_test_features.pt'...")

if not os.path.exists("student_test_features.pt"):
    raise FileNotFoundError("Please ensure the instructor's 'student_test_features.pt' file is in this folder!")

X_evaluation = torch.load("student_test_features.pt")  # Shape: [150, Time_Steps, 20]

# Switch model to evaluation mode (disables dropout etc. if present)
model.eval()
predictions = []

with torch.no_grad():  # Don't waste memory tracking gradients during inference
    for idx in range(X_evaluation.shape[0]):
        single_clip = X_evaluation[idx].unsqueeze(0)   # [1, time_steps, 20]
        output = model(single_clip)                     # [1, 3]
        predicted_idx = output.argmax(dim=1).item()     # 0, 1, or 2
        predictions.append(REVERSE_LABELS[predicted_idx])  # "go", "yes", or "no"

with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'keyword_class'])
    for idx, word in enumerate(predictions):
        writer.writerow([idx, word])

print("✅ predictions.csv successfully generated!")
print("Submit your 'predictions.csv' along with this source code script to the leaderboard portal.")