import torch
from torchsummary import summary
import librosa
import numpy as np

import os
import time
import hashlib
import soundfile
from pathlib import Path

from models.Generator import Generator
from models.Discriminator import Discriminator


def save_sounds(path, sounds, sampling_rate):
    now_time = time.time()
    for i, sound in enumerate(sounds):
        sound = sound.squeeze(0)
        sound = sound.to('cpu').detach().numpy().copy()
        hash_string = hashlib.md5(str(now_time).encode()).hexdigest()
        file_path = os.path.join(
            path, f"generated_sound_{i}_{hash_string}.wav")
        print(file_path)
        soundfile.write(file_path, sound, sampling_rate, format="WAV")


model = Generator()
latent_dim = 100
z = torch.rand(latent_dim, dtype=torch.float32)
output = model.forward(z)
output = output.squeeze(0).squeeze(0).detach().numpy().copy()
S = librosa.feature.inverse.mel_to_stft(output)
y = librosa.griffinlim(S)
print(y.shape)
soundfile.write('./output/result.wav', y, 22050, format="WAV")

# output_dir = "./output"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# sampling_rate = 16000
# save_sounds("./output/", [output], sampling_rate)


# load audio data
# DATA_DIR = Path("./data/piano/train")
# wavfiles = list(DATA_DIR.glob("*.wav"))
# y, sr = librosa.load(wavfiles[0], duration=10)
# y = librosa.feature.melspectrogram(y=y, sr=sr)
# x = torch.from_numpy(y[:sr*5].astype(np.float32)).clone()

# d = Discriminator()
# print("training...")
# output = d.forward(x.unsqueeze(0).unsqueeze(0))
# print(output)
