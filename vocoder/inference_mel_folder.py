import os
from glob import glob
from sys import platform

import torch
import torchaudio


def inference(vocoder, input_mel_folder):
    with torch.no_grad():
        files_all = []
        for input_mel_file in glob(input_mel_folder +'/*.mel'):
            x = torch.load(input_mel_file)
            audio = vocoder.decode(x)

            output_file = input_mel_file.replace('.mel','.wav')

            torchaudio.save(output_file, audio, 22_050)

            print('<<--', output_file)

            files_all.append(output_file)

            os.remove(input_mel_file)

        s = '/'
        if platform == "win32":
            s = 'results\\'

        names = []
        for k in files_all:
            names.append(int(k.replace(input_mel_folder, '').replace(s, '').replace('.wav', '')))

        names_w = [f'{it}.wav' for it in sorted(names)]

        print('sox ' + ' '.join(names_w) + ' all.wav')


def process_folder(input_mel_folder, vocoder):
    print('Initializing Inference Process..')

    inference(vocoder, input_mel_folder)
