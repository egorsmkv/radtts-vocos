import os
from glob import glob
from sys import platform

import torch
from scipy.io.wavfile import write


def inference(vocoder, input_mel_folder):
    with torch.no_grad():
        files_all = []
        for input_mel_file in glob(input_mel_folder +'/*.mel'):
            x = torch.load(input_mel_file)
            audio = vocoder.decode(x)

            audio = audio.cpu().numpy().astype('int16')

            output_file = input_mel_file.replace('.mel','.wav')
            write(output_file, 22_050, audio)
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
