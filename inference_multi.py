# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import os
import json

import torch
import torchaudio
from torch.cuda import amp

from radtts import RADTTS
from vocos import Vocos
from data import Data
from common import update_params


def lines_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def infer(radtts_path, vocoder_path, vocoder_config_path, text_path, speaker,
          speaker_text, speaker_attributes, sigma, sigma_tkndur, sigma_f0,
          sigma_energy, f0_mean, f0_std, energy_mean, energy_std,
          token_dur_scaling, denoising_strength, n_takes, output_dir, use_amp,
          plot, seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    vocoder = Vocos.from_pretrained("BSC-LT/vocos-mel-22khz")

    radtts = RADTTS(**model_config).cuda()
    radtts.enable_inverse_cache() # cache inverse matrix for 1x1 invertible convs

    checkpoint_dict = torch.load(radtts_path, map_location='cpu')
    state_dict = checkpoint_dict['state_dict']
    radtts.load_state_dict(state_dict, strict=False)
    radtts.eval()
    print("Loaded checkpoint '{}')" .format(radtts_path))

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    speaker_id = trainset.get_speaker_id(speaker).cuda()
    speaker_id_text, speaker_id_attributes = speaker_id, speaker_id
    if speaker_text is not None:
        speaker_id_text = trainset.get_speaker_id(speaker_text).cuda()
    if speaker_attributes is not None:
        speaker_id_attributes = trainset.get_speaker_id(
            speaker_attributes).cuda()

    text_list = lines_to_list(text_path)

    os.makedirs(output_dir, exist_ok=True)
    for i, text in enumerate(text_list):
        if text.startswith("#"):
            continue
        print("{}/{}: {}".format(i, len(text_list), text))
        text = trainset.get_text(text).cuda()[None]
        for take in range(n_takes):
            with amp.autocast(use_amp):
                with torch.no_grad():
                    outputs = radtts.infer(
                        speaker_id, text, sigma, sigma_tkndur, sigma_f0,
                        sigma_energy, token_dur_scaling, token_duration_max=100,
                        speaker_id_text=speaker_id_text,
                        speaker_id_attributes=speaker_id_attributes,
                        f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean,
                        energy_std=energy_std)

                    audio = vocoder.decode(outputs['mel'])

                    suffix_path = "{}_{}_{}_durscaling{}_sigma{}_sigmatext{}_sigmaf0{}_sigmaenergy{}".format(
                        i, take, speaker, token_dur_scaling, sigma, sigma_tkndur, sigma_f0,
                        sigma_energy)

                    filename_w = "{}/{}_{}.wav".format(
                        output_dir, suffix_path, denoising_strength)

                    torchaudio.save(filename_w, audio, 22_050)
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file config')
    parser.add_argument('-k', '--config_vocoder', type=str, help='vocoder JSON file config')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-r', '--radtts_path', type=str)
    parser.add_argument('-t', '--text_path', type=str)
    parser.add_argument('-s', '--speaker', type=str)
    parser.add_argument('--speaker_text', type=str, default=None)
    parser.add_argument('--speaker_attributes', type=str, default=None)
    parser.add_argument('-d', '--denoising_strength', type=float, default=0.0)
    parser.add_argument('-o', "--output_dir", default="results")
    parser.add_argument("--sigma", default=0.8, type=float, help="sampling sigma for decoder")
    parser.add_argument("--sigma_tkndur", default=0.666, type=float, help="sampling sigma for duration")
    parser.add_argument("--sigma_f0", default=1.0, type=float, help="sampling sigma for f0")
    parser.add_argument("--sigma_energy", default=1.0, type=float, help="sampling sigma for energy avg")
    parser.add_argument("--f0_mean", default=0.0, type=float)
    parser.add_argument("--f0_std", default=0.0, type=float)
    parser.add_argument("--energy_mean", default=0.0, type=float)
    parser.add_argument("--energy_std", default=0.0, type=float)
    parser.add_argument("--token_dur_scaling", default=1.00, type=float)
    parser.add_argument("--n_takes", default=1, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    infer(args.radtts_path, args.vocoder_path, args.config_vocoder,
          args.text_path, args.speaker, args.speaker_text,
          args.speaker_attributes, args.sigma, args.sigma_tkndur, args.sigma_f0,
          args.sigma_energy, args.f0_mean, args.f0_std, args.energy_mean,
          args.energy_std, args.token_dur_scaling, args.denoising_strength,
          args.n_takes, args.output_dir, args.use_amp, args.plot, args.seed)
