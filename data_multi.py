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

# Based on https://github.com/NVIDIA/flowtron/blob/master/data.py
# Original license text:
###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import os

import lmdb

import torch
import torch.utils.data
import numpy as np

from tts_text_processing.text_processing import TextProcessing


class Data(torch.utils.data.Dataset):
    def __init__(self, datasets, filter_length, hop_length, win_length,
                 sampling_rate, n_mel_channels, mel_fmin, mel_fmax, f0_min,
                 f0_max, max_wav_value, use_f0, use_energy_avg, use_log_f0,
                 use_scaled_energy, symbol_set, cleaner_names, heteronyms_path,
                 phoneme_dict_path, p_phoneme, handle_phoneme='word',
                 handle_phoneme_ambiguous='ignore', speaker_ids=None,
                 include_speakers=None, n_frames=-1,
                 use_attn_prior_masking=True, prepend_space_to_text=True,
                 append_space_to_text=True, add_bos_eos_to_text=False,
                 betabinom_cache_path="", betabinom_scaling_factor=0.05,
                 lmdb_cache_path="", dur_min=None, dur_max=None,
                 combine_speaker_and_emotion=False, **kwargs):

        self.combine_speaker_and_emotion = combine_speaker_and_emotion
        self.audio_lmdb_dict = {}

        self.tp = TextProcessing(
            symbol_set, cleaner_names, heteronyms_path, phoneme_dict_path,
            p_phoneme=p_phoneme, handle_phoneme=handle_phoneme,
            handle_phoneme_ambiguous=handle_phoneme_ambiguous,
            prepend_space_to_text=prepend_space_to_text,
            append_space_to_text=append_space_to_text,
            add_bos_eos_to_text=add_bos_eos_to_text)
        
        self.data = self.load_data(datasets)

        self.speaker_map = None
        if 'speaker_map' in kwargs:
            self.speaker_map = kwargs['speaker_map']

        if speaker_ids is None or speaker_ids == '':
            self.speaker_ids = self.create_speaker_lookup_table(self.data)
        else:
            self.speaker_ids = speaker_ids

    def get_text(self, text):
        return torch.LongTensor(self.tp.encode_text(text))

    def load_data(self, datasets, split='|'):
        dataset = []

        for dset_name, dset_dict in datasets.items():
            folder_path = dset_dict['basedir']
            audiodir = dset_dict['audiodir']
            filename = dset_dict['filelist']
            audio_lmdb_key = None
            if 'lmdbpath' in dset_dict.keys() and len(dset_dict['lmdbpath']) > 0:
                self.audio_lmdb_dict[dset_name] = lmdb.open(
                    dset_dict['lmdbpath'], readonly=True, max_readers=256,
                    lock=False).begin()
                audio_lmdb_key = dset_name

            wav_folder_prefix = os.path.join(folder_path, audiodir)
            filelist_path = os.path.join(folder_path, filename)
            with open(filelist_path, encoding='utf-8') as f:
                data = [line.strip().split(split) for line in f]

            for d in data:
                emotion = 'other' if len(d) == 3 else d[3]
                duration = -1 if len(d) == 3 else d[4]
                dataset.append({
                    'audiopath': os.path.join(wav_folder_prefix, d[0]),
                    'text': d[1],
                    'speaker': d[2] + '-' + emotion if self.combine_speaker_and_emotion else d[2],
                    'emotion': emotion,
                    'duration': float(duration),
                    'lmdb_key': audio_lmdb_key
                })

        return dataset

    def get_speaker_id(self, speaker):
        if self.speaker_map is not None and speaker in self.speaker_map:
            speaker = self.speaker_map[speaker]

        return torch.LongTensor([self.speaker_ids[speaker]])

    def create_speaker_lookup_table(self, data):
        speaker_ids = np.sort(np.unique([x['speaker'] for x in data]))

        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}

        print("Number of speakers:", len(d))
        print("Speaker IDS", d)

        return d
