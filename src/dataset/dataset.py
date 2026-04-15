import json
import logging

import torchaudio
import numpy as np
import torch
from torch.utils.data import Dataset

# The following imports are not used in this file and can be removed:
# import os
# import random
# import librosa
# import soundfile as sf
# import torchaudio.transforms as T

def _handle_wav(wav_path, target_rate=16000):
    """
    handle one wav file.
    Return:
        waveform: numpy narray(1d)
    """
    try:
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.numel() == 0:
            raise ValueError(f"Empty audio file: {wav_path}")
            
        # Ensure audio is mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sample_rate != target_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)(waveform)
            
        audio = waveform[0]
        
        # Ensure audio length is not too short
        if audio.numel() < 100:  # Minimum length check
            raise ValueError(f"Audio file too short: {wav_path}")
            
        # Normalize audio
        audio = audio / (torch.abs(audio).max() + 1e-8)

        return audio
    except Exception as e:
        logging.error(f"Error processing audio file {wav_path}: {str(e)}")
        raise

def _handle_avqa2(obj_avqa):
    choice_str = f"Please choose the answer from the following options: {obj_avqa['multi_choice']}."
    question_template = f'''{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output strictly follow the following format:
<THINK>
<PLANNING> (Step by step plan here) </PLANNING>
<CAPTION>
    <BGM>(Analysis the background music)</BGM>
    <SPEAKER>(List each speaker in order, concise summary) </SPEAKER>
    <ASR>(Raw transcript, only English)</ASR>
    <DESCRIPTION>(Overall analysis, link to question) </DESCRIPTION>
</CAPTION>
<REASONING>(Start reasoning here) </REASONING>
<SUMMARY>(Summarize reasoning results) </SUMMARY>
</THINK>
<RESPONSE>(Final answer, short and clear) </RESPONSE>
(<REFLECT>(structured reflection) </REFLECT> # optional)
(<FINAL_ANSWER>(Corrected, high-quality final answer) </FINAL_ANSWER> # optional)'''
    # If you want to improve the thinking process, uncomment the next line and design your strategy.
    # question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    obj_avqa["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_avqa["audio_path"]}, {"type": "text", "text": question_template}]}]
    mapping = {0:'A', 1:'B', 2:'C', 3:'D'}
    answer_str = mapping.get(obj_avqa["answer"], "") + " " + obj_avqa["multi_choice"][obj_avqa["answer"]]
    obj_avqa["solution"] = f"{answer_str}"
    return obj_avqa

def _handle_avqa(obj_avqa):
    choice_str = f"Please choose the answer from the following options: {obj_avqa['multi_choice']}."
    question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output the final answer in <answer> </answer>."
    # If you want to improve the thinking process, uncomment the next line and design your strategy.
    # question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    obj_avqa["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_avqa["audio_path"]}, {"type": "text", "text": question_template}]}]
    answer_str = obj_avqa["multi_choice"][obj_avqa["answer"]]
    obj_avqa["solution"] = f"<answer>{answer_str}</answer>"
    return obj_avqa

def _handle_multiSpeaker(obj_ms):
    choice_str = f"Please choose the answer from the following options: {obj_ms['multi_choice']}."
    # question_template = f"{obj_ms['question_text'].replace('video', 'audio')} {choice_str} Output the final answer in <answer> </answer>."
    # If you want to improve the thinking process, uncomment the next line and design your strategy.
    question_template = f"""Below is a multiple-choice question based on a short audio clip. 
Pick the single best answer. Return BOTH the letter and the chosen choice text. Do not output anything else outside the required tags.

Question: {obj_ms['instruction'][7:]}

Output strictly follow the format below.
(Do NOT invent any transcript. Always copy the text exactly as provided inside <BGM> and <ASR>.)

<THINK>
<PLANNING>(Step-by-step plan)</PLANNING>
<CAPTION>
    <BGM>{obj_ms['bgm']}</BGM>
    <SPEAKER>(List each speaker in order, concise summary)</SPEAKER>
    <ASR>Raw transcript: {obj_ms['asr']}</ASR>
    <DESCRIPTION>(Overall analysis linked to the question)</DESCRIPTION>
</CAPTION>
<REASONING>(Brief reasoning leading to one option)</REASONING>
<SUMMARY>(One sentence summary of the decision)</SUMMARY>
</THINK>

<RESPONSE>
<LETTER>(A|B|C|D)</LETTER>
<TEXT>(Paste the exact chosen option text)</TEXT>
</RESPONSE>
""" # <FINAL_ANSWER>(A|B|C|D). (Paste the exact chosen option text)</FINAL_ANSWER>
    obj_ms["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_ms["audio_path"]}, {"type": "text", "text": question_template}]}]
    # answer_str = obj_ms["multi_choice"][obj_ms["answer"]]
    answer_str = obj_ms["golden"]
    obj_ms["solution"] = f"{answer_str}"
    return obj_ms

def handle_json_line(json_line, sample_rate=16000):
    obj = json.loads(json_line)
    waveform = _handle_wav(obj["audio_path"], sample_rate)
    audio = waveform.numpy().astype(np.float32)
        
    # Ensure audio is not empty
    if audio.size == 0:
        raise ValueError(f"Empty audio after processing: {obj['audio_path']}")
        
    obj["audio"] = audio

    if obj["dataset_name"] == "AVQA":
        return _handle_avqa2(obj)
    elif obj["dataset_name"] == "multispeaker" or obj["dataset_name"] == "MELD":
        return _handle_multiSpeaker(obj)
    
    return obj


class AudioDataset(Dataset):
    def __init__(self, data_file, sample_rate=16000, is_perturb=False):
        super().__init__()
        self.lists = []
        with open(data_file, 'r', encoding='utf8') as fin:
            for line in fin:
                self.lists.append(line)

        self.sample_rate = sample_rate
        self.is_perturb = is_perturb
        logging.info(f"{data_file}, len:{len(self.lists)}, rate:{sample_rate}")

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        return handle_json_line(self.lists[index], self.sample_rate)