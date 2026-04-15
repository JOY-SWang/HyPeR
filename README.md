# HyPeR --- Listen, Pause, and Reason: Toward Perception-Grounded Hybrid Reasoning for Audio Understanding

## Introduction

HyPeR is a perception-grounded audio reasoning framework built on top of `Qwen2-Audio-7B-Instruct`.  
Instead of directly mapping raw audio to text responses, HyPeR explicitly grounds reasoning in structured acoustic evidence and further improves internal deliberation with reinforcement learning.

Inspired by **Auditory Scene Analysis (ASA)**, HyPeR decomposes audio understanding into two key perceptual disambiguation stages:

- **Speech vs. Environment**: separating linguistic content from background sound and environmental interference.
- **Speaker vs. Speaker**: distinguishing multiple speakers and resolving attribution in complex conversations.

To support this setting, we introduce **PAQA**, a perception-aware audio question answering dataset with structured annotations including background sound analysis, speaker analysis, ASR evidence, reflection, and final answers. HyPeR is then trained in two stages:

1. **Stage I: Explicit Perception (SFT)**  
   The model is supervised to generate structured perceptual traces, including planning, captioning, reasoning, summary, and reflection.
2. **Stage II: Latent Reasoning (GRPO)**  
   The SFT-initialized model is further optimized with GRPO, where a special `<PAUSE>` token enables latent computation during acoustically ambiguous phases.

Our main findings are as follows:

- Perceptual errors are a major bottleneck for large audio language models, especially in noisy and multi-speaker settings.
- Explicitly grounding reasoning in acoustic evidence improves audio understanding more effectively than direct audio-to-text reasoning.
- A hybrid strategy that combines **explicit perception traces** and **latent reasoning with `<PAUSE>`** yields consistent gains over the base model and over standard GRPO without such grounding.
- HyPeR significantly improves robustness on difficult audio conditions, including background-rich and multi-speaker scenarios.
- On multiple benchmarks, HyPeR achieves strong performance competitive with much larger systems while using a 7B audio backbone.

### Highlights

- **Base model**: `Qwen2-Audio-7B-Instruct`
- **Dataset**: `PAQA` for supervised finetuning, plus augmented `AVQA` samples for RL training. 
- **Training recipe**: SFT + GRPO
- **Key mechanism**: `<PAUSE>`-based latent reasoning triggered under low confidence
- **Goal**: perception-grounded hybrid reasoning for robust audio understanding

## PAQA Dataset

PAQA is a benchmark and training dataset designed for **perception-grounded audio reasoning**.  
It contains **7,470 multiple-choice Audio-QA pairs**, each enriched with structured intermediate supervision such as:

- background sound / environment tags
- speaker-level analysis
- ASR evidence
- multi-step reasoning
- reflection and corrected final answer

PAQA supports tasks including:

- multi-speaker question answering
- noisy speech translation
- environment-centric question answering

The dataset is built from speech and audio sources including **MELD**, **CoVoST2**, synthetic multi-speaker QA data, and environmental sound resources such as **MUSAN** and **FSD50K**. See more cases in "/datasets/*.jsonl". [PAQA will be available on Hugging Face](https://huggingface.co/datasets/Joysw909/PAQA)

## Method Overview

HyPeR consists of two stages:

### Stage I: Explicit Perception via SFT

In the first stage, the model is trained to generate a structured reasoning trace:

- `<PLANNING>`
- `<CAPTION>`
  - `<ENV>`
  - `<ASR>`
  - `<SPEAKER>`
- `<REASONING>`
- `<SUMMARY>`
- `<REFLECT>`
- `<FINAL_ANSWER>`

This stage teaches the model to imitate a layered auditory decomposition process before answering.

### Stage II: GRPO-based Latent Reasoning

In the second stage, HyPeR is further optimized with **Group Relative Policy Optimization (GRPO)**.

A special `<PAUSE>` token is introduced to enable **latent reasoning** when the model encounters acoustically ambiguous cues such as:

- tone
- pitch
- background noise
- emotion
- overlap between speakers

HyPeR uses a confidence-based transition mechanism to determine when to:

- continue normal decoding,
- trigger `<PAUSE>` for additional internal computation,
- or abort an unstable trajectory.

The reward function jointly considers:

- **answer accuracy**
- **format compliance**
- **perceptual consistency**
- **length shaping**

## Main Results

### Table: Accuracy (%) on MMAU, MMAR, and MMSU

| Method | MMAU Test-mini Avg. | MMAU Test Avg. | MMAR Avg. | MMSU Avg. |
|--------|----------------------|----------------|-----------|-----------|
| Gemini 2.5 Flash | 64.30 | 64.68 | 66.80 | - |
| GPT-4o | 61.40 | 59.58 | 63.50 | 56.38 |
| Audio-Flamingo-3 | 73.30 | 72.42 | 58.50 | - |
| OmniVinci | 73.10 | 71.60 | 58.30 | - |
| Qwen2.5-Omni-7B | 71.50 | 71.00 | 56.70 | 60.57 |
| Qwen2-Audio-7B-Instruct | 54.30 | 48.65 | 30.00 | 48.31 |
| +SFT | 54.41 | 57.40 | 40.90 | 51.03 |
| +GRPO | 63.40 | 63.73 | 45.40 | 53.27 |
| +GRPO + ExpCoT | 65.90 | - | 48.20 | - |
| **Ours (HyPeR)** | **67.40** | **67.15** | **55.50** | **56.38** |
| Audio-CoT | 58.10 | - | 31.67 | - |
| Audio-Reasoner | 61.71 | 57.00 | 36.71 | 35.51 |
| Audio-Thinker | **68.00** | **67.90** | 52.00 | - |


## Additional Findings

- **Reflection helps**, but too many reflection rounds may lead to overthinking and worse results.
- **Background-sound awareness** improves robustness under noisy conditions.
- **Speaker analysis** is especially beneficial in multi-speaker scenarios.
- The **consistency reward** significantly improves grounding reliability.
- `<PAUSE>` tokens induce genuine latent computation rather than merely prolonging decoding: hidden states continue to evolve and gradually align with the final answer representation.

## Training

## Data Preparation

HyPeR uses different data sources in different stages:

### Stage I: Supervised Fine-Tuning

Use the **PAQA** dataset for explicit perception training.  
Each sample contains:

- audio input
- question
- multi-choice candidates
- structured perceptual reasoning target
- reflection
- final answer

A conceptual example is shown below:

```json
{
  "id": "sample_001",
  "audio_path": "path/to/audio.wav",
  "question": "Why does the woman refuse the meeting time?",
  "choices": [
    "She is unavailable on Friday night",
    "She does not hear the speaker",
    "She dislikes the song",
    "She has already arrived"
  ],
  "answer": 0,
  "env_tag": "Background music with repeated lyrics: 'Friday night'",
  "asr": "Speaker 1: Are you free Friday night? Speaker 2: I can't make it that evening.",
  "speaker_analysis": "Speaker 2 is female and responds with a negative tone.",
  "target": "<THINK>...</THINK><REFLECT>...</REFLECT><FINAL_ANSWER>A</FINAL_ANSWER>"
}
```

### Stage II: RL Optimization

For GRPO training, we use 30,000 augmented samples derived from the AVQA dataset, where responses are reformulated into a reasoning-answer structure such as:

```json
<think>...</think><answer>...</answer>
```

#### GRPO Training

```python
sh run_grpo.sh
```


#### Notes
	•	Replace the dataset paths in the training scripts with your local paths.
	•	If you already have the base model locally, modify the model path accordingly.
	•	We recommend using high-memory GPUs for training.
	•	PAUSE-based latent reasoning improves robustness, but also increases training and inference cost.



# Citation

@article{wang2026hyper,
  title={Listen, Pause, and Reason: Toward Perception-Grounded Hybrid Reasoning for Audio Understanding},
  author={Wang, Jieyi and Niu, Yazhe and Xu, Dexuan and Wei, Zhongyu},
  journal={arXiv preprint},
  year={2026}
}
