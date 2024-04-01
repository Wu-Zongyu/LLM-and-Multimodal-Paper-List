# LLM-and-VLM-Paper-List
A paper list about large language models and multi-modal models.    
**Note:** It only records papers for my personal needs. It is welcome to open an issue if you think I missed some important or exciting work!

## Table of Contents

- [Survey](#survey)
- [Language Model](#language-model)
  - [Foundation LM Models](#foundation-lm-models)
  - [Reinforcement Learning from Human Feedback (RLHF)](#rlhf)
  - [Parameter Efficient Fine-tuning](#parameter-efficient-fine-tuning)
  - [Healthcare LLM](#healthcare-lm)
  - [Watermarking LLM](#watermarking-llm)
- [Multi-Modal Models](#multi-modal-models)
  - [Foundation Multi-Modal Models](#foundation-multi-modal-models)
  - [Multi-Modal Safety](#multi-modal-safety)
  - [VLM Hullucinations](#vlm-hullucinations)
  - [VLM Privacy](#vlm-privacy)
- [Agent](#agent)
- [Useful Resource](#useful-resource)

## Survey
- HELM: **Holistic evaluation of language models**. TMLR. [paper](https://arxiv.org/abs/2211.09110)
- HEIM: **Holistic Evaluation of Text-to-Image Models**. NeurIPS'2023. [paper](https://arxiv.org/abs/2311.04287)
- Eval Survey: **A Survey on Evaluation of Large Language Models**. Arxiv'2023. [paper](https://arxiv.org/abs/2307.03109)
- Healthcare LM Survey: **A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics**. Arxiv'2023. [paper](https://arxiv.org/abs/2310.05694), [github](https://github.com/KaiHe-better/LLM-for-Healthcare)
- Multimodal LLM Survey: **A Survey on Multimodal Large Language Model**. Arxiv'2023. [paper](https://arxiv.org/pdf/2306.13549.pdf), [github](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- VLM for vision Task Survey: **Vision Language Models for Vision Tasks: A Survey**. Arxiv'2023. [paper](https://arxiv.org/abs/2304.00685), [github](https://github.com/jingyi0000/VLM_survey)
- Efficient LLM Survey: **Efficient Large Language Models: A Survey**. Arxiv'2023. [paper](https://arxiv.org/abs/2312.03863), [github](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)
- Prompt Engineering Survey: **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing**. Arxiv'2021. [paper](https://arxiv.org/abs/2107.13586)
- Multimodal Safety Survey: **Safety of Multimodal Large Language Models on Images and Text**. Arxiv'2024. [paper](https://arxiv.org/abs/2402.00357)
- Multimodal LLM Recent Survey: **MM-LLMs: Recent Advances in MultiModal Large Language Models**. Arxiv'2024. [paper](https://arxiv.org/abs/2401.13601)
- Prompt Engineering in LLM Survey: **A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications**. Arxiv'2024. [paper](https://arxiv.org/abs/2402.07927)
- LLM Security and Privacy Survey: **A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**. Arxiv'2024. [paper](https://arxiv.org/abs/2312.02003)
- LLM Privacy Survey: **Privacy in Large Language Models: Attacks, Defenses and Future Directions**. Arxiv'2023. [paper](https://arxiv.org/abs/2310.10383)
---

## Language Model
### Foundation LM Models
- Transformer: **Attention Is All You Need**. NIPS'2017. [paper](https://arxiv.org/abs/1706.03762)
- GPT-1: **Improving Language Understanding by Generative Pre-Training**. 2018. [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- BERT: **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL'2019. [paper](https://aclanthology.org/N19-1423.pdf)
- GPT-2: **Language Models are Unsupervised Multitask Learners**. 2018. [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- RoBERTa: **RoBERTa: A Robustly Optimized BERT Pretraining Approach**. Arxiv'2019, [paper](https://arxiv.org/abs/1907.11692)
- DistilBERT: **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**. Arxiv'2019. [paper](https://arxiv.org/abs/1910.01108)
- T5: **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**. JMLR'2020. [paper](https://arxiv.org/abs/1910.10683)
- GPT-3: **Language Models are Few-Shot Learners**. NeurIPS'2020. [paper](https://arxiv.org/abs/2005.14165)
- GLaM: **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts**. ICML'2022. [paper](https://arxiv.org/abs/2112.06905)
- PaLM: **PaLM: Scaling Language Modeling with Pathways**. ArXiv'2022. [paper](https://arxiv.org/abs/2204.02311)
- BLOOM:  **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model**. Arxiv'2022. [paper](https://arxiv.org/abs/2211.05100)
- BLOOMZ: **Crosslingual Generalization through Multitask Finetuning**. Arxiv'2023. [paper](https://arxiv.org/abs/2211.01786)
- LLaMA: **LLaMA: Open and Efficient Foundation Language Models**. Arxiv'2023. [paper](https://arxiv.org/abs/2302.13971)
- GPT-4: **GPT-4 Technical Report**. Arxiv'2023. [paper]([http://arxiv.org/abs/2303.08774v2](https://arxiv.org/abs/2303.08774v4))
- PaLM 2: **PaLM 2 Technical Report**. 2023. [paper](https://arxiv.org/abs/2305.10403)
- LLaMA 2: **Llama 2: Open foundation and fine-tuned chat models**. Arxiv'2023. [paper](https://arxiv.org/abs/2307.09288)
- Mistral: **Mistral 7B**. Arxiv'2023. [paper](https://arxiv.org/abs/2310.06825)
- Phi1: [Project Link](https://huggingface.co/microsoft/phi-1)
- Phi1.5: [Project Link](https://huggingface.co/microsoft/phi-1_5)
- Phi2: [Project Link](https://huggingface.co/microsoft/phi-2)
- Falcon: [Project Link](https://huggingface.co/tiiuae)
### RLHF 
- PPO: **Proximal Policy Optimization Algorithms**. Arxiv'2017. [paper](https://arxiv.org/abs/1707.06347)
- DPO: **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**. NeurIPS'2023. [paper](https://arxiv.org/abs/2305.18290)
### Parameter Efficient Fine-tuning
- LoRA: **LoRA: Low-Rank Adaptation of Large Language Models**. Arxiv'2021. [paper](https://arxiv.org/abs/2106.09685)
- Q-LoRA: **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS'2023. [paper](https://arxiv.org/abs/2305.14314)
### Healthcare LM
- Med-PaLM: **Large Language Models Encode Clinical Knowledge**. Arxiv'2022. [paper](https://arxiv.org/abs/2212.13138)
- MedAlpaca: **MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data**. Arxiv'2023. [paper](https://arxiv.org/abs/2304.08247)
- Med-PaLM 2: **Towards Expert-Level Medical Question Answering with Large Language Models**. Arxiv'2023. [paper](https://arxiv.org/abs/2305.09617)
- HuatuoGPT: **HuatuoGPT, towards Taming Language Model to Be a Doctor**. EMNLP'2023(findings). [paper](https://arxiv.org/abs/2305.15075)
- GPT-4-Med: **Capabilities of GPT-4 on Medical Challenge Problems**. Arxiv'2023. [paper](https://arxiv.org/abs/2303.13375)
### Watermarking LLM

### Prompt Engineering in LLM
#### Hard Prompt
- PET: **Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference**. EACL'2021. [paper](https://arxiv.org/abs/2001.07676)
- **Making Pre-trained Language Models Better Few-shot Learners**. ACL'2021. [paper](https://arxiv.org/abs/2012.15723)
#### Soft Prompt
- Prompt-Tuning:**The Power of Scale for Parameter-Efficient Prompt Tuning**. EMNLP'2021 [paper]
- Prefix-Tuning: **Prefix-Tuning: Optimizing Continuous Prompts for Generation**. ACL'2021. [paper](https://arxiv.org/abs/2101.00190)
- P-tuning: **P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks**. ACL'2022. [paper](https://aclanthology.org/2022.acl-short.8/)
- P-tuning v2: **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**. Arxiv'2022. [Paper](https://arxiv.org/abs/2110.07602)
#### Between Soft and Hard
- Auto-Prompt: **AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts**. EMNLP'2020. [paper](https://arxiv.org/abs/2010.15980)
- FluentPrompt: **Toward Human Readable Prompt Tuning: Kubrick's The Shining is a good movie, and a good prompt too?**. EMNLP'2023 (findings). [paper](https://arxiv.org/abs/2212.10539)
- PEZ: **Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery**. Arxiv'2023. [paper](https://arxiv.org/abs/2302.03668)
---


## Multi-modal Models
### Foundation Multi-Modal Models
- CLIP: **Learning Transferable Visual Models From Natural Language Supervision**. ICML'2021. [paper](https://arxiv.org/abs/2103.00020)
- DeCLIP: **Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm**. ICLR'2022. [paper](https://arxiv.org/abs/2110.05208)
- FILIP: **FILIP: Fine-grained Interactive Language-Image Pre-Training**. ICLR'2022. [paper](https://arxiv.org/abs/2111.07783)
- Stable Diffusion: **High-Resolution Image Synthesis with Latent Diffusion Models**. CVPR'2022. [paper](https://arxiv.org/abs/2112.10752)
- BLIP: **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**. ICML'2022. [paper](https://arxiv.org/abs/2201.12086)
- BLIP2: **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**. ICML'2023. [paper](https://arxiv.org/abs/2301.12597)
- LLaMA-Adapter: **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**. Arxiv'2023. [paper](https://arxiv.org/abs/2303.16199)
- LLaVA: **Visual Instruction Tuning**. NeurIPS'2023. [paper](https://arxiv.org/abs/2304.08485)
- Instruct BLIP: **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**. NeurIPS'2023. [paper](https://arxiv.org/abs/2305.06500)
### Multi-modal Safety
- SLD: **Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models**. CVPR'2023. [paper](https://arxiv.org/abs/2211.05105)
- ESD: **Erasing Concepts from Diffusion Models**. ICCV'2023. [paper](https://arxiv.org/abs/2303.07345)
### VLM Hullucinatins
- POPE: **Evaluating Object Hallucination in Large Vision-Language Models**. EMNLP'2023. [paper](https://arxiv.org/abs/2305.10355)
### VLM Privacy

### Prompt Engineering in VLM

---

## Agent

---
## Useful-Resource
- Hugging Face course. https://huggingface.co/learn
- LLaMA Factory. https://github.com/hiyouga/LLaMA-Factory
- DeepSpeed. https://github.com/microsoft/DeepSpeed
- trlx. https://github.com/CarperAI/trlx
- Prompt Engineering Update. https://github.com/thunlp/PromptPapers

