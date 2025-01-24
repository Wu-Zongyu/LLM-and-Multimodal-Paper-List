# LLM-and-VLM-Paper-List
This repo contains papers and relevant resources about large language models and multi-modal models. From foundation papers to downstream tasks such as trustworthy (E.g., Robustness, Privacy, and Fairness) and agent...   
**Note:** It only records Papers for my personal needs :). It is welcome to open an issue if you think I missed some important or exciting work!

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
  - [Text-to-Image Concept Erasing/Safety](#t2i-concept-removal-or-safety)
  - [LVLM Adversarial Attack](#lvlm-adversarial-attack)
  - [LVLM Hullucinations](#lvlm-hullucinations)
  - [LVLM Privacy](#lvlm-privacy)
- [AI4Science](#ai-for-science)
- [Agent](#agent)
  - [LLM-based Agent](#llm-based-agent)
  - [VLM-based Agent](#vlm-based-agent)
- [Useful Resource](#useful-resource)

## Survey
- **Adversarial attacks and defenses on text-to-image diffusion models: A survey**. Information Fusion (2025). [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253524004792), [GitHub](https://GitHub.com/datar001/Awesome-AD-on-T2IDM)]
- **A Survey of Attacks on Large Vision-Language Models: Resources, Advances, and Future Trends**. Arxiv'2024. [[Paper](https://arxiv.org/pdf/2407.07403)], [[GitHub](https://GitHub.com/liudaizong/Awesome-LVLM-Attack)]
- **Holistic evaluation of language models**. TMLR. [Paper](https://arxiv.org/pdf/2211.09110)
- **Holistic Evaluation of Text-to-Image Models**. NeurIPS'2023. [Paper](https://arxiv.org/pdf/2311.04287)
- **A Survey on Evaluation of Large Language Models**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2307.03109)
- **A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2310.05694), [GitHub](https://GitHub.com/KaiHe-better/LLM-for-Healthcare)
- **A Survey on Multimodal Large Language Model**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2306.13549.pdf), [GitHub](https://GitHub.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- **Vision Language Models for Vision Tasks: A Survey**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2304.00685), [GitHub](https://GitHub.com/jingyi0000/VLM_survey)
- **Efficient Large Language Models: A Survey**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2312.03863), [GitHub](https://GitHub.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)
- **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing**. Arxiv'2021. [Paper](https://arxiv.org/pdf/2107.13586)
- **Safety of Multimodal Large Language Models on Images and Text**. Arxiv'2024. [Paper](https://arxiv.org/pdf/2402.00357)
- **MM-LLMs: Recent Advances in MultiModal Large Language Models**. Arxiv'2024. [Paper](https://arxiv.org/pdf/2401.13601)
- **A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications**. Arxiv'2024. [Paper](https://arxiv.org/pdf/2402.07927)
- **A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**. Arxiv'2024. [Paper](https://arxiv.org/pdf/2312.02003)
- **Privacy in Large Language Models: Attacks, Defenses and Future Directions**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2310.10383)
---

## Language Model
### Foundation LM Models
- Transformer: **Attention Is All You Need**. NIPS'2017. [Paper](https://arxiv.org/pdf/1706.03762)
- GPT-1: **Improving Language Understanding by Generative Pre-Training**. 2018. [Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_Paper.pdf)
- BERT: **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL'2019. [Paper](https://aclanthology.org/N19-1423.pdf)
- GPT-2: **Language Models are Unsupervised Multitask Learners**. 2018. [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- RoBERTa: **RoBERTa: A Robustly Optimized BERT Pretraining Approach**. Arxiv'2019, [Paper](https://arxiv.org/pdf/1907.11692)
- DistilBERT: **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**. Arxiv'2019. [Paper](https://arxiv.org/pdf/1910.01108)
- T5: **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**. JMLR'2020. [Paper](https://arxiv.org/pdf/1910.10683)
- GPT-3: **Language Models are Few-Shot Learners**. NeurIPS'2020. [Paper](https://arxiv.org/pdf/2005.14165)
- GLaM: **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts**. ICML'2022. [Paper](https://arxiv.org/pdf/2112.06905)
- PaLM: **PaLM: Scaling Language Modeling with Pathways**. ArXiv'2022. [Paper](https://arxiv.org/pdf/2204.02311)
- BLOOM:  **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model**. Arxiv'2022. [Paper](https://arxiv.org/pdf/2211.05100)
- BLOOMZ: **Crosslingual Generalization through Multitask Finetuning**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2211.01786)
- LLaMA: **LLaMA: Open and Efficient Foundation Language Models**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2302.13971)
- GPT-4: **GPT-4 Technical Report**. Arxiv'2023. [Paper]([http://arxiv.org/abs/2303.08774v2](https://arxiv.org/pdf/2303.08774v4))
- PaLM 2: **PaLM 2 Technical Report**. 2023. [Paper](https://arxiv.org/pdf/2305.10403)
- Llama 2: **Llama 2: Open foundation and fine-tuned chat models**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2307.09288)
- Mistral: **Mistral 7B**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2310.06825)
- Phi1: [Project Link](https://huggingface.co/microsoft/phi-1)
- Phi1.5: [Project Link](https://huggingface.co/microsoft/phi-1_5)
- Phi2: [Project Link](https://huggingface.co/microsoft/phi-2)
- Falcon: [Project Link](https://huggingface.co/tiiuae)
- Llama 3: **The Llama 3 Herd of Models**. Arxiv'2024. [Paper](https://arxiv.org/pdf/2407.21783)
  
### RLHF 
- PPO: **Proximal Policy Optimization Algorithms**. Arxiv'2017. [Paper](https://arxiv.org/pdf/1707.06347)
- DPO: **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**. NeurIPS'2023. [Paper](https://arxiv.org/pdf/2305.18290)
### Parameter Efficient Fine-tuning
- LoRA: **LoRA: Low-Rank Adaptation of Large Language Models**. Arxiv'2021. [Paper](https://arxiv.org/pdf/2106.09685)
- Q-LoRA: **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS'2023. [Paper](https://arxiv.org/pdf/2305.14314)
### Healthcare LM
- Med-PaLM: **Large Language Models Encode Clinical Knowledge**. Arxiv'2022. [Paper](https://arxiv.org/pdf/2212.13138)
- MedAlpaca: **MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2304.08247)
- Med-PaLM 2: **Towards Expert-Level Medical Question Answering with Large Language Models**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2305.09617)
- HuatuoGPT: **HuatuoGPT, towards Taming Language Model to Be a Doctor**. EMNLP'2023(findings). [Paper](https://arxiv.org/pdf/2305.15075)
- GPT-4-Med: **Capabilities of GPT-4 on Medical Challenge Problems**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2303.13375)
### Watermarking LLM

### Prompt Engineering in LLM
#### Hard Prompt
- PET: **Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference**. EACL'2021. [Paper](https://arxiv.org/pdf/2001.07676)
- **Making Pre-trained Language Models Better Few-shot Learners**. ACL'2021. [Paper](https://arxiv.org/pdf/2012.15723)
#### Soft Prompt
- Prompt-Tuning:**The Power of Scale for Parameter-Efficient Prompt Tuning**. EMNLP'2021 [Paper]
- Prefix-Tuning: **Prefix-Tuning: Optimizing Continuous Prompts for Generation**. ACL'2021. [Paper](https://arxiv.org/pdf/2101.00190)
- P-tuning: **P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks**. ACL'2022. [Paper](https://aclanthology.org/2022.acl-short.8/)
- P-tuning v2: **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**. Arxiv'2022. [Paper](https://arxiv.org/pdf/2110.07602)
#### Between Soft and Hard
- Auto-Prompt: **AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts**. EMNLP'2020. [Paper](https://arxiv.org/pdf/2010.15980)
- FluentPrompt: **Toward Human Readable Prompt Tuning: Kubrick's The Shining is a good movie, and a good prompt too?**. EMNLP'2023 (findings). [Paper](https://arxiv.org/pdf/2212.10539)
- PEZ: **Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2302.03668)
---


## Multi-modal Models
### Foundation Multi-Modal Models
- CLIP: **Learning Transferable Visual Models From Natural Language Supervision**. ICML'2021. [Paper](https://arxiv.org/pdf/2103.00020)
- DeCLIP: **Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm**. ICLR'2022. [Paper](https://arxiv.org/pdf/2110.05208)
- FILIP: **FILIP: Fine-grained Interactive Language-Image Pre-Training**. ICLR'2022. [Paper](https://arxiv.org/pdf/2111.07783)
- Stable Diffusion: **High-Resolution Image Synthesis with Latent Diffusion Models**. CVPR'2022. [Paper](https://arxiv.org/pdf/2112.10752)
- BLIP: **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**. ICML'2022. [Paper](https://arxiv.org/pdf/2201.12086)
- BLIP2: **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**. ICML'2023. [Paper](https://arxiv.org/pdf/2301.12597)
- LLaMA-Adapter: **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2303.16199)
- LLaVA: **Visual Instruction Tuning**. NeurIPS'2023. [Paper](https://arxiv.org/pdf/2304.08485)
- LLaVA 1.5: **Improved Baselines with Visual Instruction Tuning**. CVPR'2024. [Paper](https://arxiv.org/pdf/2310.03744)
- Instruct BLIP: **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**. NeurIPS'2023. [Paper](https://arxiv.org/pdf/2305.06500)
- InternVL 1.0: **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks**. CVPR'2024 (Oral). [Paper](https://arxiv.org/pdf/2312.14238)
- InternVL 1.5: **How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites**. Arxiv'2024. [Tech Report](https://arxiv.org/pdf/2312.14238)
### T2I Concept Removal or Safety
- SLD: **Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models**. CVPR'2023. [Paper](https://arxiv.org/pdf/2211.05105)
- ESD: **Erasing Concepts from Diffusion Models**. ICCV'2023. [Paper](https://arxiv.org/pdf/2303.07345)
- UCE: **Unified Concept Editing in Diffusion Models**. Arxiv'2023. [Paper](https://arxiv.org/pdf/2308.14761)
- POSI: **Universal Prompt Optimizer for Safe Text-to-Image Generation**. NAACL'2024. [Paper](https://arxiv.org/pdf/2402.10882)
- **Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts**. Arxiv'2024. [[Paper](https://arxiv.org/pdf/2410.12777), [GitHub](https://github.com/sail-sg/Meta-Unlearning)]
- EIUP: **EIUP: A Training-Free Approach to Erase Non-Compliant Concepts Conditioned on Implicit Unsafe Prompts**. Arxiv'2024. [Paper](https://arxiv.org/pdf/2408.01014)
### LVLM Hallucinations
- POPE: **Evaluating Object Hallucination in Large Vision-Language Models**. EMNLP'2023. [Paper](https://arxiv.org/pdf/2305.10355)
- **HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models**. CVPR'2024. [Paper](https://arxiv.org/pdf/2310.14566)

### LVLM Adversarial Attack
- **On the Adversarial Robustness of Multi-Modal Foundation Models**. ICCV Workshop'2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023W/AROW/Papers/Schlarmann_On_the_Adversarial_Robustness_of_Multi-Modal_Foundation_Models_ICCVW_2023_Paper.pdf)
### LVLM Privacy


### Prompt Engineering in VLM

## AI for Science
- GALLON: **LLM and GNN are Complementary: Distilling LLM for Multimodal Graph Learning**. Arxiv 2024. [Paper](https://arxiv.org/pdf/2406.01032)
---

## Agent
### LLM-based Agent
- Stanford Town: **Generative Agents: Interactive Simulacra of Human Behavior.**  UIST'2023. [Paper](https://arxiv.org/pdf/2304.03442)


### VLM-based Agent
- OSWorld: **OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments**  Arxiv'2024. [Paper](https://arxiv.org/pdf/2404.07972)

---
## Useful-Resource
- Hugging Face course. https://huggingface.co/learn
- LLaMA Factory. https://GitHub.com/hiyouga/LLaMA-Factory
- DeepSpeed. https://GitHub.com/microsoft/DeepSpeed
- trlx. https://GitHub.com/CarperAI/trlx
- Prompt Engineering Update. https://GitHub.com/thunlp/PromptPapers

