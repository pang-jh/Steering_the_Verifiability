# Steering the Verifiability of Multimodal AI Hallucinations

<p align="center">
  <img src="intro.png" alt="outline" width="400"/>
</p>

## Introduction
This project studies multimodal hallucinations from a **human-verifiability** perspective.  
We distinguish hallucinations into two types:

- **Obvious hallucinations**: easy for humans to spot
- **Elusive hallucinations**: difficult for humans to verify quickly

Based on this distinction, we propose an **activation-space intervention** framework that learns separate directions for:

- **OHI**: Obvious Hallucination Intervention
- **EHI**: Elusive Hallucination Intervention

These directions are used to steer model behavior at inference time and provide fine-grained control over hallucination verifiability.

## Dataset
We construct **HHVD**, a human-annotated benchmark for multimodal hallucination verifiability. The dataset is designed to evaluate how easily users can verify hallucinated content and supports fine-grained analysis of hallucination control in MLLMs.

Get the dataset from this 
[🤗 link](https://huggingface.co/datasets/BeEnough/HHVD), 
[🤖 link](https://www.modelscope.cn/datasets/Nothing07/HHVD)
## QuickStart
```bash
git clone https://github.com/pang-jh/Steering_the_Verifiability.git
cd Steering_the_Verifiability
conda create -n stv python=3.10
conda activate stv
pip install -r requirements.txt
```
```bash
python3 -m pipeline.run_pipeline
```
