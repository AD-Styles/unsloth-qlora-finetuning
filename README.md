# 🚀 Efficient LLM Fine-Tuning with Unsloth & QLoRA
### Unsloth 및 4-bit QLoRA를 활용한 단일 GPU 환경 LLM 파인튜닝(SFT) 파이프라인

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-000000?style=flat-square&logo=github&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HF-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 대규모 언어 모델(LLM) 학습 시 발생하는 막대한 컴퓨팅 자원 요구 문제를 해결하기 위해, **PEFT(Parameter-Efficient Fine-Tuning)** 기법을 적용한 엔드투엔드 파인튜닝 파이프라인입니다. 
Hugging Face 생태계에 **Unsloth** 커널 최적화와 **4-bit QLoRA** 기술을 결합하여, VRAM 사용량을 극적으로 최소화하고 훈련 속도를 비약적으로 향상시켜 일반적인 단일 GPU 환경에서도 원활한 지도 미세 조정(SFT)이 가능하도록 구현했습니다.

---

## 📂 프로젝트 구조 (Project Structure)
```text
📂 unsloth-qlora-finetuning
├── 📄 .gitignore                # 환경 변수 및 가중치 업로드 방지
├── 📄 LICENSE                   # MIT License
├── 📄 README.md                 # 프로젝트 기술 보고서 및 아키텍처 설명
├── 📄 train_sft_pipeline.py     # 데이터 전처리부터 Unsloth 학습까지의 통합 파이프라인 스크립트
├── 📄 dataset.jsonl             # SFT(지도 미세 조정) 테스트용 샘플 데이터셋
├── 📄 requirements.txt          # 의존성 라이브러리 목록
├── 🖼️ peft_architecture.png     # PEFT 및 LoRA 동작 원리 도식화 이미지
└── 📁 saved_lora_adapters/      # 학습 완료 후 추출된 최종 LoRA 어댑터 가중치
