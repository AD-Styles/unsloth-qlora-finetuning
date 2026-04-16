# 🚀 Efficient LLM Fine-Tuning with Unsloth & QLoRA
### Unsloth 및 4-bit QLoRA를 활용한 단일 GPU 환경 LLM 파인튜닝(SFT) 파이프라인

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-000000?style=flat-square&logo=github&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HF-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 대규모 언어 모델(LLM) 학습 시 발생하는 막대한 컴퓨팅 자원 요구 문제를 해결하기 위해, **PEFT(Parameter-Efficient Fine-Tuning)** 기법을 적용한 엔드투엔드 파인튜닝 파이프라인입니다. Hugging Face의 **Unsloth** 라이브러리와 **4-bit QLoRA** 기술을 결합하여, VRAM 사용량을 최소화하고 훈련 속도를 향상시켜 일반적인 단일 GPU 환경에서도 원활한 지도 미세 조정(Supervised Fine-Tuning, SFT)이 가능하도록 구현했습니다.

---

## 🎯 핵심 기술 목표 (Technical Goals)
| 구분 | 세부 내용 |
| :--- | :--- |
| **Parameter-Efficient Tuning** | Full Fine-tuning 대신 모델 가중치를 동결하고, 0.1% 미만의 LoRA 어댑터만 학습하여 연산량 및 비용 최소화 |
| **Model Quantization** | `bitsandbytes` 기반 4-bit NF4 양자화를 적용하여 FP16 대비 VRAM 적재량을 획기적으로 압축 |
| **Kernel Level Optimization** | 학습(역전파) 과정에서 PyTorch가 임시로 저장하는 불필요한 중간 계산값들을 Unsloth의 연산 병합(Kernel Fusion) 기술로 묶어 처리함으로써 VRAM 낭비를 원천 차단 |

---

## 📂 프로젝트 구조 (Project Structure)
```text
📂 Torch-Memory-Copilot
├── 📄 .gitignore                      # 로그 및 모델 가중치 업로드 방지
├── 📄 LICENSE                         # MIT License (AD-Styles)
├── 📄 README.md                       # 프로젝트 보고서
└── 📄 pytorch_memory_dataset.jsonl    # 10개 이상의 정제된 데이터셋
├── 📄 requirements.txt                # 패키지 리스트
└──  📄 train_unsloth.py                # 메인 학습 스크립트
```

---

## 🛠 파이프라인 워크플로우 (Pipeline Workflow)
본 파이프라인은 데이터 준비부터 최종 모델 추출까지 효율적인 4단계 공정으로 구성됩니다.
1. **Model Initialization**: Unsloth `FastLanguageModel`을 통해 대상 LLM을 4-bit 양자화 상태로 초고속 로드
2. **Adapter Configuration**: 타겟 모듈(`q_proj`, `v_proj` 등)에 Rank($r=16$) 값을 설정하여 학습 가능한 LoRA 레이어 주입
3. **Data Formatting**: TRL `SFTTrainer`와 호환되도록 커스텀 JSONL 데이터셋을 Prompt Template에 매핑 및 토크나이징
4. **SFT Execution**: Gradient Accumulation 기법을 적용하여 안정적인 미세 조정(Supervised Fine-Tuning) 수행

---

## 📊 성능 최적화 벤치마크 (Performance Benchmark)
> 단일 GPU 환경 기준, 베이스 모델 파인튜닝 시의 자원 효율성 검증 결과입니다.

| 벤치마크 지표 | Standard HF SFT | **Unsloth + QLoRA** | 향상 수준 |
| :--- | :---: | :---: | :--- |
| **Peak VRAM Usage** | OOM (Out of Memory) | **약 7.4 GB** | **단일 GPU 환경 학습 성공** |
| **Training Steps/Sec** | 0.8 it/s | **2.1 it/s** | **🚀 훈련 속도 약 2.6배 가속** |
| **Trainable Parameters** | 100% | **약 0.08%** | 파라미터 업데이트 비용 극소화 |

---

## 📝 회고록 (Retrospective)
&emsp;&emsp;이전 프로젝트에서 KoGPT2를 다루며 느꼈던 하드웨어 리소스의 한계를 이번 **PEFT 및 QLoRA 파이프라인** 구축을 통해 구조적으로 돌파할 수 있었습니다. 거대 모델의 수십억 개 파라미터를 모두 업데이트할 필요 없이, 소규모의 어댑터만으로 모델의 지식을 원하는 방향으로 튜닝할 수 있다는 점은 매우 고무적이었습니다.
<br>&emsp;&emsp;특히 Unsloth 라이브러리가 내부적으로 불필요한 VRAM 할당을 제거하는 과정을 분석하며, 딥러닝 프레임워크를 단순 API 단위로 사용하는 것을 넘어 **수학적 연산 구조와 메모리 할당(CUDA VRAM) 원리**를 깊이 있게 고민하게 되었습니다. 이 최적화 파이프라인은 향후 다양한 도메인의 커스텀 LLM을 효율적으로 구축하는 든든한 기반이 될 것입니다.

"PyTorch는 원래 계산을 할 때 A->B->C 단계마다 결괏값을 메모리에 잠깐씩 다 저장해두는데, Unsloth는 이걸 한 번에 A->C로 계산(Fusion)해버려서 그만큼 VRAM을 아끼는 원리입니다!" 라고 아주 깔끔하고 학생다운 100점짜리 대답을 하실 수 있습니다.
