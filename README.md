# 🚀 Efficient LLM Fine-Tuning with Unsloth & QLoRA
### Unsloth 및 4-bit QLoRA를 활용한 단일 GPU 환경 LLM 파인튜닝(SFT) 파이프라인

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-000000?style=flat-square&logo=github&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HF-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 대규모 언어 모델(LLM) 학습 시 발생하는 막대한 컴퓨팅 자원 요구 문제를 해결하기 위해, **PEFT(Parameter-Efficient Fine-Tuning)** 기법을 적용한 엔드투엔드 파인튜닝 파이프라인입니다. Hugging Face의 **Unsloth** 라이브러리와 **4-bit QLoRA** 기술을 결합하여, VRAM 사용량을 최소화하고 훈련 속도를 향상시켜 일반적인 단일 GPU 환경에서도 원활한 지도 미세 조정(Supervised Fine-Tuning, SFT)이 가능하도록 구현했습니다. 각 단계별로 핵심 내용을 상세하게 풀어서 기술해보았습니다.

---

## 🎯 핵심 기술 목표 (Technical Goals)

| 구분 | 세부 내용 및 도입 이유 |
| :--- | :--- |
| **Parameter-Efficient Tuning (LoRA)** | **[본체는 그대로, 확장팩만 학습]** <br> 수십억 개의 모델 파라미터를 전부 업데이트하는 '전체 파인튜닝'은 천문학적인 자원이 소요됩니다. 대신 모델 본체는 동결(Freeze)하고, 핵심 연결 부위에만 **0.08% 미만의 작은 어댑터 레이어(LoRA)** 를 추가하여 학습했습니다. 이를 통해 개인 환경에서도 적은 연산량으로 충분한 도메인 <br>성능을 확보했습니다. |
| **Model Quantization <br>(4-bit NF4)** | **[지능은 유지하되 무게는 압축]** <br>16비트(FP16) 정밀도의 모델을 그대로 로드하면 VRAM 부족으로 실행조차 불가능합니다. 이를 <br>**4비트(NF4) 형식으로 압축(Quantization)** 하여 메모리 점유율을 약 1/4로 줄였습니다. 특히 LLM의 가중치 분포에 최적화된 NF4(Normal Float 4) 방식을 사용하여, 모델의 '똑똑함'은 보존하면서 메모리 효율만 극대화했습니다. |
| **Kernel-Level Optimization (Unsloth)** | **[메모리 찌꺼기를 줄이는 연산 통합]** <br> 딥러닝 연산 시 매 단계마다 발생하는 중간 계산값들은 VRAM을 불필요하게 차지합니다. Unsloth의 **연산 병합(Kernel Fusion)** 기술을 적용하여 여러 수학 연산을 하나로 묶어 처리했습니다. 중간 과정에서 생기는 '데이터 찌꺼기'를 최소화하여 훈련 속도를 2배 이상 높이고 GPU 한계를 극복했습니다. |

---

## 📂 프로젝트 구조 (Project Structure)
```text
📂 Torch-Memory-Copilot
├── 📄 .gitignore                      # 로그 및 모델 가중치 업로드 방지
├── 📄 LICENSE                         # MIT License (AD-Styles)
├── 📄 README.md                       # 프로젝트 보고서
├── 📄 pytorch_memory_dataset.jsonl    # 10개 이상의 정제된 데이터셋
├── 📄 requirements.txt                # 패키지 리스트
└── 📄 train_unsloth.py                # 메인 학습 스크립트
```

---

## 🛠 파이프라인 워크플로우 (Pipeline Workflow)
### 본 파이프라인은 데이터 준비부터 최종 모델 검증까지 효율적인 4단계 공정으로 구성됩니다.

1. **모델 가속화 및 로드 (Model Initialization)**
   - 수십 GB의 거대 모델을 단일 GPU 환경(T4)에서 구동할 수 있도록 **4-bit NF4 양자화**를 적용했습니다. 이를 통해 지능은 유지하면서 메모리 점유율을 획기적으로 낮춰 초고속 로딩을 구현했습니다.

2. **효율적 파라미터 설계 (Adapter Configuration)**
   - 모델 전체를 재학습하는 비효율을 피하기 위해, 핵심 연산 부위에만 **'학습용 요약 노트' 역할을 하는 LoRA 어댑터**를 <br>부착했습니다. 전체 파라미터의 0.1% 미만만 업데이트하여 학습 속도와 비용을 동시에 잡았습니다.

3. **데이터 최적화 및 구조화 (Data Formatting)**
   - 원본 데이터를 AI가 문맥을 가장 잘 파악할 수 있는 **'질의응답(Q&A)' 프롬프트 템플릿**으로 규격화했습니다. 텍스트 <br>데이터를 토큰 단위로 변환하여 모델이 학습하기 가장 최적화된 상태로 정제했습니다.

4. **안정적인 고효율 훈련 (SFT Execution)**
   - 하드웨어 한계를 극복하기 위해 **Gradient Accumulation(기울기 누적)** 기법을 도입했습니다. 데이터를 잘게 쪼개 학습하면서도 큰 배치 사이즈의 학습 효과를 내어, VRAM 부족 현상 없이 안정적으로 미세 조정(SFT)을 완수했습니다.

---

## 📊 성능 최적화 벤치마크 (Performance Benchmark)
> **실험 환경 (Environment)** : `Google Colab` / `NVIDIA T4 (16GB VRAM)` / `Llama-3 (8B)` / `Max Sequence Length: 2048`

| 벤치마크 지표 (Metrics) | Standard HF SFT | **Unsloth + QLoRA** | 향상 수준 (Improvement) |
| :--- | :---: | :---: | :--- |
| **Peak VRAM Usage** | OOM (> 16GB) | **약 7.4 GB** | **📉 VRAM 약 60% 이상 절감** (Colab 환경 학습 가능) |
| **Training Steps/Sec** | 0.8 it/s | **2.1 it/s** | **🚀 훈련 속도 약 2.6배 가속** |
| **Trainable Parameters** | 100% (Full) | **약 0.08%** | **🧠 파라미터 업데이트 연산량 극소화** |

---

## 📝 회고록 (Retrospective)
&emsp;&emsp;솔직히 처음 거대 언어 모델(LLM)을 직접 파인튜닝해 보겠다고 마음먹었을 때, 가장 먼저 마주한 벽은 '하드웨어'였습니다. 흔히 접할 수 있는 Colab의 16GB GPU로는 7B 규모의 모델을 올려보는 것조차 버거워서 OOM(메모리 초과) 에러만 멍하니 바라봐야 했습니다. '결국 AI는 거대한 자본과 장비 싸움인가?' 하며 좌절하기도 했지만, 이번 프로젝트를 통해 그 물리적 한계를 기술적으로 뛰어넘는 경험을 할 수 있었습니다. 이전 프로젝트에서 KoGPT2를 다루며 느꼈던 하드웨어 리소스의 한계를 이번 실습을 통해 구조적으로 돌파할 수 있었습니다. 거대 모델의 수십억 개 파라미터를 모두 업데이트할 필요 없이, 소규모의 어댑터만으로 모델의 지식을 원하는 방향으로 튜닝할 수 있다는 점은 매우 고무적이었습니다.
<br>&emsp;&emsp;가장 큰 깨달음은 **'전부를 다 바꿀 필요는 없다'**는 것이었습니다. 수십억 개의 파라미터를 전부 재학습하는 대신, 베이스 모델을 4비트로 가볍게 압축하고 핵심 부위에만 작은 '학습용 메모지(LoRA)'를 붙이는 방식은 충격적일 정도로 효율적이었습니다. 또한, Unsloth 라이브러리를 활용하면서 단순히 남들이 짜둔 코드를 복사해서 돌려보는 수준을 벗어날 수 있었습니다. 학습 중간에 발생하는 불필요한 임시 데이터들을 한 번에 묶어서 처리(Kernel Fusion)해 메모리 낭비를 막는 원리를 공부하면서, 제가 무심코 작성했던 PyTorch 코드들이 왜 그렇게 VRAM을 많이 잡아먹었는지도 뼈저리게 이해하게 되었습니다. PyTorch는 원래 계산을 할 때 A->B->C 단계마다 결괏값을 메모리에 잠깐씩 다 저장해두는데, Unsloth는 이걸 한 번에 A->C로 계산(Fusion)해버려서 그만큼 VRAM을 아끼는 원리를 직접 수치로 확인해보니 파인튜닝의 혁신적인 기능이 더 와닿았습니다.
<br>&emsp;&emsp;이번 프로젝트는 저에게 단순한 '학습 성공' 이상의 의미를 가집니다. 그저 주어진 API만 호출하던 학생의 시야에서 벗어나, 한정된 자원 속에서 어떻게든 최적의 효율을 짜내기 위해 하드웨어와 메모리의 동작 방식까지 고민하는 '엔지니어'의 관점을 조금이나마 갖게 되었습니다. 앞으로 어떤 무거운 모델이나 에러를 만나더라도 무작정 장비 탓을 하기보다는, 시스템 내부를 뜯어보고 효율화할 방법을 먼저 찾는 개발자로 성장해 나가고 싶습니다.
