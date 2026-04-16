import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

def main():
    # 1. 환경 설정 및 베이스 모델 (Llama-3 8B 4-bit)
    MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
    MAX_SEQ_LENGTH = 2048

    print("[1/6] Unsloth 4-bit 양자화 모델 및 토크나이저 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )

    print("[2/6] LoRA 어댑터 부착 및 최적화 설정...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # VRAM 최적화 커널 퓨전 적용
    )

    print("[3/6] 커스텀 데이터셋 로드 중...")
    dataset = load_dataset("json", data_files="pytorch_memory_dataset.jsonl", split="train")

    print("[4/6] 훈련 교관(SFTTrainer) 세팅 중...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # 가상 배치 사이즈 효과
            max_steps = 60,                  # 포트폴리오 시연용 스텝
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            output_dir = "outputs_finetuned_model",
            optim = "adamw_8bit",            # 8비트 옵티마이저로 메모리 추가 절약
        ),
    )

    print("[5/6] 파인튜닝(Fine-tuning) 학습 시작...")
    trainer.train()

    print("[6/6] 모델 및 토크나이저 저장 중...")
    # 추론 시 토크나이저가 없으면 에러가 발생하므로 반드시 같이 저장해야함.
    model.save_pretrained_lora("outputs_finetuned_model") # 1. 학습해서 똑똑해진 '뇌' 조각(LoRA 어댑터) 저장
    tokenizer.save_pretrained("outputs_finetuned_model")  # 2. 이 뇌랑 짝꿍인 '번역가' 정보 저장

        # 학습 완료 후 간단한 결과 테스트 (Inference)
    print("\n✨ 학습 결과 테스트 중...")
    FastLanguageModel.for_inference(model)
    prompt = "### 질문: PyTorch에서 CUDA 메모리 누수를 방지하는 가장 좋은 방법은?\n\n### 답변:"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("-" * 50)
    print(result)
    print("-" * 50)
    print("✅ 모든 프로세스가 완료되었습니다.")

if __name__ == "__main__":
    main()
