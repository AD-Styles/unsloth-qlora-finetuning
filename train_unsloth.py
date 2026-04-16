import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

def main():
    # 1. 환경 설정 및 베이스 모델 (예: Llama-3 8B)
    MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
    MAX_SEQ_LENGTH = 2048

    print("[1/5] Unsloth 4-bit 양자화 모델 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )

    print("[2/5] LoRA 어댑터 부착 및 최적화 설정...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # VRAM 절약의 핵심 커널 퓨전
    )

    print("[3/5] 커스텀 데이터셋 로드 중...")
    # 아래 4번 항목에서 만든 JSONL 파일을 불러옵니다.
    dataset = load_dataset("json", data_files="pytorch_memory_dataset.jsonl", split="train")

    print("[4/5] 훈련 교관(SFTTrainer) 세팅 중...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # OOM 방지를 위한 배치 분할
            max_steps = 60,                  # 포트폴리오 시연용 스텝 수
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            output_dir = "outputs_pytorch_optimizer",
            optim = "adamw_8bit",            # 8비트 옵티마이저로 추가 메모리 확보
        ),
    )

    print("[5/5] 모델 학습(Fine-tuning)을 시작합니다!")
    trainer.train()
    
    print("✅ 학습 완료! 모델 가중치가 'outputs_pytorch_optimizer' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
