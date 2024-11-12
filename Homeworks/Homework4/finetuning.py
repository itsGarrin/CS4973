import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

def format_item(item):
    q = item["questions"].strip()
    a = item["answers"].strip()
    return { "content": f"Problem: {q}\nAnswer: {a}" }

data_dict = load_dataset(
    "nickeldime/questionsanswerskhoury",
    split="train"
).train_test_split(
    0.01
).map(
    format_item
).remove_columns(
    ["questions", "answers"]
)

sft_config = SFTConfig(
    dataset_text_field="content",
    max_seq_length=2048,
    output_dir="finetuned",
    learning_rate=3e-05,
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    bf16=True,
    logging_steps=10,
)

model = AutoModelForCausalLM.from_pretrained(
    "/scratch/bchk/aguha/models/llama3p2_1b_base",
    torch_dtype=torch.bfloat16).to("cuda")

trainer = SFTTrainer(
    model,
    train_dataset=data_dict["train"],
    eval_dataset=data_dict["test"],
    args=sft_config,
)
trainer.train()

model.save_model("finetuned/model.pt")
