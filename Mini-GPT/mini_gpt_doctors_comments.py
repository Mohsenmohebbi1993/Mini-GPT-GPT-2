import sys
import os
import re
import importlib

import torch
import pandas as pd

from data_pipeline import (
    deduplicate,
    SimpleTokenizer,
    tokenize_corpus,
    pack_sequences,
    PreTrainingDataLoader
)
import mini_gpt_torch
importlib.reload(mini_gpt_torch)
from mini_gpt_torch import MiniGPT, cross_entropy_loss, generate

print("Import Packages from mini_gpt_torch and data_pipline")
#------------------------------------------------------------
 # def clear_data is NOT for persian
def clean_persian_text(text: str) -> str:
    """Persian clear text 
    """
    if not isinstance(text, str):
        return ""
    # remove html
    text = re.sub(r"<[^>]+>", " ", text)
    # remove link
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # standart space
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------1. Load dataset----------------
print("1. Load dataset".center(60, "-"))
PATH = os.path.join(os.getcwd(), "Mini-GPT\comments_raw_canonical.csv")
df = pd.read_csv(PATH)
df = df[["specialty", "text"]]
df.dropna(subset=["text"], inplace=True)

raw_docs = []
for _, row in df.iterrows():
    cleaned = clean_persian_text(str(row["text"]))
    if len(cleaned.split()) >= 3:  # comments more than  3 vocabes
        spec = clean_persian_text(str(row["specialty"])) if pd.notna(row["specialty"]) else ""
        doc = f"[{spec}] {cleaned}" if spec else cleaned
        raw_docs.append(doc)

print(f"Count comment is -----  {len(raw_docs):,}")


# ---------------2. MinHash/LSH------------------
print("2. MinHash/LSH".center(60, "-"))
sample_size = min(15000, len(raw_docs))
docs_sample = raw_docs[:sample_size]
print(f"remove dublicate {sample_size:,} doc")
unique_docs, removed_count = deduplicate(docs_sample, threshold=0.85, num_hashes=64, bands=8)
print(f"count doc unique : {len(unique_docs):,} (count removed comment {removed_count:,})")


# -------------3. Learn SimpleTokenizer-------------
print("3. Learn SimpleTokenizer".center(60, "-"))
corpus_sample_for_bpe = "\n".join(unique_docs[:3000])
tokenizer = SimpleTokenizer(vocab_size=256)
tokenizer.train_bpe(corpus_sample_for_bpe, num_merges=300)

vocab_size = tokenizer.vocab_size()
print(f"end Vocab Size: {vocab_size}")


# --------4. change doc to tokens------------------
print("4. change doc to tokens".center(60, "-"))
flat_tokens = tokenize_corpus(unique_docs, tokenizer)

SEQ_LEN = 64
BATCH_SIZE = 16
packed_seqs, att_masks = pack_sequences(flat_tokens, seq_length=SEQ_LEN + 1, pad_id=tokenizer.pad_id)
print(f"Number of sequences ready for training {len(packed_seqs):,}")

train_loader = PreTrainingDataLoader(packed_seqs, att_masks, batch_size=BATCH_SIZE, shuffle=True)


# ---------5. Create model----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("5. Create model".center(60, "-"))
print(f"Use divece:------- {device}")

model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    max_seq_len=SEQ_LEN,
    ff_dim=512
).to(device) # Use GPU :))

print(f"Count parameters model {model.count_parameters():,}")


#---------6. start learning---------------
print("6. start learning".center(60, "-"))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
EPOCHS = 2
MAX_STEPS = 1000  # speed test
LOG_EVERY = 50

model.train()
step = 0
stopped = False

for epoch in range(1, EPOCHS + 1):
    for batch_x, _ in train_loader:
        step += 1

        # change to tensor
        batch_tensor = torch.tensor(batch_x, dtype=torch.long, device=device)

        x = batch_tensor[:, :-1].contiguous()
        y = batch_tensor[:, 1:].contiguous()

        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % LOG_EVERY == 0:
            print(f"Epoch {epoch} | Step {step:4d} | Loss: {loss.item():.4f}")

        if step >= MAX_STEPS:
            stopped = True
            break
    if stopped:
        break

print(f"Training completed Final loss---- {loss.item():.4f}")



# ---------------Sampel text to test------------------
print("Sampel text to test".center(60, "-"))
model.eval()
# test_prompt = "دکتر بسیار با حوصله"
test_prompt = input("persian text e.g. `دکتر بسیار با حوصله`  :   ")

prompt_tokens = tokenizer.encode(test_prompt)
print(f"\nPrompt: '{test_prompt}'")
print("Generating the rest of the comment--- ")

output_tokens = generate(model, prompt_tokens, max_new_tokens=40, temperature=0.7)
generated_text = tokenizer.decode(output_tokens)
print(f"\nGenerated Result:\n{generated_text}")


