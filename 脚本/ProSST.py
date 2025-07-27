import sys
sys.path.append('..')
from transformers import AutoModelForMaskedLM, AutoTokenizer
from prosst.structure.get_sst_seq import SSTPredictor
from Bio import SeqIO
import torch
import pandas as pd
from scipy.stats import spearmanr

import os
os.environ["http_proxy"] = "http://127.0.0.1:15777"
os.environ["https_proxy"] = "http://127.0.0.1:15777"

prosst_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
prosst_tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)

predictor = SSTPredictor(structure_vocab_size=2048)

residue_sequence = str(SeqIO.read('data/ProSST_data/Protein_Mutated.fasta', 'fasta').seq)

structure_sequence = predictor.predict_from_pdb("data/ProSST_data/Protein_Mutated.pdb")[0]['2048_sst_seq']

structure_sequence_offset = [i + 3 for i in structure_sequence]

tokenized_res = prosst_tokenizer([residue_sequence], return_tensors='pt')
input_ids = tokenized_res['input_ids']
attention_mask = tokenized_res['attention_mask']
structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    outputs = prosst_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=structure_input_ids
    )
logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()

df = pd.read_csv("data/ProSST_data/Protein_Mutated.csv")
mutants = df['mutant'].tolist()

vocab = prosst_tokenizer.get_vocab()
pred_scores = []
for mutant in mutants:
    mutant_score = 0
    for sub_mutant in mutant.split(":"):
        wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
        pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
        mutant_score += pred.item()
    pred_scores.append(mutant_score)
    
df['ProSST_score'] = pred_scores
df.to_csv('data/ProSST_data/Protein_Mutated_with_ProSST.csv', index=False)