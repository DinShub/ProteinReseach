from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split

# Read all the records from fasta
records_human = SeqIO.parse('FASTA_human.fa', 'fasta')
records_swine = SeqIO.parse('FASTA_swine.fa', 'fasta')

# dict = [{label: 'Human', sequence: 'dasfdsdafg'}, ...]
results = {'label': [], "sequence": [], "length": []}

for record in records_human:
  results['label'].append('human')
  results['sequence'].append(str(record.seq))
  results['length'].append(len(record.seq))
for record in records_swine:
  results['label'].append('swine')
  results['sequence'].append(str(record.seq))
  results['length'].append(len(record.seq))

# df = pd.DataFrame(results)
# df.index.name = 'id'
# df.to_csv('./all.csv')
# df.to_json('./all_records.jsonl', orient='records', lines=True)
X_train, X_test, y_train, y_test = train_test_split(results['sequence'], results['label'], test_size=0.3, shuffle=True, stratify=results['label'])
# df_train = pd.DataFrame({'sequence': X_train, 'label': y_train})
df_test = pd.DataFrame({'sequence': X_test, 'label': y_test})

# df_train.to_csv('./train.csv')
df_test.to_csv('./test.csv')

import torch
print(torch.cuda.is_available())