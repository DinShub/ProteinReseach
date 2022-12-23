from Bio import SeqIO
import pandas as pd

# Read all the records from fasta
records_any = SeqIO.parse('FASTA_any.fa', 'fasta')
records_human = SeqIO.parse('FASTA_human.fa', 'fasta')
records_swine = SeqIO.parse('FASTA_swine.fa', 'fasta')

# dict = [{label: 'Human', sequence: 'dasfdsdafg'}, ...]
results = {'label': [], "sequence": [], "length": []}
for record in records_any:
  results['label'].append(2)
  results['sequence'].append(str(record.seq))
  results['length'].append(len(record.seq))
for record in records_human:
  results['label'].append(1)
  results['sequence'].append(str(record.seq))
  results['length'].append(len(record.seq))
for record in records_swine:
  results['label'].append(0)
  results['sequence'].append(str(record.seq))
  results['length'].append(len(record.seq))

df = pd.DataFrame(results)
df.to_json('./all_records.jsonl', orient='records', lines=True)