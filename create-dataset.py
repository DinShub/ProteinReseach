from Bio import SeqIO

for record in SeqIO.parse('FASTA_any.fa', 'fasta'):
  print(record.seq)