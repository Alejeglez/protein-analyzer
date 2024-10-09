from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

def read_fasta_files():
    files = os.listdir("proteins")
    sequences = []
    count = 0
    files_processed = []
    labels = []

    for file in files:
        if file.endswith(".fasta"):
            for seq_record in SeqIO.parse("proteins/"+file, "fasta"):
                if len(seq_record.seq) > 2:
                    sequences.append(str(seq_record.seq))
                    count += 1
                    files_processed.append(file)
                    description = seq_record.description
                    
                    match = re.search(r'\((.*?)\)', description)
                    
                    if match:
                        label = match.group(1)
                        labels.append(label)
                    else:
                        pass
                
    
    return sequences, labels, files_processed


def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def tfidf_method(sequences):

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(sequences)

    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    for column in df_tfidf.columns:
        max_value = df_tfidf[column].max()
        max_index = df_tfidf[column].idxmax()
        print(f"Columna: {column} | Valor más alto: {max_value} | Fila: {max_index}")
    
    return tfidf_matrix


def cosine_similarity_method(tfidf_matrix, files_processed, plot=True):

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    df_cosine_sim = pd.DataFrame(cosine_sim, index=files_processed, columns=files_processed)

    n = df_cosine_sim.shape[0]

    resultados = []

    for i in range(n):
        for j in range(i + 1, n):
            seq1 = df_cosine_sim.index[i]
            seq2 = df_cosine_sim.index[j]
            similarity = df_cosine_sim.iloc[i, j]

            resultados.append((seq1, seq2, round(similarity, 2)))

    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cosine_sim, cmap='coolwarm', annot=False, xticklabels=files_processed, yticklabels=files_processed)
        plt.title('Matriz de Similitud de Coseno entre Proteínas')
        plt.xlabel('Proteínas')
        plt.ylabel('Proteínas')
        plt.show()
        
        print("\nLas 5 secuencias más similares son:")
        resultados.sort(key=lambda x: x[2], reverse=True)
        print(resultados[:5])



if __name__ == "__main__":
    sequences, labels, files_processed = read_fasta_files()
    k = 3 
    kmers = [generate_kmers(seq, k) for seq in sequences]
    sequences = [' '.join(kmer) for kmer in kmers]
    tfidf_matrix = tfidf_method(sequences)
    cosine_similarity_method(tfidf_matrix, files_processed)