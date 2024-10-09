from protein_analyzer import read_fasta_files, tfidf_method, cosine_similarity_method, generate_kmers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score 

sequences, labels, files_processed = read_fasta_files()

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

k = 3
X_train_kmers = [' '.join(generate_kmers(seq, k)) for seq in X_train]
X_test_kmers = [' '.join(generate_kmers(seq, k)) for seq in X_test]


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train_kmers)
X_train_tfidf = tfidf_vectorizer.transform(X_train_kmers)
X_test_tfidf = tfidf_vectorizer.transform(X_test_kmers)


lsv = LinearSVC()
lsv.fit(X_train_tfidf, y_train)
y_predicted = lsv.predict(X_test_tfidf)
lsv_accuracy = round(accuracy_score(y_test, y_predicted),2)

print(f"Accuracy: {lsv_accuracy}")

print("Tamaño de etiquetas de entrenamiento: ", len(y_train))
print("Valores únicos de etiquetas de entrenamiento: ", len(set(y_train)))

