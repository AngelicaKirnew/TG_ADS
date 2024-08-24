#bibliotecas utilizadas no projeto
import pandas as pd
import spacy
import nltk
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

# Importe as stopwords do NLTK para o português
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('portuguese')

# Carregar o modelo do spaCy - lematização
nlp = spacy.load('pt_core_news_lg')

# Carregar os dados do arquivo CSV
data = pd.read_csv('noticias.csv')

# Inicialize o vetorizador TF-IDF com as stopwords em português
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)

# Separar os recursos (X) e os rótulos (y)
X = data['Text']
y = data['Label']

# Inicializar os modelos de aprendizado de máquina
nb_classifier = MultinomialNB()
rf_classifier = RandomForestClassifier()
svm_classifier = SVC()
nn_classifier = MLPClassifier()
dt_classifier = DecisionTreeClassifier()

# Inicializar as métricas de classificação
avaliacao = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Inicialização das listas para armazenar as matrizes de confusão
confusion_matrices_nb = []
confusion_matrices_rf = []
confusion_matrices_svm = []
confusion_matrices_nn = []
confusion_matrices_dt = []

# Inicializar o StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Loop de validação cruzada
for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Vetorizar os dados de treinamento e teste
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Treinar e avaliar os modelos de aprendizado de máquina
    nb_classifier.fit(X_train_tfidf, y_train)
    nb_pred = nb_classifier.predict(X_test_tfidf)
    confusion_matrices_nb.append(confusion_matrix(y_test, nb_pred))

    rf_classifier.fit(X_train_tfidf, y_train)
    rf_pred = rf_classifier.predict(X_test_tfidf)
    confusion_matrices_rf.append(confusion_matrix(y_test, rf_pred))

    svm_classifier.fit(X_train_tfidf, y_train)
    svm_pred = svm_classifier.predict(X_test_tfidf)
    confusion_matrices_svm.append(confusion_matrix(y_test, svm_pred))

    nn_classifier.fit(X_train_tfidf, y_train)
    nn_pred = nn_classifier.predict(X_test_tfidf)
    confusion_matrices_nn.append(confusion_matrix(y_test, nn_pred))

    dt_classifier.fit(X_train_tfidf, y_train)
    dt_pred = dt_classifier.predict(X_test_tfidf)
    confusion_matrices_dt.append(confusion_matrix(y_test, dt_pred))

    #inicializa as métricas de classificação
    accuracy_scores.append([accuracy_score(y_test, nb_pred), accuracy_score(y_test, rf_pred),
                            accuracy_score(y_test, svm_pred), accuracy_score(y_test, nn_pred),
                            accuracy_score(y_test, dt_pred)])

    precision_scores.append(
        [precision_score(y_test, nb_pred, pos_label='E'), precision_score(y_test, rf_pred, pos_label='E'),
         precision_score(y_test, svm_pred, pos_label='E'), precision_score(y_test, nn_pred, pos_label='E'),
         precision_score(y_test, dt_pred, pos_label='E')])

    recall_scores.append([recall_score(y_test, nb_pred, pos_label='E'), recall_score(y_test, rf_pred, pos_label='E'),
                          recall_score(y_test, svm_pred, pos_label='E'), recall_score(y_test, nn_pred, pos_label='E'),
                          recall_score(y_test, dt_pred, pos_label='E')])

    f1_scores.append([f1_score(y_test, nb_pred, pos_label='E'), f1_score(y_test, rf_pred, pos_label='E'),
                      f1_score(y_test, svm_pred, pos_label='E'), f1_score(y_test, nn_pred, pos_label='E'),
                      f1_score(y_test, dt_pred, pos_label='E')])

# Exibir as métricas médias para cada modelo
avg_accuracy = [sum(scores) / len(scores) for scores in zip(*accuracy_scores)]
avg_precision = [sum(scores) / len(scores) for scores in zip(*precision_scores)]
avg_recall = [sum(scores) / len(scores) for scores in zip(*recall_scores)]
avg_f1 = [sum(scores) / len(scores) for scores in zip(*f1_scores)]

# Exibir as matrizes de confusão para cada modelo
avg_confusion_nb = sum(confusion_matrices_nb)
avg_confusion_rf = sum(confusion_matrices_rf)
avg_confusion_svm = sum(confusion_matrices_svm)
avg_confusion_nn = sum(confusion_matrices_nn)
avg_confusion_dt = sum(confusion_matrices_dt)

# Resultados do treinamento
print("---Métricas Médias - Naive Bayes---")
print("Acurácia:", avg_accuracy[0])
print("Precisão:", avg_precision[0])
print("Recall:", avg_recall[0])
print("F1-score:", avg_f1[0])
print("---Matriz de Confusão - Naive Bayes---")
print(avg_confusion_nb)

print("---Métricas Médias - Random Forest---")
print("Acurácia:", avg_accuracy[1])
print("Precisão:", avg_precision[1])
print("Recall:", avg_recall[1])
print("F1-score:", avg_f1[1])
print("---Matriz de Confusão - Random Forest---")
print(avg_confusion_rf)

print("---Métricas Médias - SVM---")
print("Acurácia:", avg_accuracy[2])
print("Precisão:", avg_precision[2])
print("Recall:", avg_recall[2])
print("F1-score:", avg_f1[2])
print("---Matriz de Confusão - SVM---")
print(avg_confusion_svm)

print("---Métricas Médias - Neural Network---")
print("Acurácia:", avg_accuracy[3])
print("Precisão:", avg_precision[3])
print("Recall:", avg_recall[3])
print("F1-score:", avg_f1[3])
print("---Matriz de Confusão - Neural Network---")
print(avg_confusion_nn)

print("---Métricas Médias - Decision Tree---")
print("Acurácia:", avg_accuracy[4])
print("Precisão:", avg_precision[4])
print("Recall:", avg_recall[4])
print("F1-score:", avg_f1[4])
print("---Matriz de Confusão - Decision Tree---")
print(avg_confusion_dt)
