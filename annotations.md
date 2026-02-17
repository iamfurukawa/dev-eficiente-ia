BPE(Byte Pair Encoding) - Quando temos um vocabulário muito grande, podemos usar o BPE para reduzir o vocabulário.

## Tipos de Recuperação
- Boolean Retrieval - É um modelo de recuperação de informações que usa operadores booleanos (AND, OR, NOT) para buscar documentos.
- VSM (Vector Space Model) - É um modelo de recuperação de informações que usa vetores para representar documentos e consultas.
- Probabilistic Retrieval - É um modelo de recuperação de informações que usa probabilidades para buscar documentos.

## TF-IDF
- TF (Term Frequency) - É a frequência de um termo em um documento. TF(t,d) = Ocorrencia de t no documento d / total de termos no documento d
- IDF (Inverse Document Frequency) - É a inversa da frequência de um termo em todos os documentos. IDF(t) = log(Numero total de documentos / Numero de documentos que contem t)
- TF-IDF - É a combinação de TF e IDF. TF-IDF(t,d) = TF(t,d) * IDF(t)

## Para stop-words
nltk.download("stopwords")
stop_words = nltk.corpus.stopwords.words("portuguese")