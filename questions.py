import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    all_files = {}
    for file in os.listdir(directory):
        with open(os.path.join(directory,file),encoding="utf-8") as f:
            all_files[file] = f.read()
    return all_files
    # raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stop_words = nltk.corpus.stopwords.words('english')
    words = nltk.tokenize.word_tokenize(document.lower())
    for word in words:
        if word in string.punctuation or word in stop_words:
            words.remove(word)
    return words
    # raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_words = dict()
    for files in documents.keys():
        for word in documents[files]:
            idf_words[word] = 0
    
    for word in idf_words.keys():
        for files in documents.keys():
            if word in documents[files]:
                idf_words[word]+=1
        
    
    for word in idf_words.keys():
        idf_words[word] = math.log(len(documents)/idf_words[word])
    
    return idf_words
    # raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    rank = dict()
    for fi in files.keys():
        local_rank = 0
        for word in query:
            if word in idfs.keys():
                local_rank+=files[fi].count(word) * idfs[word]
        rank[fi] = local_rank
    
    return sorted(list(rank.keys()),key=lambda f:rank[f],reverse= True)[:n]
    # raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    rank = {
        sentence:[0,0]
        for sentence in sentences.keys()
    }
    for sentence in sentences.keys():
        for word in query:
            if word in sentences[sentence] and word in idfs.keys():
                rank[sentence][0]+=idfs[word]
                rank[sentence][1] = query_term_density(query,sentence,sentences)

    topSentences = sorted(rank.keys(),key= lambda x:rank[x],reverse = True)
    return topSentences[:n]

    raise NotImplementedError

def query_term_density(query,sentence,sentences):
    count = 0
    for word in query:
        if word in sentences[sentence]:
            count+=1
    return count/len(sentences[sentence])

if __name__ == "__main__":
    main()
