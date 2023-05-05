import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from heapq import nlargest

def summarize(text, n):
    """
    Function to summarize text using extractive text summarization
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words
    words = [word_tokenize(sentence.lower()) for sentence in sentences]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [[word for word in sentence if word not in stop_words] for sentence in words]

    # Calculate the word frequency for each word
    word_freq = {}
    for sentence in filtered_words:
        for word in sentence:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    # Calculate the sentence score for each sentence based on the word frequency
    sentence_scores = {}
    for i, sentence in enumerate(filtered_words):
        sentence_score = 0
        for word in sentence:
            if word in word_freq:
                sentence_score += word_freq[word]
        sentence_scores[i] = sentence_score

    # Get the top n sentences with the highest scores
    summary_sentences = nlargest(n, sentence_scores, key=sentence_scores.get)
    summary_sentences = [sentences[i] for i in summary_sentences]
    summary = ' '.join(summary_sentences)

    return summary

if __name__ == '__main__':
    # Sample text for testing the summarization function
    text = """
    Natural language processing (NLP) is a field of computer science, artificial intelligence, and computational
    linguistics concerned with the interactions between computers and human (natural) languages. As such, NLP is related
    to the area of human–computer interaction. Many challenges in NLP involve natural language understanding, that is,
    enabling computers to derive meaning from human or natural language input, and others involve natural language generation.
    """

    # Summarize the text and print the summary
    summary = summarize(text, 2)
    print(summary)
