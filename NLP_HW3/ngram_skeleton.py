import math, random

################################################################################
# Utility Functions
################################################################################

def start_pad(c):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    padded_text = start_pad(c) + text
    return [(padded_text[i:i+c], padded_text[i+c]) for i in range(len(text))]

def create_ngram_model(model_class, path, c=2):
    ''' Creates and returns a new n-gram model '''
    model = model_class(c)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model WITH smoothing, because something had the perplexity misbehaving otherwise '''

    def __init__(self, c):
        self.c = c
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set()

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for context, char in ngrams(self.c, text):
            if context not in self.ngram_counts:
                self.ngram_counts[context] = {}
            if char not in self.ngram_counts[context]:
                self.ngram_counts[context][char] = 0
            self.ngram_counts[context][char] += 1

            if context not in self.context_counts:
                self.context_counts[context] = 0
            self.context_counts[context] += 1

            self.vocab.add(char)

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        alpha = 1  # SMOOTHING! Read that article!
        vocab_size = len(self.vocab)
        
        if context in self.ngram_counts:
            count = self.ngram_counts[context].get(char, 0) + alpha
            total = self.context_counts[context] + (alpha * vocab_size)
        else:
            count = alpha
            total = vocab_size * alpha

        return count / total

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        if context in self.ngram_counts:
            chars, weights = zip(*self.ngram_counts[context].items())
            return random.choices(chars, weights=weights)[0]
        else:
            return random.choice(list(self.vocab)) if self.vocab else ''

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        context = start_pad(self.c)
        generated_text = []
        
        for _ in range(length):
            char = self.random_char(context)
            generated_text.append(char)
            context = (context + char)[-self.c:]
            
        return ''.join(generated_text)

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model - NOTE: edited somewhat 

        Acknowledgment: 
          https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3 
          https://courses.cs.washington.edu/courses/csep517/18au/
          ChatGPT with GPT-3.5
        '''
        # TODO debug and remove
        if not self.vocab:
            return float('inf')
        
        # Remove any unseen characters
        text = ''.join([c for c in text if c in self.vocab])
        N = len(text)

        # Handle very short text cases
        if N < self.c:
            return float('inf')
        
        # # Calculate product of the inverse probabilities of the text according to the model
        # char_probs = []
        # for i in range(self.c, N):
        #     context = text[i-self.c+2:i]
        #     char = text[i]
        #     char_probs.append(math.log2(self.prob(context, char)))
        # ppl = 2 ** (-1 * sum(char_probs) / (N - self.c + 2))

        # return ppl

        log_prob_sum = 0
        for i in range(self.c, N):
            context = text[i-self.c:i]
            char = text[i]
            prob = self.prob(context, char)

            if prob > 0:
                log_prob_sum += math.log2(prob)
            else:
                return float('inf')

        return 2 ** (-log_prob_sum / N)
