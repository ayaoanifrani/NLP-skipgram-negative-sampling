****************
# 1. Preprocessing

First, we use the function `text2sentences` as initially implemented. Consequences are numerous:
- training was too long
- train set was too big and polluted with noisy word like `"'clock"` for example (which obviously comes from `"o'clock"`)
- the performance was not good as the contexts are somehow meaningless (too many noisy words)

#### PUNCTUATION
To usual practice in NLP, we tried to get preprocess punctuation characters. Two ideas were to simply get rid of them or transform them into one and only token : `"<PUNC>"`
As we didn't see any significant difference between the two models after training, we simply got rid of the punctuation.

#### NEGATION
As the main interest of words like `"not"`, `"'nt"` or `"nt"` is the negation effect. So, we logically transform them to a unique token `"<NEGATIVE>"`.

#### NUMBERS
Same as for negations, we convert all numbers into an only token `"<NUMBER>"`. This is specifically for some words which generally have numbers in their context (for example `"days"`, `"hours"`, `"years"` which are generally preceded by a number).
The important thing here is not the number's value but simply the fact that it is a number.

#### ADITIONAL PREPROCESSING
- We decided to split composite words constituted with many word joined by a dash. For example, `"three-year-old-child"` is less interesting for us than `"three year old child"` as the latter gives a better context.
- For numbers tokenization, we also consider their letter forms (`"twenty"` is processed as `"20"`)
- We also used the stop words from spacy library to get rid of some stop words which appear in too much contexts to give enough information.



# 2. The model

## 2.1 Sampling: unigram table

To make the sampling process, fast, we used the unigram table model as in the original code by Mikolov and al. and explained here: http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
This is a table of length `1e8` in which all the words in vocabulary are present with a distribution following a power law of parameter 3/4.
As this was mainly a question of execution time, we didn't change the method. It provides a sample of 5 negative words within broadly `1e-5` second.
We also did some modifications on the 3/4 law but the loss decreased faster with a 3/4 as said in Mikolov and al.'s paper. We didn't go further on this.

## 2.2 Pre-training: the ``train`` function
First, we delete the filtering part of the function 
```python
sentence = filter(lambda word: word in self.vocab, sentence)
```
which was very time consuming and not necessary because there is no need to filter again.
This is important as in our particular case, as we kept this for a long time (we transformed the filter object into a list object to avoid any error), our model was spending about 2 to 3 minutes per iteration of training (15 to 20 seconds now).
The, we tested a lot of modifications to make the function faster but the improvement was not very important and we finally came back to a simpler version:
- Choosing a window size ``size`` with all in once before the ``wpos`` and ``word`` for loop. The idea is to call the ``numpy.random.rand`` only once and keep an array instead of calling it at every iteration.
	We didn't keep this because it didn't give a faster implementation.
- For each word, we give all the context words at once to the ``trainWord`` function to avoid a for loop. We use the same negative words for all context words.
	This one was very interesting to observe because for an unknown reason, the loss function was decreasing slower. But the function itself was faster. Thus, we can train more epochs on the train set.
	In result, this is equivalent to training on one epoch but considering a new set of negative words for each context word.
- All our other modifications were to make the code faster.

## 2.3 Training: ``trainWord`` function
This is the most interesting part. First, we split it into two part with an another function called ``gradient`` which perform the computation of the gradients. But there is no need because the gradients can be performed in one line each and it is faster to keep them here.
The modifications of this function are mainly to adapt it to the ``train`` function.
Also, we implement the formulas in some weird forms just to make the computation faster. A sanity check has been perform on this.
The function also helps to find the good values for the learning rate.

It is important to notice that only one step of gradient descent is performed at every call of the function ``trainWord``. We have tried to see what happens when we perform more steps on every couple of (word, context word). The results are in our report.



# 3. Some specifity

To save or load a model, our preferred file type is .zip. Here we can see the parameters of the model without loading it. And also it is easier to load some of the values of a model (vectors for example)
	

*****************
# References:
- Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.  *Distributedrepresentations of words and phrases and their compositionality*, 2013
- Chris McCormick. *Word2vec Tutorial Part 2: Negative Sampling*. 11 Jan 2017. http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
- Yoav Goldberg and Omer Levy. word2vec Explained: *Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method*, 2014
- Eric Kim. *Optimize Computational Efficiency of Skip-Gram with Negative Sampling*. 26 May 2019. https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling