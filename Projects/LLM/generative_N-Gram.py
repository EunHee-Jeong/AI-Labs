import nltk
nltk.download('gutenberg')

from nltk.corpus import gutenberg
print(gutenberg.fileids())

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.util import everygrams
from nltk.lm.preprocessing import (
	pad_both_ends,
	flatten,
	padded_everygram_pipeline,
)
from nltk.lm import MLE

my_corpus = PlaintextCorpusReader("/Users/eunie/nltk_data/corpora/gutenberg", r".*\.txt")
 # 다수의 일반 txt 파일로부터 말뭉치(corpus) 생성
for sent in my_corpus.sents(fileids="shakespeare-hamlet.txt"): # sents() -> 문장을 토큰 리스트로 분할
	print(sent)

padded_trigrams = list(
	pad_both_ends(my_corpus.sents(fileids="shakespeare-hamlet.txt")[1104], n=2)
	# 말뭉치에 있는 모든 행(line)의 양쪽에 발화의 시작과 끝을 나타내는 <s>와 </s>를 채워넣음 (= padding)
	# n=2는 3-gram을 만들기 위한 세팅 (n-1개의 context)
)
list(everygrams(padded_trigrams, max_len=3)) # everygrams: 1-gram, 2-gram, 3-gram 모두 생성

# 각 문장의 토큰 리스트를 1D 리스트로 이어 붙임
list(
	flatten(
		pad_both_ends(sent, n=2)
		for sent in my_corpus.sents(fileids="shakespeare-hamlet.txt")
	)
)

# NLTK 라이브러리의 everygram을 이용해 데이터로부터 훈련 데이터 집합과 어휘 객체를 생성
train, vocab = padded_everygram_pipeline(
	3, my_corpus.sents(fileids="shakespeare-hamlet.txt")
)

lm = MLE(3) # MLE 기반 n-gram 언어 모델
len(lm.vocab)

lm.fit(train, vocab)

print(lm.vocab)
len(lm.vocab)

lm.generate(6, ["to", "be"]) # 언어(문장) 생성, n-1개의 선행 토큰으로 조건 지정