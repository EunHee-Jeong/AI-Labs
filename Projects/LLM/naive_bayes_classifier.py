from utils.lookup import lookup
from utils.process_utt import process_utt
from nltk.corpus.reader import PlaintextCorpusReader
import numpy as np

my_corpus = PlaintextCorpusReader("/Users/eunie/nltk_data/corpora/gutenberg", r".*\.txt")

sents = my_corpus.sents(fileids="shakespeare-hamlet.txt")

# 전처리+카운팅 함수
def count_utts(result, utts, ys):
	"""
	:param result: 각 튜플을 해당 빈도에 대응시킨 dictionary 객체
	:param utts: 발화(utterence)들의 목록(list)
	:param ys: 발화가 나타내는 긍정적/부정적 감성(0 또는 1)들의 목록
	:return: 각 튜플을 해당 빈도에 대응시킨 dictionary 객체
	"""

	for y, utt in zip(ys, utts):
		for word in process_utt(utt): # process_utt(): 단어들을 전처리
			pair = (word, y) # 단어-label 튜플로 이루어진 key 정의

			if pair in result:
				result[pair] += 1 # 해당 key가 dictionary 객체에 있으면 출현 횟수(빈도) 카운팅 증가
			else:
				result[pair] = 1 # 새로운 key이면 dictionary 객체에 추가하고 출현 횟수 세팅

	return result


result = {}
utts = [" ".join(sent) for sent in sents] # 문장 단위로 끊음
ys = [sent.count("be") > 0 for sent in sents] # be가 포함되면 긍정(1), 없으면 부정(0)
"""
⭐️⭐ [sent.count("be") > 0 for sent in sents]:
	의미가 있는 감성 label이 아닌, 단순 실험용이기 때문에
	실제로 분류 task를 수행할 때는 별도로 정답 label을 만들어야함.
"""
count_utts(result, utts, ys)

freqs = count_utts({}, utts, ys)
lookup(freqs, "be", True) # (word, label) 쌍의 빈도를 return
for k, v in freqs.items():
	if "be" in k:
		print(f"{k}:{v}")

# 각 단어의 log-likelihood 계산 (-> 다 합하면 문장이 긍정일 log 확률 추정 가능)
def train_naive_bayes(freqs, train_x, train_y):
	"""
	:param freqs: (단어, label) 튜플을 해당 단어의 빈도와 대응시킨 dictionary 객체
	:param train_x: 발화들의 목록
	:param train_y: 발화의 label(0 또는 1)들의 목록
	:return:
		- logprior: 로그 dictionary 확률
		- loglikelihood: 단순 bayes 방정식의 로그우도(로그가능도)
	"""
	loglikelihood = {}
	logprior = 0

	#  어휘 사전의 고유 단어 수 V 계산
	vocab = set([pair[0] for pair in freqs.keys()])
	V = len(vocab)

	N_pos = N_neg = 0 # 긍정적 단어 개수 N_pos와 부정적 단어 개수 N_neg 초기화
	for pair in freqs.keys():
		if pair[1] > 0: # 감성 분류를 나타내는 label이 양수이면
			N_pos += lookup(freqs, pair[0], True) # 긍정적 단어 튜플 (단어, label)의 카운팅 증가
		else:
			N_neg += lookup(freqs, pair[0], False)

	D = len(train_y) # 전체 문서 개수

	D_pos = sum(train_y) # 긍정적 문서의 개수

	D_neg = D - D_pos # 부정적 문서 개수

	logprior = np.log(D_pos) - np.log(D_neg) # 로그 사전확률 계산

	for word in vocab: # 어휘의 각 단어에 대해
		freq_pos = lookup(freqs, word, 1)
		freq_neg = lookup(freqs, word, 0)

		# 주어진 단어가 긍정적일 확률과 부정적일 확률을 계산
		# 등장하지 않은 단어의 확률이 0이 되는 것을 방지하기 위해 라플라스 스무딩 사용
		p_w_pos = (freq_pos + 1) / (N_pos + V)
		p_w_neg = (freq_neg + 1) / (N_neg + V)

		# 단어의 로그우도 계산
		loglikelihood[word] = np.log(p_w_pos / p_w_neg)

	return logprior, loglikelihood

# 문장을 입력으로 받아 logprior + 각 단어의 loglikelihood를 더한 총합을 출력
# 출력값이 0보다 크면 긍정(1), 작으면 부정(0)
def naive_bayes_predict(utt, logprior, loglikelihood):
	"""
	:param utt: 발화를 담은 문자열
	:param logprior: 로그 사전확률
	:param loglikelihood: 단어별 로그우도를 담은 dictionary 객체
	:return:
		- p: 모든 로그우도 + 로그 사전확률의 합
	"""
	word_l = process_utt(utt) # 발화(utterance)들을 처리한 단어 list를 가져옴

	p = 0

	p += logprior

	for word in word_l:
		if word in loglikelihood: # 단어가 로그우도 dictionary 객체에 존재하다면
			p += loglikelihood[word] # 해당 단어의 로그우도를 확률에 더함

	return p

# 단순 정확도 측정
def evaluate_naive_bayes(test_x, test_y, logprior, loglikelihood):
	"""
	:param test_x: 발화들의 목록
	:param test_y: 발화에 대응하는 label들
	:param logprior: 로그 사전확률
	:param loglikelihood: 단어별 로그우도를 담은 dictionary 객체
	:return:
		- accuracy: (정확히 분류된 발화 개수) / (전체 발화 개수)
	"""
	accuracy = 0

	y_hats = []
	for utt in test_x:
		if naive_bayes_predict(utt, logprior, loglikelihood) > 0: # 예측값이 0보다 크면
			y_hat_i = 1 # 예측 클래스는 1
		else:
			y_hat_i = 0

		y_hats.append(y_hat_i) # 예측 클래스를 y_hats 목록에 추가

	# 오차
	error = sum(
		[abs(y_hat - test) for y_hat, test in zip(y_hats, test_y)]
	) / len(y_hats)

	accuracy = 1 - error

	return accuracy

if __name__ == "__main__":
	logprior, loglikelihood = train_naive_bayes(freqs, utts, ys)
	print(logprior)
	print(len(loglikelihood))

	my_utt = "To be or not to be, that is the question."
	p = naive_bayes_predict(my_utt, logprior, loglikelihood)
	print("The expected output is ", p)

	print(
		f"Naive Bayes accuracy = {evaluate_naive_bayes(utts, ys, logprior, loglikelihood):0.4f}"
	)