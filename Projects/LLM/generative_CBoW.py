import nltk
import numpy as np
from utils.get_batches import get_batches
from utils.compute_pca import compute_pca
from utils.get_dict import get_dict
import re
from matplotlib import pyplot

with open("/Users/eunie/nltk_data/corpora/gutenberg/shakespeare-hamlet.txt") as f:
	data = f.read() # 훈련을 위한 말뭉치 생성

data = re.sub(r"[,!?;-]", ".", data) # 문장부호 제거
data = nltk.word_tokenize(data) # 단어 단위 토큰화
data = [ch.lower() for ch in data if ch.isalpha() or ch == "."] # 소문자 변환
print("Number of tokens: ", len(data), "\n", data[500:515])

fdist = nltk.FreqDist(word for word in data) # 단어 주머니와 분포를 가져옴
print("Size of vocabulary: ", len(fdist))
print("Most Frequent Tokens: ", fdist.most_common(20))

word2Ind, Ind2word = get_dict(data) # 변환 시간을 단축하고 어휘를 추적하기 위한 dictionary 객체 두 개 선언
V = len(word2Ind)
print("Size of vocabulary: ", V)

print("Index of the word 'king': ", word2Ind["king"])
print("Word which has index 2743: ", Ind2word[2743])

# 은닉층 하나와 매개변수 두 개짜리 신경망 모델을 만드는 함수
def initialize_model(N, V, random_seed=1):
	"""
	:param N: 은닉 벡터의 차원
	:param V: 어휘의 차원
	:param random_seed: 테스트 시 일관된 결과를 위한 난수 시드값
	:return:
		- W1, W2, b1, b2: 초기화된 가중치들과 편향값들
	"""
	np.random.seed(random_seed)

	W1 = np.random.rand(N, V)
	W2 = np.random.rand(V, N)
	b1 = np.random.rand(N, 1)
	b2 = np.random.rand(V, 1)

	return W1, W2, b1, b2

# 최종 분류층에 사용할 소프트맥스 함수
# 모든 확률의 합 == 1
def softmax(z):
	"""
	:param z: 은닉층의 출력 점수들
	:return:
		- yhat: 예측값(y의 추정치)
	"""
	yhat = np.exp(z) / np.sum(np.exp(z), axis=0)
	return yhat

# forward pass
# 활성화 함수와 함께 입력이 모델을 순방향으로 통과하는 방법을 정의
def forward_prop(x, W1, W2, b1, b2):
	"""
	:param x: context를 위한 평균 one-hot 벡터
	:param W1, W2, b1, b2: 학습할 가중치와 편향들
	:return:
		- z: 출력 점수 벡터
	"""
	h = W1 @ x + b1
	h = np.maximum(0, h)
	z = W2 @ h + b2
	return z, h

# 비용 계산 함수
# 실측값과 모델 예측 사이의 거리 측정 방법을 정의
def compute_cost(y, yhat, batch_size):
	logprobs = np.multiply(np.log(yhat), y) + np.multiply(
		np.log(1 - yhat), 1 - y
	)
	cost = -1 / batch_size * np.sum(logprobs)
	cost = np.squeeze(cost)
	return cost

# backword pass 함수
# 모델을 역방향으로 통과하며 기울기들을 취합하는 역전파 방법을 정의
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
	"""
	:param x: context를 위한 평균 one-hot 벡터
	:param yhat: 예측갑(y의 추정치)
	:param y: 목표 벡터
	:param h: 은닉 벡터
	:param W1, W2, b1, b2: 가중치들과 편향들
	:param batch_size: 배치(일괄 처리 단위) 크기
	:return:
		- grad_W1, grad_W2, grad_b1, grad_b2: 가중치 기울기들과 편향 기울기들
	"""
	l1 = np.dot(W2.T, yhat - y)
	l1 = np.maximum(0, l1)
	grad_W1 = np.dot(l1, x.T) / batch_size
	grad_W2 = np.dot(yhat - y, h.T) / batch_size
	grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size
	grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / batch_size

	return grad_W1, grad_W2, grad_b1, grad_b2

# gradient descent 함수
# 모든 요소를 통합해서 train
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
	"""
	:param data:  text
	:param word2Ind:  word-idx 매핑
	:param N: 은닉 벡터 차원 수
	:param V: 어휘 차원 수
	:param num_iters: 반복 횟수
	:return:
		- W1, W2, b1, b2: 갱신된 가중치들과 편향들
	"""
	W1, W2, b1, b2 = initialize_model(N, V, random_seed=8855)
	batch_size = 128
	iters = 0
	C = 2
	for x, y in get_batches(data, word2Ind, V, C, batch_size):
		z, h = forward_prop(x, W1, W2, b1, b2)
		yhat = softmax(z)
		cost = compute_cost(y, yhat, batch_size)
		if (iters + 1) % 10 == 0:
			print(f"iters: {iters+1} cost: {cost:.6f}")
		grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(
			x, yhat, y, h, W1, W2, b1, b2, batch_size
		)
		W1 = W1 - alpha * grad_W1
		W2 = W2 - alpha * grad_W2
		b1 = b1 - alpha * grad_b1
		b2 = b2 - alpha * grad_b2
		iters += 1
		if iters == num_iters:
			break
		if iters % 100 == 0:
			alpha *= 0.66

	return W1, W2, b1, b2

# 모델 train 시작
C = 2
N = 50
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
num_iters = 150
print("Call gradient_descent")
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)