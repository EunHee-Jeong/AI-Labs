"""
문자 단위 Vanilla RNN 구현
참고: Andrej Karpathy (@karpathy)
"""

import numpy as np
import requests

# 1. 데이터 로드
url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
response = requests.get(url)
response.raise_for_status()  # 오류 발생 시 예외 던짐

# 바이너리 콘텐츠를 파일로 저장
with open('shakespeare.txt', 'wb') as f:
	f.write(response.content)

# 텍스트로 읽기
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
	data = f.read()

# 2. 어휘(고유 문자) 추출, 크기 계산
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

print(f"데이터에는 {data_size}개의 문자, {vocab_size}개의 고유 문자가 있습니다.")

# 3. 문자 ↔ 인덱스 매핑
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 4. Hyperparameter 설정 (히든 레이어, 시퀀스 길이, 학습률)
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# 5. 모델 파라미터(가중치, 편향) 초기화
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


# 6. lossFunc
def lossFunc(inputs, targets, hprev):
	"""
	:param inputs: 정수 인덱스 리스트
	:param targets: 정수 인덱스 리스트
	:param hprev: H X 1 크기의 초기 hidden state 배열
	:return: 손실, 각 파라미터의 기울기, 마지막 hidden state
	"""
	xs, hs, ys, ps = {}, {}, {}, {}  # 시간별 input, hidden state, output, softmax 확률을 담는 사전
	hs[-1] = np.copy(hprev)  # t=0일 때는 h_{-1} 들어감 (루프 시작 전에 초기화 필요), 원본 보존을 위해 copy->forward, backward 과정에서 일관되게 사용
	loss = 0

	# 순전파 (Forward Pass)
	for t in range(len(inputs)):
		xs[t] = np.zeros((vocab_size, 1))  # vocab_size X 1
		xs[t][inputs[t]] = 1  # one-hot encodding
		hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)  # hidden state
		ys[t] = Why @ hs[t] + by  # 다음 문자에 대한 logit
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # softmax 확률
		loss += -np.log(ps[t][targets[t], 0])  # CrossEntropy 손실을 loss에 누적

	# 역전파 (Backward Pass)
	dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	dhnext = np.zeros_like(hs[0])

	for t in reversed(range(len(inputs))):  # 각 가중치와 편향 기울기를 누적
		dy = np.copy(ps[t])

		# 1. 출력층
		dy[targets[t]] -= 1  # backpropagation (Softmax + CrossEntropy 미분: dy = p_t - y_{one-hot})
		# TODO: - 여기서부터는 논문 읽고 나서 다시 보기.
		dWhy += dy @ hs[t].T  # 은닉층→출력층 가중치 기울기 누적
		dby += dy

		# 2. 은닉층
		dh = Why.T @ dy + dhnext  # 은닉층으로 전파
		dhraw = (1 - hs[t] * hs[t]) * dh  # tanh 미분
		dbh += dhraw
		dWxh += dhraw @ xs[t].T
		dWhh += dhraw @ hs[t - 1].T

		# 3. 입력층
		dhnext = Whh.T @ dhraw

	# 기울기 클리핑 (기울기 폭주 방지)
	for dparam in (dWxh, dWhh, dWhy, dbh, dby):
		np.clip(dparam, -5, 5, out=dparam)

	return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]  # hs[len(inputs)-1]는 마지막 은닉 상태

# 7. 텍스트 생성용 sample 함수
def sample(h, seed_idx, n):
	"""
	:param h: 초기 은닉 상태
	:param seed_idx: 시드 문자 인덱스
	:param n: 생성할 길이
	:return: 모델로부터 생성한 시퀀스
	"""
	x = np.zeros((vocab_size, 1))
	x[seed_idx] = 1
	idxes = [] # 인덱스를 담을 리스트
	for _ in range(n):
		# 매 step마다 확률 분포 p에 따라 다음 문자 인덱스를 샘플링
		# 총 n 길이의 글자 생성
		h = np.tanh(Wxh @ x + Whh @ h + bh)
		y = Why @ h + by
		p = np.exp(y) / np.sum(np.exp(y))
		idx = np.random.choice(range(vocab_size), p=p.ravel()) # ravel()은 flatten 함수임
		x = np.zeros((vocab_size, 1))
		x[idx] = 1
		idxes.append(idx)
	return idxes

# 8. 학습 루프 초기화
n, p = 0, 0 # 반복 카운터 n, 데이터 포인터 p
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why) # Adagrad용 누적 기울기 제곱 m* 초기화
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # Adagrad 메모리
smooth_loss = -np.log(1.0 / vocab_size) * seq_length # 초기 손실 (지수평균이동 == '무작위 예측' 시 손실의 기대값임)

# 9. 미니배치 생성
while True:
	# 데이터 준비 (seq_length 길이씩)
	if p + seq_length + 1 >= len(data) or n == 0: # 데이터 포인터가 끝에 가까워지거나, 첫 반복이라면
		hprev = np.zeros((hidden_size, 1)) # 은닉 상태 초기화
		p = 0 # 파일 맨 앞으로
	inputs = [char_to_idx[ch] for ch in data[p: p+seq_length]] # 현재 블록 문자 인덱스
	targets = [char_to_idx[ch] for ch in data[p+1: p+seq_length+1]] # 다음 문자 인덱스

	# 중간 샘플 출력
	# 매번 100회차마다 200글자를 생성
	if n % 100 == 0:
		sample_idx = sample(hprev, inputs[0], 200)
		txt = ''.join(idx_to_char[idx] for idx in sample_idx)
		print(f"----\n{txt}\n----")

	# 손실 계산 및 출력
	# 순전파, 역전파
	loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFunc(inputs, targets, hprev)
	smooth_loss = smooth_loss * 0.999 + loss * 0.001
	if n % 100 == 0:
		print(f"반복 {n}회차, 손실: {smooth_loss:.6f}")

	# Adagrad 파라미터 업데이트
	for param, dparam, mem in zip(
			(Wxh, Whh, Why, bh, by),
			(dWxh, dWhh, dWhy, dbh, dby),
			(mWxh, mWhh, mWhy, mbh, mby)
	):
		mem += dparam * dparam # 각 파라미터에 대해 누적 기울기 제곱 mem 갱신
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # 학습률을 기울기로 나눠 adaptive 업데이트

	# 포인터 반복, 카운터 갱신
	p += seq_length # 다음 블록으로 넘어가기 위해 p 증가시킴
	n += 1 # 전체 반복 수 n을 증가시키고, 루프 재시작