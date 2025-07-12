"""
문자 단위 LSTM 구현
참고: Andrej Karpathy의 Vanilla RNN 구조 기반
"""

import numpy as np
import requests
from distributed.objects import WhoHas

from Projects.RNN.Vanilla_RNN import vocab_size

# 데이터 로드
url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
response = requests.get(url)
with open('shakespeare.txt', 'wb') as f:
	f.write(response.content)
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
	data = f.read()

chars = list(set(data))
data_size, char_size = len(data), len(chars)
print(f"데이터에는 {data_size}개의 문자, {vocab_size}개의 고유 문자가 있습니다.")

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameter
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# LSTM 파라미터 초기화
# W: (4 X hidden_size, input_size + hidden_size)
# Wxh, Whh, b는 LTSM용, Why, by는 출력층용
Wxh = np.random.randn(4 * hidden_size, vocab_size) * 0.01
Whh = np.random.randn(4 * hidden_size, hidden_size) * 0.01
b = np.zeros((4 * hidden_size, 1))
Why = np.random.randn(vocab_size, hidden_size) * 0.01
by = np.zeros((vocab_size, 1))

# sigmoid 함수 정의
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# lossFunc
def lossFunc(inputs, targets, hprev, cprev):
	xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
	hs[-1] = np.copy(hprev)
	cs[-1] = np.copy(cprev)
	loss = 0

	# forward pass: LSTM 내부 계산 및 softmax 출력 확률 계산
	for t in range(len(inputs)):
		xs[t] = np.zeros((vocab_size, 1))
		xs[t][inputs[t]] = 1

		z = Wxh @ xs[t] + Whh @ hs[t - 1] + b
		i = sigmoid(z[0:hidden_size])
		f = sigmoid(z[hidden_size:2*hidden_size])
		o = sigmoid(z[2*hidden_size:3*hidden_size])
		g = tanh(z[3*hidden_size:4*hidden_size])

		cs[t] = f * cs[t - 1] + i * g
		hs[t] = o * np.tanh(cs[t])

		# loss: cross-entropy
		ys[t] = Why @ hs[t] + by
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
		loss += -np.log(ps[t][targets[t], 0])

	# backward pass: LSTM의 chain-rule을 기반으로 gradient 계산
	dWxh, dWhh, db = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(b)
	dWhy, dby = np.zeros_like(Why), np.zeros_like(by)
	dhnext = np.zeros_like(hs[0])
	dcnext = np.zeros_like(cs[0])

	for t in reversed(range(len(inputs))):
		# 1. softmax 출력에 대한 gradient 계산
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1

		# 2. 출력층 파라미터 gradient 계산
		dWhy += dy @ hs[t].T
		dby += dy

		# 3. 은닉 상태에 대한 gradient 계산
		dh = Why.T @ dy + dhnext

		# 4. 출력 게이트 o에 대한 gradient 계산
		do = dh * np.tanh(cs[t]) # h = o * tanh(o) -> chain-rule 적용
		do = do * o * (1 - o) # o * (1 - o): sigmoid 미분

		# 5. 셀 상태에 대한 gradient 계산 (현재 시점 기준)
		dc = dh * o * (1 - np.tanh(cs[t])**2) + dcnext
			# h = o * tanh(o) -> chain-rule 적용
			# dcnext: 다음 시점에서 넘어온 셀 상태 gradient도 더함 (long-term dependency)

		# 6. forget gate f에 대한 gradient 계산
		df = dc * cs[t - 1]
		df = df * f * (1 - f) # sigmoid 미분 적용

		# 7. input gate i와 gate g에 대한 gradient 계산
		di = dc * g
		di = di * i * (1 - i)

		dg = dc * i
		dg = dg * (1 - g ** 2) # tanh 미분은 1-g^2

		# 8. 4개의 gradient를 하나로 쌓기
		dz = np.vstack((di, df, do, dg)) # 4H X 1

		# 9. 파라미터에 대한 gradient 계산
		dWxh += dz @ xs[t].T
		dWhh += dz @ hs[t - 1].T
		db += dz

		# 10. 다음 시점으로 넘길 gradient 계산
		dhnext = Whh.T @ dz # 현재 시점에서 이전 시점으로 넘기는 은닉 상태의 gradient
		dcnext = dc * f # 다음 시점으로 넘길 셀 상태의 gradient (c는 시간이므로 누적됨)

	# gradient clipping
	for dpram in [dWxh, dWhh, db, dWhy, dby]:
		np.clip(dpram, -5, 5, out=dparam)

	return loss, dWxh, db, dWhy, dby, hs[len(inputs)-1], cs[len(inputs)-1]

def sample(h, c, seed_idx, n):
	x = np.zeros((vocab_size, 1))
	x[seed_idx] = 1
	idxes = []
	for _ in range(n):
		z = Wxh @ x + Whh @ h + b
		i = sigmoid(z[0:hidden_size])
		f = sigmoid(z[hidden_size:2*hidden_size])
		o = sigmoid(z[2*hidden_size:3*hidden_size])
		g = np.tanh(z[3*hidden_size:4*hidden_size])
		c = f * c + i * g
		h = o * np.tanh(c)
		y = Why @ h + by
		p = np.exp(y) / np.sum(np.exp(y))
		idx = np.random.choice(range(vocab_size), p=p.ravel())
		x = np.zeros((vocab_size, 1))
		x[idx] = 1
		idxes.append(idx)
	return idxes

# 학습 루프
n, p = 0, 0
mWxh, mWhh, mb = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(b)
mWhy, mby = np.zeros_like(Why), np.zeros_like(by)
smooth_loss = -np.log(1.0 / vocab_size) * seq_length

hprev = np.zeros((hidden_size, 1))
cprev = np.zeros((hidden_size, 1))

while True:
	if p + seq_length + 1 >= len(data) or n == 0:
		hprev = np.zeros((hidden_size, 1))
		cprev = np.zeros((hidden_size, 1))
		p = 0

	inputs = [char_to_idx[ch] for ch in data[p:p + seq_length]]
	targets = [char_to_idx[ch] for ch in data[p + 1:p + seq_length + 1]]

	if n % 100 == 0:
		sample_idx = sample(hprev, cprev, inputs[0], 200)
		txt = ''.join(idx_to_char[i] for i in sample_idx)
		print(f"----\n{txt}\n----")

	loss, dWxh, dWhh, db, dWhy, dby, hprev, cprev = lossFunc(inputs, targets, hprev, cprev)
	smooth_loss = smooth_loss * 0.999 + loss * 0.001

	if n % 100 == 0:
		print(f"반복 {n}회차, 손실: {smooth_loss:.4f}")

	for param, dparam, mem in zip(
			[Wxh, Whh, b, Why, by],
			[dWxh, dWhh, db, dWhy, dby],
			[mWxh, mWhh, mb, mWhy, mby]
	):
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

	p += seq_length
	n += 1