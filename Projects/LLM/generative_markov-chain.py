import re
import random
from nltk.tokenize import word_tokenize
from collections import defaultdict, deque

# 2차 Markov-Chain을 기반으로 문장을 자동 생성하는 간단한 text 생성기
# 핵심: 단어쌍의 연쇄 확률 구조를 이용해 문장 생성
class MarkovChain:
	def __init__(self):
		self.lookup_dict = defaultdict(list) # markov-chain의 핵심 구조 (예: { "to": ["be", "go"], "be": ["or", "gone"], ... })
		self._seeded = False
		self.__seed_me()

	# random seed 설정
	def __seed_me(self, random_seed=None):
		if self._seeded is not True: # 중복 초기화를 방지하기 위한 플래그
			try:
				if random_seed is not None:
					random.seed(random_seed)
				else:
					random.seed()
				self._seeded = True
			except NotImplementedError:
				self._seeded = False

	# 문서 입력 및 학습
	def add_document(self, str):
		preprocessed_list = self._preprocess(str) # text를 토큰화
		pairs = self.__generate_tuple_keys(preprocessed_list)
		for pair in pairs:
			self.lookup_dict[pair[0]].append(pair[1]) # 연속 단어쌍을 (앞단어->뒷단어) 형태로 저장

	# 텍스트 전처리
	def _preprocess(self, str):
		cleaned = re.sub(r"\W+", " ", str).lower() # 문장부호 제거 + 소문자 변환
		tokenized = word_tokenize(cleaned) # 단어 단위 분할
		return tokenized # 예: "To be, or not to be." → ["to", "be", "or", "not", "to", "be"]

	# 단어쌍 생성기
	def __generate_tuple_keys(self, data):
		if len(data) < 1:
			return

		for i in range(len(data) - 1):
			yield [data[i], data[i + 1]] # 예: ["to", "be"], ["be", "or"], ["or", "not"], ...

	# 텍스트 생성기
	def generate_text(self, max_length=50):
		context = deque()
		output = []
		if len(self.lookup_dict) > 0:
			self.__seed_me(random_seed=len(self.lookup_dict)) # 시작 단어를 랜덤으로 정함
			chain_head = [list(self.lookup_dict)[0]]
			context.extend(chain_head)

			while len(output) < (max_length - 1):
				next_choices = self.lookup_dict[context[-1]] # 직전에 나온 단어를 기반으로 다음 단어 후보들을 가져옴
				if len(next_choices) > 0:
					next_word = random.choice(next_choices) # 랜덤하게 하나 선택
					context.append(next_word)
					output.append(context.popleft())
				else:
					break
			output.extend(list(context))
		return " ".join(output) # 이어 붙이기

if __name__ == "__main__":
	with open("/Users/eunie/nltk_data/corpora/maroon.txt", "r", encoding="utf-8") as f:
		text = f.read()
	HMM = MarkovChain()
	HMM.add_document(text) # maroon 전체 가사를 학습

	print(HMM.generate_text(max_length=25)) # markov-chain을 이용해 길이 25짜리 문장을 생성

