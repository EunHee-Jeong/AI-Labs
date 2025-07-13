import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
	def __init__(
			self,
			input_size,
			hidden_size=2,
			output_size=3,
			num_hidden_layers=1,
			hidden_activation=nn.Sigmoid,
	):
		"""
		:param input_size (int): 입력 크기
		:param hidden_size (int): 은닉층 크기
		:param output_size (int): 출력 크기
		:param num_hidden_layers (int): 은닉층 개수
		:param hidden_activation (torch.nn.*): 활성화 클래스
		"""
		super(MultiLayerPerceptron, self).__init__()
		self.module_list = nn.ModuleList()
		interim_input_size = input_size
		interim_output_size = hidden_size
		torch.device("cuda:0" if torch.cuda.is_available else "cpu")

		for _ in range(num_hidden_layers):
			self.module_list.append(
				nn.Linear(interim_input_size, interim_output_size)
			)
			self.module_list.append(hidden_activation())
			interim_input_size = interim_output_size

		self.fc_final = nn.Linear(interim_input_size, output_size)

		self.last_forward_cache = []

	def forward(selfself, x, apply_softmax=False):
		"""
		:param x (torch.Tensor): 입력 데이터 텐서
			- x.shape는 반드시 (batch, input_dim)
		:param apply_softmax (bool): softmax 활성화 함수 적용 여부
			- 교차 엔트로피 손실 함수를 사용하는 경우에는 반드시 False
		:return: 계산된 출력 Tensor
			- 텐서의 .shape는 반드시 (batch, output_dim)
		"""
		for module in self.module_list:
			x = module(x)

		output = self.fc_final(x)

		if apply_softmax:
			output = F.softmax(output, dim=1)

		return output