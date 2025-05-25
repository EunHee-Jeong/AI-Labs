import argparse
import torch
from requests.packages import target
from torch.utils.data import DataLoader

from models.simple_cnn import SimpleCNN
from datasets.load_mnist import get_dataloaders
from notebooks.shape_trace import output
from train.train import train_model
from train.test import evaluate_model

def main(args):
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    model = SimpleCNN()

    if args.train:
        train_model(model, train_loader, lr=args.lr, epochs=args.epochs)

    if args.eval:
        evaluate_model(model, test_loader)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='간단한 CNN 모델 구현(MNIST 숫자 분류기)')
    #
    # parser.add_argument('--train', action='store_true', help='Train the model')
    # parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--epochs', type=int, default=3)
    # parser.add_argument('--lr', type=float, default=0.001)
    #
    # args = parser.parse_args()
    # main(args)
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super().__init__()
            self.n_hidden = torch.nn.Linear(n_feature, n_hidden) # 은닉층
            self.relu = torch.nn.ReLU(inplace=True)
            self.out = torch.nn.Linear(n_hidden, n_output) # 출력층
            self.softmax = torch.nn.Softmax(dim=n_output)
        def forward(self, x):
            x = self.hidden(x)
            x = self.relu(x) # 은닉층을 위한 랠루 활성화 함수
            x = self.out(x)
            x = self.softmax(x) # 출력층을 위한 소프트맥스 활성화 함수
            return x

    loss_fn = torch.nn.MSELoss(reduction='sum')
    y_pred = model(x)
    loss_fn = loss_fn(y_pred, y)

    loss_fn = nn.CrossEntropyLoss()
    input = torch.randn(5, 6, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()

    class DropoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(784, 1200)
            self.dropout1 = torch.nn.Dropout(0.5) # 50%의 노드를 무작위로 선택하여 사용하지 않겠다
            self.layer2 = torch.nn.Linear(1200, 1200)
            self.dropout2 = torch.nn.Dropout(0.5)
            self.layer3 = torch.nn.Linear(1200, 10)

        def forward(self, x):
            x = F.relu(self, layer1(x))
            x = self.dropout1(x)
            x = F.relu(self, layer2(x))
            x = self.dropout2(x)
            return self.layer3(x)

    class CustomDataset(Dataset):
        def __init__(self):
            self.x_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            self.y_data = [[12], [18], [11]]
            def __len__(self):
                return len(self.x_data)
            def __getitem__(self, idx):
                x = torch.FloatTensor(self.x_data[idx])
                y = torch.FloatTensor(self.y_data[idx])
                return x, y
    dataset = CustomDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=2, # 미니배치 크기로 2의 제곱수 사용
        shuffle=True # 데이터를 불러올 때마다 랜덤으로 섞어서 가져옴
    )