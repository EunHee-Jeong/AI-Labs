import argparse
import torch

from models.simple_cnn import SimpleCNN
from datasets.load_mnist import get_dataloaders
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
    parser = argparse.ArgumentParser(description='간단한 CNN 모델 구현(MNIST 숫자 분류기)')

    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    main(args)