"""MNIST 학습 스크립트 — SageMaker 컨테이너 '안'에서 실행됩니다.

이 파일은 내 노트북이 아니라, SageMaker가 띄운 AWS Deep Learning Container 안에서 돕니다.
그래서 torch / torchvision 을 로컬에 설치하지 않아도 됩니다 (컨테이너에 이미 있음).

SageMaker가 넣어주는 환경변수 두 가지만 알면 됩니다:
  SM_MODEL_DIR    이 경로에 모델을 저장하면 자동으로 S3(model.tar.gz)로 업로드됨.
  SM_NUM_GPUS     할당된 GPU 개수.
하이퍼파라미터(epochs 등)는 "--epochs 3" 같은 커맨드라인 인자로 전달됩니다.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    """간단한 CNN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)  # launch.py에서 전달됨
    # SageMaker가 지정하는 모델 저장 경로 (없으면 로컬 기본값)
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "./model"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)  # print는 CloudWatch Logs로 그대로 나감

    # MNIST 데이터 로드 (컨테이너가 인터넷에서 자동 다운로드)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tfm),
        batch_size=128, shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=tfm),
        batch_size=1000,
    )

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        # --- 학습 ---
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

        # --- 평가 ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                correct += model(data).argmax(1).eq(target).sum().item()
        acc = correct / len(test_loader.dataset)
        # 이 "test_accuracy=" 포맷을 launch.py 정규식이 파싱해 콘솔 그래프로 만듦
        print(f"epoch {epoch}: test_accuracy={acc:.4f}", flush=True)

    # SM_MODEL_DIR 에 저장 → SageMaker가 자동으로 S3의 model.tar.gz 로 업로드
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    print("saved model", flush=True)


if __name__ == "__main__":
    main()
