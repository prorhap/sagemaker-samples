# SageMaker 학습 예제 모음

Amazon SageMaker에서 모델을 학습시키는 방법을 여러 방식으로 보여주는 예제 저장소입니다.
모두 같은 MNIST 분류 문제를 쓰되, SageMaker에 학습을 맡기는 방식을 달리해서 각각의 구조와
차이를 비교할 수 있게 했습니다.

이 예제들은 내 노트북에서 IDE로 학습 코드를 작성하고, 터미널에서 명령을 실행해 SageMaker에
학습 Job을 제출하는 흐름을 기준으로 작성했습니다. SageMaker Studio나 노트북 인스턴스 안에서
실행하는 상황은 다루지 않습니다.

여기서 중요한 건 환경이 두 개로 나뉜다는 점입니다.

- **로컬 환경(내 노트북)** — 학습 코드를 작성하고, Job을 제출하고, 상태를 확인하는 곳입니다.
  여기에는 `sagemaker`, `boto3` 같은 제어용 패키지만 있으면 됩니다.
- **학습 컨테이너(AWS)** — 실제 학습이 GPU/CPU 위에서 도는 곳입니다. `torch` 같은 학습
  프레임워크와 CUDA는 여기에 들어갑니다.

그래서 `torch`를 로컬에 설치하지 않습니다. 학습 코드는 내 노트북이 아니라 AWS가 띄운
컨테이너 안에서 돌기 때문입니다. 로컬 환경이 가벼워지고, "무엇이 어디서 도는지"가 분명해집니다.

## 프로젝트별 uv 환경

각 예제는 독립된 프로젝트이고, 환경은 [uv](https://docs.astral.sh/uv/)로 관리합니다.
프로젝트마다 `pyproject.toml`, `uv.lock`, `.python-version`을 두어 서로 간섭 없이 정확히
같은 의존성을 재현할 수 있게 했습니다.

사용법은 프로젝트 폴더로 들어가 `uv sync`로 환경을 만든 뒤, `uv run <스크립트>`로 실행하는
식입니다. `uv run`이 그 프로젝트의 가상환경에서 실행하므로 `activate` 과정은 필요 없습니다.

```bash
cd sm-training-job/hello-mnist
uv sync
uv run launch.py --dry-run
```

로컬 의존성은 `sagemaker`(v3)와 `boto3`뿐입니다. 학습 프레임워크는 각 예제의 컨테이너
정의(스크립트 모드는 관리형 DLC, BYOC는 `container/Dockerfile`) 안에 들어갑니다.

## 예제 목록

- **[hello-mnist](sm-training-job/hello-mnist/README.md) — 스크립트 모드**
  가장 쉬운 방식입니다. AWS가 관리하는 PyTorch 컨테이너(DLC)를 그대로 쓰고, 내 학습 코드
  (`train.py`)만 넘기면 SageMaker가 나머지를 처리합니다. Docker를 다룰 필요가 없어서,
  SageMaker 학습을 처음 본다면 여기서 시작하는 게 좋습니다.

- **[byoc-mnist](sm-training-job/byoc-mnist/README.md) — BYOC (커스텀 컨테이너)**
  Docker 이미지를 직접 만들어 ECR에 올리고 그 이미지로 학습시키는 방식(Bring Your Own
  Container)입니다. 컨테이너 규약(`/opt/ml` 경로, 실행 방식 등)을 직접 지켜야 하는 대신
  런타임을 완전히 통제할 수 있습니다. hello-mnist와 비교해 보면 스크립트 모드에서 SageMaker가
  뒤에서 무엇을 대신 해주는지 드러납니다.

## 사전 준비 (공통)

- **uv** — 파이썬 환경/패키지 관리
- **AWS CLI v2** + 자격증명과 기본 리전 (`aws configure`, `aws sts get-caller-identity`로 확인)
- **SageMaker 실행 역할** — 각 예제의 `launch.py`가 계정에서 자동 탐지하며, 없으면 만들어야
  합니다. 자세한 내용은 각 프로젝트 README를 참고하세요.
- **Docker** — BYOC 예제(byoc-mnist)에서 이미지 빌드에 필요합니다. 스크립트 모드에는 필요 없습니다.

각 예제의 구체적인 실행 방법과 설정은 해당 프로젝트의 README에 있습니다.
