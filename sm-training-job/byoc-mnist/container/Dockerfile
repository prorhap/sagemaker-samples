# SageMaker BYOC (Bring Your Own Container) — 학습용 커스텀 이미지
#
# 핵심: SageMaker는 학습 시 'docker run <image> train' 으로 컨테이너를 실행합니다.
#   → 'train' 이라는 실행파일이 PATH 위에 있고 실행 권한이 있어야 합니다.
#   → ENTRYPOINT를 두지 않습니다. 그래야 SageMaker가 붙이는 'train'(학습)/'serve'(추론)
#     인자가 그대로 명령이 되어, 한 이미지로 두 모드를 모두 처리할 수 있습니다.

FROM python:3.11-slim

# 로그가 버퍼링 없이 CloudWatch로 바로 나가도록
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE

# 학습에 필요한 패키지 (torch는 CPU 빌드 — 이미지가 가벼워짐)
# GPU(NVIDIA) 인스턴스로 돌리려면: 베이스 이미지를 CUDA 지원본(nvidia/cuda 또는 AWS PyTorch GPU DLC)으로
# 바꾸고, 아래 --index-url 을 CUDA 빌드(예: .../whl/cu121)로 교체하세요.
# (빌드 시 --platform 은 GPU여도 그대로 linux/amd64 입니다 — build_and_push.sh 주석 참고)
RUN pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu

# 우리 프로그램을 /opt/program 에 두고 PATH에 추가
COPY train /opt/program/train
RUN chmod 755 /opt/program/train
ENV PATH="/opt/program:${PATH}"
WORKDIR /opt/program

# ENTRYPOINT / CMD 없음 (의도된 것):
#   docker run <image> train  →  train 실행 (학습)
#   docker run <image> serve  →  serve 실행 (추론 — README 참고, 이 예제엔 미포함)
