"""BYOC(커스텀 컨테이너) Training Job을 제출하는 스크립트 (로컬에서 실행).

hello-mnist(스크립트 모드)와의 차이는 딱 하나:
  - hello-mnist: 관리형 PyTorch DLC + source_code(train.py) 를 SDK가 컨테이너에 넣어줌
  - byoc-mnist : 내가 만든 ECR 이미지(training_image)를 그대로 실행. source_code 를 주지 '않음'.
    → SDK가 entrypoint를 덮어쓰지 않으므로 SageMaker는 'docker run <image> train' 으로 실행하고,
      컨테이너 안의 'train' 실행파일이 돌아갑니다 (= 진짜 BYOC).

먼저 이미지를 ECR에 올려야 합니다:  ./build_and_push.sh
그다음:
  uv run launch.py --dry-run   # AWS 제출 없이 이미지 URI/설정 확인
  uv run launch.py             # 제출, 완료까지 대기
"""

import gzip
import json
import pathlib
import sys
import urllib.request

import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes.shapes import MetricDefinition, StoppingCondition
from sagemaker.core.training.configs import Compute, InputData
from sagemaker.train import ModelTrainer

HERE = pathlib.Path(__file__).parent

# ============================================================
# CONFIG — 여기 값만 바꾸면 됩니다.
# ============================================================
EPOCHS = 3
INSTANCE_TYPE = "ml.c6i.2xlarge"   # BYOC 이미지가 CPU 빌드라 CPU 인스턴스 사용
MAX_RUN_SECONDS = 3600             # 최대 실행 시간(안전장치)
WAIT = True                        # False면 제출 즉시 리턴(비동기)

ECR_REPO = "byoc-mnist"            # build_and_push.sh 에서 만든 리포명
IMAGE_TAG = "latest"
IMAGE_URI = None                   # 직접 지정 가능. None이면 계정/리전 + ECR_REPO 로 자동 조합

ROLE = None                        # SageMaker 실행 역할 ARN. None이면 자동 탐지
REGION = None                      # AWS 리전. None이면 기본 세션 리전
# ============================================================


# torchvision이 MNIST(download=False)로 읽을 때 기대하는 레이아웃은 <root>/MNIST/raw/ 입니다.
# 로컬에 그 구조로 원시 파일을 준비하면, SDK가 S3로 올리고 SageMaker가 컨테이너의
# /opt/ml/input/data/training/ 에 그대로 마운트해 줍니다.
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def prepare_mnist_data():
    """MNIST를 로컬 data/MNIST/raw/ 에 (압축 해제까지) 준비하고 그 경로를 반환.

    이미 있으면 건너뜁니다. torchvision 없이 stdlib만으로 처리 → 로컬 환경이 가볍게 유지됩니다.
    """
    raw = HERE / "data" / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for fname in MNIST_FILES:
        gz_path = raw / fname
        out_path = raw / fname[:-3]  # .gz 제거한 실제 파일 (torchvision이 이걸 읽음)
        if out_path.exists():
            continue
        print(f"  다운로드: {fname}")
        urllib.request.urlretrieve(f"{MNIST_MIRROR}/{fname}", gz_path)
        with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
            fout.write(fin.read())
        gz_path.unlink()  # 압축본은 지움 (torchvision은 압축 해제본만 필요)
    return str(HERE / "data")


def resolve_role(session):
    """실행 역할 ARN: CONFIG의 ROLE > Studio 자동 > 계정의 SageMaker 역할 탐지."""
    if ROLE:
        return ROLE
    try:
        return get_execution_role(session)
    except Exception:
        pass
    for r in boto3.client("iam").list_roles()["Roles"]:
        if "SageMaker" in r["RoleName"] and "Execution" in r["RoleName"]:
            return r["Arn"]
    raise SystemExit(
        "SageMaker 실행 역할을 찾지 못했습니다. CONFIG의 ROLE 에 ARN을 지정하세요."
    )


def main():
    dry_run = "--dry-run" in sys.argv[1:]

    region = REGION or boto3.session.Session().region_name
    if not region:
        raise SystemExit("리전을 찾을 수 없습니다. CONFIG의 REGION 을 지정하거나 'aws configure' 하세요.")

    account = boto3.client("sts").get_caller_identity()["Account"]
    image_uri = IMAGE_URI or f"{account}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG}"

    print(f"region   : {region}")
    print(f"image    : {image_uri}   (내가 만든 ECR 이미지)")
    print(f"instance : {INSTANCE_TYPE}")
    print(f"epochs   : {EPOCHS}")

    if dry_run:
        print("[dry-run] 제출하지 않았습니다. 설정 확인 완료.")
        print("이미지가 ECR에 없다면 먼저: ./build_and_push.sh")
        return

    session = Session(boto_session=boto3.Session(region_name=region))

    # 데이터 준비: 로컬에 MNIST를 만들고, 'training' 채널로 넘깁니다.
    # 로컬 경로를 주면 SDK가 알아서 S3에 업로드한 뒤 채널로 연결해 줍니다.
    print("\nMNIST 데이터 준비 중...")
    data_dir = prepare_mnist_data()
    training_channel = InputData(channel_name="training", data_source=data_dir)

    # 핵심: training_image 는 내 이미지, source_code 는 주지 않음 → 진짜 BYOC
    trainer = ModelTrainer(
        training_image=image_uri,
        compute=Compute(instance_type=INSTANCE_TYPE, instance_count=1),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUN_SECONDS),
        hyperparameters={"epochs": EPOCHS},  # /opt/ml/input/config/hyperparameters.json 으로 전달됨
        base_job_name="byoc-mnist",
        role=resolve_role(session),
        sagemaker_session=session,
    ).with_metric_definitions(
        [MetricDefinition(name="test:accuracy", regex=r"test_accuracy=([0-9\.]+)")]
    )

    # 'training' 채널 주입 → 컨테이너의 /opt/ml/input/data/training/ 에 마운트됨
    print(f"제출 중... (wait={WAIT})")
    trainer.train(input_data_config=[training_channel], wait=WAIT, logs=WAIT)

    job = trainer._latest_training_job
    print(f"\n✅ Training Job: {job.training_job_name}")

    (HERE / ".last_job.json").write_text(
        json.dumps({"job_name": job.training_job_name, "region": region})
    )

    if WAIT:
        job.refresh()
        if job.model_artifacts:
            print(f"모델: {job.model_artifacts.s3_model_artifacts}")
    else:
        print("상태 확인: uv run jobstatus.py")


if __name__ == "__main__":
    main()
