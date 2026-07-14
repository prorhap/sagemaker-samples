"""SageMaker Training Job을 제출하는 스크립트 (로컬에서 실행).

핵심은 딱 두 단계입니다:
  1) ModelTrainer 로 "무엇을(train.py) / 어디서(인스턴스) / 어떤 이미지로" 돌릴지 정의
  2) trainer.train() 으로 제출 → SageMaker가 인스턴스를 띄우고 학습을 실행

설정은 아래 CONFIG 값만 바꾸면 됩니다.
  uv run launch.py            # 설정대로 제출, 완료까지 대기
  uv run launch.py --dry-run  # AWS 제출 없이 이미지 URI/설정만 확인
"""

import json
import pathlib
import sys

import boto3
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes.shapes import MetricDefinition, StoppingCondition
from sagemaker.core.training.configs import Compute, SourceCode
from sagemaker.train import ModelTrainer

HERE = pathlib.Path(__file__).parent

# ============================================================
# CONFIG — 여기 값만 바꾸면 됩니다.
# ============================================================
EPOCHS = 3
INSTANCE_TYPE = "ml.c6i.2xlarge"   # GPU로 학습하려면 예: "ml.g4dn.xlarge" (계정 쿼터 필요)
FRAMEWORK_VERSION = "2.7.1"        # PyTorch DLC 버전 (예: "2.7.1", "2.8.0")
MAX_RUN_SECONDS = 3600             # 최대 실행 시간(안전장치). 예: 86400(24h)
WAIT = True                        # False면 제출 즉시 리턴(비동기)
ROLE = None                        # SageMaker 실행 역할 ARN. None이면 자동 탐지
REGION = None                      # AWS 리전. None이면 기본 세션 리전
# ============================================================


def resolve_role(session):
    """실행 역할 ARN 결정: CONFIG의 ROLE > Studio 자동 > 계정의 SageMaker 역할 탐지."""
    if ROLE:
        return ROLE
    try:
        return get_execution_role(session)  # 노트북/Studio 안에서는 바로 됨
    except Exception:
        pass
    for r in boto3.client("iam").list_roles()["Roles"]:  # 로컬: 계정에서 하나 찾아 씀
        if "SageMaker" in r["RoleName"] and "Execution" in r["RoleName"]:
            return r["Arn"]
    raise SystemExit(
        "SageMaker 실행 역할을 찾지 못했습니다. CONFIG의 ROLE 에 ARN을 지정하세요.\n"
        "(README의 'IAM 실행 역할' 참고)"
    )


def main():
    dry_run = "--dry-run" in sys.argv[1:]

    region = REGION or boto3.session.Session().region_name
    if not region:
        raise SystemExit("리전을 찾을 수 없습니다. CONFIG의 REGION 을 지정하거나 'aws configure' 하세요.")

    # 관리형 PyTorch DLC(Deep Learning Container) 이미지 URI 계산 — torch가 이미 들어있음
    image_uri = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version=FRAMEWORK_VERSION,
        image_scope="training",
        instance_type=INSTANCE_TYPE,
    )

    print(f"region   : {region}")
    print(f"image    : {image_uri}")
    print(f"instance : {INSTANCE_TYPE}")
    print(f"epochs   : {EPOCHS}")

    if dry_run:
        print("[dry-run] 제출하지 않았습니다. 설정 확인 완료.")
        return

    session = Session(boto_session=boto3.Session(region_name=region))

    # 1) Training Job 정의
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(source_dir=str(HERE / "src"), entry_script="train.py"),
        compute=Compute(instance_type=INSTANCE_TYPE, instance_count=1),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUN_SECONDS),
        hyperparameters={"epochs": EPOCHS},  # train.py에 "--epochs 3" 으로 전달됨
        base_job_name="hello-mnist",
        role=resolve_role(session),
        sagemaker_session=session,
    ).with_metric_definitions(
        # train.py의 "test_accuracy=..." 출력을 파싱해 콘솔 메트릭 그래프로 표시
        [MetricDefinition(name="test:accuracy", regex=r"test_accuracy=([0-9\.]+)")]
    )

    # 2) 제출 (WAIT=True면 완료까지 로그를 붙어서 봄)
    print(f"\n제출 중... (wait={WAIT})")
    trainer.train(wait=WAIT, logs=WAIT)

    job = trainer._latest_training_job
    print(f"\n✅ Training Job: {job.training_job_name}")

    # 나중에 jobstatus.py 로 확인할 수 있게 Job 이름을 남겨둠
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
