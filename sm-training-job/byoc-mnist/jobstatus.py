"""제출한 Training Job의 상태와 결과를 확인한다 (SageMaker SDK v3).

사용 예:
  uv run jobstatus.py                     # .last_job.json 의 마지막 Job 확인
  uv run jobstatus.py <job-name>          # 특정 Job 확인
  uv run jobstatus.py --region us-west-2 <job-name>
"""

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("job_name", nargs="?", default=None, help="Training Job 이름 (생략 시 마지막 Job)")
    p.add_argument("--region", default=None)
    return p.parse_args()


def load_last_job():
    f = HERE / ".last_job.json"
    if not f.exists():
        raise SystemExit(
            "마지막 Job 기록(.last_job.json)이 없습니다. Job 이름을 인자로 주세요:\n"
            "  uv run jobstatus.py <job-name>"
        )
    return json.loads(f.read_text())


def main():
    args = parse_args()

    region = args.region
    job_name = args.job_name
    if job_name is None:
        rec = load_last_job()
        job_name = rec["job_name"]
        region = region or rec.get("region")

    from sagemaker.core.resources import TrainingJob

    job = TrainingJob.get(training_job_name=job_name, region=region)
    if job is None:
        raise SystemExit(f"Job을 찾을 수 없습니다: {job_name} (region={region})")

    status = job.training_job_status
    secondary = job.secondary_status or ""
    print(f"Job     : {job_name}")
    print(f"Status  : {status}  ({secondary})")

    if getattr(job, "training_time_in_seconds", None) is not None:
        print(f"Runtime : {job.training_time_in_seconds}s")

    if status == "Completed":
        if job.model_artifacts:
            print(f"Model   : {job.model_artifacts.s3_model_artifacts}")
        for m in job.final_metric_data_list or []:
            print(f"  metric {m.metric_name} = {m.value}")
    elif status == "Failed":
        print(f"Reason  : {job.failure_reason or '(없음)'}")

    # CloudWatch 로그로 바로 가는 링크
    import boto3

    log_region = region or boto3.session.Session().region_name
    print(
        "\nCloudWatch 로그:\n"
        f"  https://{log_region}.console.aws.amazon.com/cloudwatch/home"
        f"?region={log_region}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FTrainingJobs"
    )


if __name__ == "__main__":
    main()
