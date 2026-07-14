#!/usr/bin/env bash
# 커스텀 이미지를 빌드해서 Amazon ECR에 올리는 스크립트.
#
# GPU(NVIDIA)로 돌려도 --platform 은 그대로 linux/amd64 입니다.
#   --platform 은 'CPU 아키텍처'(amd64 vs arm64)를 뜻하고, GPU 유무와는 무관합니다.
#   SageMaker의 GPU 인스턴스(ml.g4dn, ml.g5, ml.p3/p4 등)도 CPU는 전부 x86_64(amd64)라서
#   linux/amd64 로 빌드하는 게 맞습니다. (예외: Graviton+GPU 같은 arm64 GPU 인스턴스라면
#   linux/arm64 지만, SageMaker 학습에선 드뭅니다.)
#   GPU에서 바뀌는 것은 --platform 이 아니라 '이미지 내용'입니다:
#     1) CUDA 지원 베이스 이미지 사용 (예: nvidia/cuda:*-runtime 또는 AWS PyTorch GPU DLC)
#     2) CPU 빌드 torch 대신 CUDA 빌드 torch 설치 (Dockerfile의 --index-url 을 cu121 등으로)
#     3) nvidia-docker 호환 (NVIDIA 드라이버는 호스트가 제공하므로 이미지엔 CUDA 툴킷만)
#
# 사용법:
#   ./build_and_push.sh              # 리포명 기본값(byoc-mnist), 리전 자동감지
#   ./build_and_push.sh my-repo      # 리포명 지정
set -euo pipefail

REPO_NAME="${1:-byoc-mnist}"
TAG="latest"

# 계정/리전 자동 감지 (실패하면 친절히 안내)
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGION="$(aws configure get region || true)"
if [[ -z "${REGION}" ]]; then
  echo "리전을 찾을 수 없습니다. 'aws configure' 로 기본 리전을 설정하거나 AWS_REGION 을 지정하세요." >&2
  exit 1
fi

IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"
echo "대상 이미지: ${IMAGE_URI}"

# Docker 데몬 확인
if ! docker info >/dev/null 2>&1; then
  echo "Docker 데몬이 실행 중이 아닙니다. Docker Desktop을 먼저 켜주세요." >&2
  exit 1
fi

# 1) ECR 로그인
echo "==> ECR 로그인"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# 2) 리포지토리 생성 (이미 있으면 무시)
echo "==> ECR 리포지토리 준비"
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}" >/dev/null

# 3) 빌드 (반드시 linux/amd64!) → 태그 → 푸시
echo "==> 빌드 (linux/amd64)"
docker build --platform linux/amd64 -t "${REPO_NAME}:${TAG}" container/

echo "==> 태그 & 푸시"
docker tag "${REPO_NAME}:${TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "✅ 완료: ${IMAGE_URI}"
echo "이제 launch.py 의 IMAGE_URI 를 위 값으로 두고 (기본은 자동 조합) 실행하세요:"
echo "  uv run launch.py --dry-run"
