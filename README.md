# BugSweeper

AST 기반 코드 그래프 분류(특히 스마트컨트랙트 취약점 분류)를 위한 연구 코드입니다.
핵심 아이디어는 **deterministic hierarchical pooling**으로, 단순 global pooling 대신 AST 구조를 이용해 계층적으로 그래프를 coarsening 합니다.

## What this repo includes

- `preprocess.py`: Solidity 코드 → AST(JSON) → 그래프(PyG Data) 전처리
- `pool_model.py`: 계층 풀링을 지원하는 GNN 인코더/분류기
- `train.py`: 학습/검증/테스트
- `scripts/run_ablation_matrix.sh`: seed × pooling 설정 ablation 실행
- `scripts/summarize_metrics.py`: 실험 CSV 결과 집계
- `docs/benchmark_matrix_template.md`: Devign/Big-Vul/Smart-contract 벤치마크 템플릿
- `docs/table_templates.md`: 논문용 결과표 템플릿

---

## 1) 빠른 실행 가이드

### 1-1. 환경 준비

```bash
python -m pip install -r requirements.txt
# 필요 시(환경에 따라)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch_geometric
```

### 1-2. Solidity 컴파일러 준비

본 프로젝트 전처리는 `solc`가 필요합니다.

```bash
# solc-select 사용 예시
pip install solc-select
solc-select install 0.8.20
solc-select use 0.8.20
solc --version
```

### 1-3. 전처리

```bash
python preprocess.py -m train -l function -c 4
python preprocess.py -m valid -l function -c 4
python preprocess.py -m test  -l function -c 4
```

### 1-4. 기본 학습

```bash
python train.py \
  -S train -M POOL -L function \
  --coverage 4 \
  --epochs 50 \
  --batch_size 64 \
  --gpu 0
```

### 1-5. 계층 풀링 ablation 실험

```bash
bash scripts/run_ablation_matrix.sh smart-contract 4 0
```

- 기본 조합:
  - `single,0` (기본 단일 풀링)
  - `hier,2`
  - `hier,4`
  - `hier,0` (모든 계층)
- 기본 seed: `42, 7, 13`

### 1-6. 결과 집계

```bash
python scripts/summarize_metrics.py models/metrics_4_ce.csv
```

---

## 2) 주요 실험 인자

`train.py`에서 사용:

- `--seed`: 실험 시드
- `--pool_mode {single,hier}`
- `--max_pool_levels N` (0이면 가능한 모든 계층 사용)
- `--coverage`: 전처리 coverage와 반드시 일치

---

## 3) Private GitHub repo 생성 (자동화 스크립트)

현재 컨테이너에서 GitHub CLI(`gh`) 및 토큰이 기본 제공되지 않을 수 있어,
`curl + GitHub API` 방식 스크립트를 제공합니다.

```bash
bash scripts/create_private_repo.sh <github_token> <owner_or_org> <new_repo_name>
```

예시:

```bash
bash scripts/create_private_repo.sh $GITHUB_TOKEN my-org BugSweeper-private
```

성공 시:
- private repo 생성
- 현재 코드를 해당 remote에 push 할 수 있는 URL 출력

> 토큰 권한: `repo` (private repository 생성/푸시 가능)

---

## 4) 논문 작성용 템플릿

- 벤치마크/비교 매트릭스: `docs/benchmark_matrix_template.md`
- 결과표(Table 1/2/3) 템플릿: `docs/table_templates.md`

---

## 5) 참고

- 데이터/스플릿/토크나이저 설정은 `config.py`, `models/tokenizer/`를 확인하세요.
- 전처리 산출물(`preprocessed/...`)이 없으면 학습이 시작되지 않습니다.
