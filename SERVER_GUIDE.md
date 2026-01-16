# 서버 실행 가이드

## 빠른 시작

### 1. 환경 확인
```bash
cd /path/to/diffsion-processing-inpection-playground
python --version  # Python 3.12+ 확인
nvidia-smi        # GPU 확인
```

### 2. 의존성 설치 (필요시)
```bash
pip install -r requirements.txt
# 또는
uv sync
```

### 3. 빠른 테스트 (2~3분)
```bash
jupyter nbconvert --to notebook --execute quick_test.ipynb --output quick_test_result.ipynb
```

### 4. 전체 벤치마크 (30분~1시간)
```bash
jupyter nbconvert --to notebook --execute benchmark_playground.ipynb --output benchmark_result.ipynb
```

## 백그라운드 실행 (추천)

### nohup 사용
```bash
nohup jupyter nbconvert --to notebook --execute benchmark_playground.ipynb --output benchmark_result.ipynb > benchmark.log 2>&1 &
```

### tmux 사용
```bash
tmux new -s benchmark
jupyter nbconvert --to notebook --execute benchmark_playground.ipynb --output benchmark_result.ipynb
# Ctrl+B, D로 detach
# tmux attach -t benchmark로 재접속
```

## 결과 확인

### 실행 중 로그 확인
```bash
tail -f benchmark.log
```

### 완료 후 결과 확인
```bash
jupyter notebook benchmark_result.ipynb
# 또는
cat benchmark_results.csv
```

## 파라미터 조정

`benchmark_playground.ipynb`에서 다음 값들을 수정 가능:

```python
ALPHA_DECAY_VALUES = [0.03, 0.05, 0.07, 0.10]  # 테스트할 decay 값
N_SAMPLES = 50          # 샘플 수 (줄이면 빨라짐)
STEPS = 64              # 디퓨전 스텝 (줄이면 빨라짐)
GEN_LENGTH = 64         # 생성 길이
REMASK_BUDGET = 0.05    # Remasking 예산
```

## 문제 해결

### CUDA Out of Memory
```python
# benchmark_playground.ipynb에서
N_SAMPLES = 10  # 50 → 10
STEPS = 32      # 64 → 32
```

### 너무 느림
```python
ALPHA_DECAY_VALUES = [0.05, 0.10]  # 2개만 테스트
N_SAMPLES = 20
```

## 예상 소요 시간

| 설정 | 샘플 수 | 설정 수 | 예상 시간 |
|------|---------|---------|-----------|
| Quick Test | 2 | 2 | 2~3분 |
| Small | 10 | 2 | 10~15분 |
| Medium | 25 | 3 | 20~30분 |
| Full | 50 | 4 | 30~60분 |
