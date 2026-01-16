# 다른 샘플링 아키텍처 비교 예시

## 방법 1: Baseline vs Temporal Decay (기본)

```python
import experiment_utils

# 기본 사용 - Baseline vs Temporal Decay 비교
results_df = experiment_utils.run_academic_benchmark(
    model=model,
    tokenizer=tokenizer,
    thresholds=[0.03, 0.05, 0.07, 0.10],  # 테스트할 alpha_decay 값들
    samples=50
)
```

## 방법 2: 커스텀 샘플링 함수 사용

```python
from decoding import baseline_sampling, generate_with_temporal_decay, generate_with_remdm

# 다른 baseline 사용
results_df = experiment_utils.run_academic_benchmark(
    model=model,
    tokenizer=tokenizer,
    baseline_fn=baseline_sampling,  # 다른 baseline 함수
    experimental_fn=generate_with_temporal_decay,
    thresholds=[0.05, 0.10],
    samples=20
)
```

## 방법 3: 완전히 다른 두 아키텍처 비교

```python
# 예: REMDM vs Temporal Decay 비교
def remdm_wrapper(model, tokenizer, prompt, steps, gen_length, block_length):
    """REMDM을 baseline 형식에 맞게 래핑"""
    if prompt:
        prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    else:
        prompt_tokens = torch.tensor([[]], dtype=torch.long, device=model.device)
    
    result = generate_with_remdm(
        model, prompt_tokens, steps, gen_length, block_length
    )
    # inspect_sampling과 같은 형식으로 반환
    return result, []  # history는 빈 리스트

results_df = experiment_utils.run_academic_benchmark(
    model=model,
    tokenizer=tokenizer,
    baseline_fn=remdm_wrapper,  # REMDM을 baseline으로
    experimental_fn=generate_with_temporal_decay,  # Temporal Decay를 experimental로
    thresholds=[0.05, 0.10],
    samples=20
)
```

## 방법 4: 여러 아키텍처 순차 비교

```python
import pandas as pd

# 1. Baseline vs Temporal Decay
results1 = experiment_utils.run_academic_benchmark(
    model, tokenizer,
    baseline_fn=inspect_sampling,
    experimental_fn=generate_with_temporal_decay,
    thresholds=[0.05],
    samples=20
)
results1['Method'] = 'TemporalDecay'

# 2. Baseline vs REMDM
results2 = experiment_utils.run_academic_benchmark(
    model, tokenizer,
    baseline_fn=inspect_sampling,
    experimental_fn=remdm_wrapper,
    thresholds=[0.5],  # REMDM의 파라미터
    samples=20
)
results2['Method'] = 'REMDM'

# 3. 결합
all_results = pd.concat([results1, results2])

# 4. 비교 분석
print(all_results.groupby('Method')[['Acc_Exp', 'PPL_Delta']].mean())
```

## 핵심 포인트

1. **`baseline_fn`**: 기준이 되는 샘플링 함수
   - 시그니처: `(model, tokenizer, prompt, steps, gen_length, block_length) -> (result, history)`
   
2. **`experimental_fn`**: 비교할 실험 샘플링 함수
   - 시그니처: `(model, prompt_tokens, steps, gen_length, block_length, alpha_decay, remask_budget) -> (result, logs)`
   
3. **`thresholds`**: 실험 함수의 파라미터 값들 (보통 alpha_decay)

4. **래퍼 함수**: 다른 시그니처를 가진 함수는 래퍼로 감싸서 사용
