# LLaDA Diffusion Inspection Project

이 프로젝트는 **LLaDA (Large Language Diffusion Architecture)** 모델의 텍스트 생성 과정(Diffusion Process)을 심층적으로 분석하고 실험하기 위한 독립형 환경입니다.

## 기능
*   **단계별 시각화 (Step-by-Step Visualization)**: 모델이 노이즈 상태에서 텍스트를 생성해 나가는 과정을 투명하게 시각화합니다.
*   **신뢰도 분석 (Confidence Analysis)**: 생성 과정에서 각 토큰에 대한 모델의 확신도가 어떻게 변화하는지 히트맵으로 확인 가능합니다.
*   **커스텀 알고리즘 실험**: `decoding.py`의 기본 로직 외에 자신만의 샘플링/디코딩 알고리즘을 구현하고 테스트할 수 있습니다.

## 설치 방법

1.  Python 3.8 이상 환경 준비 (Conda 권장)
2.  패키지 설치:
    ```bash
    pip install -r requirements.txt
    ```
    *(참고: `requirements.txt`는 `torch`, `transformers` 등을 포함해야 합니다.)*

## 실행 방법

1.  VS Code 또는 Jupyter Lab에서 이 폴더(`diff_processing_inspection_llada8b`)를 엽니다.
2.  `inspection_playground.ipynb` 파일을 실행합니다.
3.  **모델 로드**:
    *   기본적으로 HuggingFace Hub의 `GSAI-ML/LLaDA-8B-Base` 모델을 다운로드합니다.
    *   이미 다운로드 받은 모델이 있다면 노트북 내 `USE_LOCAL_PATH = True`로 설정하고 경로를 수정하세요.
4.  **실험**:
    *   `inspect_diffusion_process` 함수를 실행하여 생성 과정을 관찰합니다.
    *   함수를 수정하여 새로운 로직을 테스트합니다.

## 파일 구조

*   `inspection_playground.ipynb`: 메인 분석 노트북
*   `modeling_llada.py`: LLaDA 모델 아키텍처 정의
*   `configuration_llada.py`: 모델 설정 파일
*   `decoding.py`: 디코딩/샘플링 알고리즘 원본 소스
*   `requirements.txt`: 의존성 목록

## 참고 사항
이 프로젝트는 `Saber` 프로젝트의 파일들을 기반으로 재구성되었습니다.
