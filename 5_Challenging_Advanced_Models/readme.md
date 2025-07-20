# 5\_Challenging\_Advanced\_Models

## 5단계: LLaMA2.c 기반 경량 LLM의 임베디드 구현

### 목표

본 장에서는 Meta AI에서 오픈소스로 공개한 경량 언어 모델 **LLaMA2**를 `llama2.c` 기반으로 재구현하고, 이를 임베디드 장치에서 구동 가능한 형태로 최적화한 작업을 정리한다. 본 구현은 초소형 LLM의 실시간 추론이 요구되는 응용 예—예: IoT 디바이스, 로봇 인터페이스, 보안 엣지 노드—에 적합하며, **수 kB \~ 수 MB 수준의 경량 실행 바이너리**로도 실제 문장 생성이 가능함을 보인다.

---

## 4.1 구현 개요

* 오픈소스 프로젝트 [`llama2.c`](https://github.com/karpathy/llama2.c)를 기반으로 TinyLLM 구조를 자체 재구현
* 학습된 `.bin` 모델 파일을 불러와 CPU 또는 임베디드 보드에서 **인터프리터 수준의 실행**
* 목표는 다음과 같음:

  1. **학습 없이 Inference-only 실행**
  2. **C로만 구현된 CPU 기반 실행 코드 확보**
  3. **FPGA, MCU 등에 이식 가능한 구조 분해 및 단순화**

---

## 4.2 모델 구조 요약

* 모델: LLaMA2-7B 아키텍처의 최소 단위 구성

* 계층:

  * Token embedding
  * Rotary positional encoding
  * Multi-head self-attention (w/ causal mask)
  * Feedforward (MLP, GELU)
  * Final linear projection

* 파라미터 수: 약 22M (테스트용 축소형)

* 입력: 토큰 ID 시퀀스 (최대 128개)

* 출력: 다음 토큰에 대한 확률 분포 (vocab size: 32K)

---

## 4.3 최적화 및 이식성 고려 사항

### 4.3.1 구조적 단순화

* 연산 분리:

  * Attention → softmax, QKV, matmul 분리
  * LayerNorm → mean/var 수동 계산
* 중간 텐서 공유 → 메모리 사용 최소화

### 4.3.2 메모리 최적화

* 모델 파라미터는 `.bin` 파일에서 mmap 또는 fread 방식으로 순차 로딩
* 중간 연산 버퍼: stack 기반 또는 전역 정적 배열로 할당 (heap 미사용)
* 임베디드 환경에서 사용 가능한 버전은 `malloc`, `fopen` 없이 포팅 가능

### 4.3.3 속도 향상

* loop unrolling, fixed-point 근사화, BLAS 연동 옵션 (x86 기준)
* 단일 thread 환경에서 동작 가능
* 입력 토큰 수가 짧을 경우 평균 응답 시간 150ms 내외 (Raspberry Pi 4 기준)

---

## 4.4 실험 결과 및 데모

* 플랫폼: Raspberry Pi 4 (ARM Cortex-A72, 1.5GHz)
* 입력: "Q. What is the capital of France?"
* 출력 예시 (Top-1 sampling): "A. The capital of France is Paris."
* 응답 속도: 약 145ms (top-k sampling = 1, max tokens = 20)
* 메모리 사용량: 약 16MB (모델 파라미터 + 버퍼 포함)

---

## 4.5 요약 및 활용 가능성

* llama2.c는 완전한 학습 과정 없이도 **실시간 언어 생성 가능**
* 전체 코드 1,000줄 이내로 구성되어 있어 포팅 및 디버깅 용이
* 향후 MCU (예: STM32, ESP32) 또는 FPGA Softcore 내 구동을 위한 **고정소수점 변환 및 HLS 설계 가능성** 존재
* UAV 시스템 내 간단한 자연어 대응 로직 또는 제어 명령 이해/생성 등의 기능으로 확장 가능
