# 2\_Understanding\_Architecture

## 2단계: Attention 연산의 효율적인 HLS 구현

##  목표

Attention 메커니즘은 Transformer와 대규모 언어 모델(LLM)의 핵심 요소입니다. 입력 특징 간의 연산을 통해 문맥을 이해하고 병렬 처리를 가능하게 만들어, 강력한 표현력을 제공합니다.

이번 실습에서는 다음을 진행합니다:

1. Scaled Dot-Product Attention을 수행하는 `compute_attention` 함수 구현
2. 입력 텐서를 이진 파일에서 불러오고, 결과를 기준 출력과 비교
3. 레이턴시(지연 시간)를 최소화하면서 자원 제약 조건을 만족하도록 최적화

---

##  문제 설명

`compute_attention` 함수는 다음과 같은 연산을 수행합니다:

$$
\text{Attention}(i, j) = \text{Softmax}\left(\frac{Q \cdot K^\top}{\sqrt{d_k}}\right) \cdot V
$$

###  입력

* 입력 텐서 (`Q`, `K`, `V`)는 이진 파일 형식으로 제공됩니다:

  * `Q_tensor.bin`
  * `K_tensor.bin`
  * `V_tensor.bin`

###  출력

* 계산된 출력 텐서 (`Output_HLS`)를 생성합니다.
* 기준 출력 텐서(`Output_tensor.bin`)와 비교하여 정확도 평가
* 평균제곱오차(MSE)를 계산

### 참고 구현

기준(reference) 구현이 제공됩니다. 이는 최적화되지 않은 형태이며, 여러분의 목표는 이 구현보다 빠르게 만들면서 정확도와 구현 가능성을 유지하는 것입니다.

---

##  설계 조건 및 제약

###  필수 조건

1. **DRAM Partition 금지**

   * DRAM에 있는 배열은 partition하지 마세요.

2. **Place-and-Route 완료**

   * 합성된 디자인이 구현 가능해야 합니다 (배치 및 라우팅이 성공해야 함)

3. **고정된 차원 유지**

   * `B`, `N`, `dk`, `dv` 값은 변경하지 마세요.

4. **정확한 지연 시간 리포트**

   * **C/RTL Co-simulation** 을 사용해서 latency를 측정하세요.

>  공동 시뮬레이션은 1\~2시간 이상 소요될 수 있으니, 초기에는 작은 행렬로 실험을 추천합니다.

---

###  팁과 참고사항

1. **데이터 로딩 최적화**

   * AXI 버스트 모드를 활용하고 DRAM 대역폭을 최대로 활용하세요.
   * Ping-Pong 버퍼, streaming 등을 이용해 BRAM으로 데이터를 효율적으로 로딩

2. **병렬 처리**

   * `dataflow` 및 `stream` 구조를 활용해 데이터 로딩, 연산, 결과 쓰기를 동시에 수행하도록 구성

3. **Softmax 최적화**

   * 정확도 저하가 크지 않다면, 소프트맥스 연산을 단순화 또는 근사화하여 속도를 높이세요.

4. **Latency 분석**

   * 병목 구간을 파악하고 해당 부분을 집중적으로 최적화하세요 (예: softmax, scaling)

5. **LightningSim 활용**

   * Sharc Lab에서 제공하는 LightningSim을 활용하면 빠르게 레이턴시 예측 및 비교가 가능합니다.

6. **ap\_fixed 값 출력 시 주의사항**

   * `printf("%f", val)`처럼 출력할 경우, `val.to_float()`를 명시적으로 써야 제대로 출력됩니다.

---

