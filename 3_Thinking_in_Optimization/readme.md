# 3\_Thinking\_in\_Optimization

## 3단계: Sparse Matrix Multiplication의 효율적인 HLS 구현

### 목표

희소 행렬 곱셈은 과학 계산, 머신 러닝, 그래프 기반 알고리즘에서 필수적인 연산입니다. FPGA에서 이를 최적화하려면 불규칙한 메모리 접근, 희소 데이터 구조 처리, 하드웨어 자원 제약을 고려해야 합니다.

이번 실습에서는 다음을 수행합니다:

* CSR 형식의 희소 행렬 A와 CSC 형식(전치) 희소 행렬 B를 곱하여 밀집 행렬 C를 계산
* A, B를 이진 파일에서 불러오고, 결과 C를 이진 파일로 저장
* 레이턴시(지연 시간)를 최소화하면서 FPGA 자원 제약을 만족하도록 최적화

---

### 문제 설명

희소 행렬 곱셈:

$C[i][j] = \sum_k A[i][k] \cdot B[k][j]$

#### 입력:

* CSR 형식의 희소 행렬 A:

  * `values_A[]`, `column_indices_A[]`, `row_ptr_A[]`
* CSC 형식(전치된) 희소 행렬 B:

  * `values_B[]`, `row_indices_B[]`, `col_ptr_B[]`
* 이진 파일:

  * `A_matrix_csr.bin`
  * `B_matrix_csc.bin`
* 다양한 희소도 (0.1, 0.5, 0.8)의 테스트 벤치 제공

#### 출력:

* 밀집 행렬 C를 `C_matrix_result.bin`에 저장
* 기준 출력과 비교하여 정확도(MSE) 평가

#### 참고 구현:

* 기준(reference) 구현이 제공되며, 이는 최적화되지 않은 버전입니다. 목표는 정확도와 구현 가능성을 유지하면서 속도를 높이는 것입니다.

---

### 설계 조건 및 제약

#### 필수 조건

* **DRAM Partition 금지**

  * DRAM 배열은 partition 하지 마세요.
* **Place-and-Route 성공**

  * 합성된 디자인은 구현 가능해야 하며, bitstream 생성을 완료해야 합니다.
* **정확도 확보**

  * 출력 C는 기준 구현과 MSE 오차 이내에서 일치해야 합니다.
* **정확한 지연 시간 리포트**

  * C/RTL Co-simulation 또는 LightningSim으로 latency를 측정하세요.

---

### 팁과 참고사항

* **단순 구현부터 시작**: 작은 행렬(N=8 등)로 먼저 검증
* **점진적 최적화**: pipeline, unroll, dataflow 등을 적용
* **메모리 접근 최적화**: AXI 버스트 모드, ping-pong 버퍼 등 활용
* **Softmax 등 고비용 연산**은 근사화 고려
* **LightningSim 사용**: 빠른 레이턴시 평가 가능
* **ap\_fixed 출력 시 주의**: `val.to_float()` 명시 필요


---

행운을 빕니다!
