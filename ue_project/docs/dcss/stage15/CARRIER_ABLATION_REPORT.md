# Carrier 因果消融报告

状态：carrier Gate **pass**；相对random机制Gate仍fail。

| ID | Q/Update | target energy | NT leakage | R_shift | valid support | NT overlap | PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| M5 | random/weighted | 0.6933 | 0.0890 | 7.7882 | 0.1886 | 0 | 32.14 |
| M6 | no-P_t/weighted | 1.0011 | 0.1236 | 8.0960 | 0.1886 | 0 | 32.10 |
| M7 | no-P_t/constrained | 0.0773 | 0.00625 | 12.3716 | 0.1886 | 0 | 48.27 |

离线GT几何审计显示Legacy target-box core中平均36.89%面积与non-target GT重叠；object-aligned有效support通过膨胀排除后overlap为0。M6相对Legacy M3 leakage降低41.5%，target energy保留147.5%，coverage=0.5595，证明Legacy carrier coupling是重要来源。但M6 leakage 0.1236仍高于对应random M5×1.10=0.0979。M7虽继续降leakage，target energy仅保留M4的19.1%。
