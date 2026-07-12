# 等预算机制报告

状态：Gate **fail**。M0–M3使用相同1 epoch、batch order、seed、初始carrier、support、Adam与D5权重。

| ID | Q | target energy | outside | in-space | NT leakage | R_shift |
|---|---|---:|---:|---:|---:|---:|
| M0 | random | 0.4924 | 3.5379 | 0.1244 | 0.1745 | 2.8220 |
| M1 | target-only | 0.9885 | 3.7111 | 0.1974 | 0.3961 | 2.4955 |
| M2 | original P_t | 1.1784 | 3.2203 | 0.2566 | 0.4786 | 2.4624 |
| M3 | no-P_t | 0.6786 | 2.9538 | 0.1794 | 0.2113 | 3.2111 |

M3通过target energy、R_shift、in-space、coverage和finite，但0.2113>1.10×M0=0.1919。因此不扩展到3 epoch。
