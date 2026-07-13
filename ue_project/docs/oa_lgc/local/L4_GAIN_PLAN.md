# L4 Learning Gain 计划

实现 `G_t^c/G_t^p`、target ratio/protect、未见 poison query carrier loss，以及逐类 `G_k^c/G_k^p` authorized gap。有效性判断先于除法；invalid target/class 的 ratio 为 null，不通过 clamp 伪装成有效。

synthetic 正例要求 `G_t^c > G_t^p` 且 authorized gap≈0；反例覆盖 target clean gain≈0、缺失 class、负 gain 和过小 denominator。所有有效 tensor 必须支持反向传播且有限。

