# Clean victim 失败记录

100 epoch稳定后non-target mAP50仅0.2485。已排除NaN/Inf、评估不可重复、train/val overlap和未收敛；由50延长到100仍未达到Gate。分类：baseline underfitting（800图规模限制），不影响历史结果，也不是DCSS method failure。
