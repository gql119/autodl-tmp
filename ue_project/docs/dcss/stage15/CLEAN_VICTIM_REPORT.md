# 收敛 clean victim 报告

状态：Gate **fail**。100 epoch、640 px、batch 16、SGD；固定scratch初始化hash `38f99181a02b6ca51371a2408ce432d16f762b6ca0fe40bdb0277b3dd03af403`。

best target/non-target/all mAP50为0.5069/0.2485/0.2614。最近20 epoch稳定、训练正常、重复评估一致、初始化独立，但non-target未达到0.70。50 epoch仍明显上升，故已继续至100；100 epoch平台化，无需150。后续victim decision被禁止。
