# Virtual LR Sweep Report

Chosen LR: `1e-05`

Conclusion: all tested virtual learning rates have negative mean `meta_selectivity`; the chosen LR is the least damaging fallback, not evidence that the virtual update direction is already effective.

## LR 1e-05
- valid batches: `30`
- empty protected/authorized: `1` / `0`
- protected_learning_gap mean: `-1.6609827677408854e-06`
- authorized_learning_gap mean: `1.4050801595052083e-05`
- meta_selectivity mean: `-1.571178436279297e-05`
- clean_query_delta mean: `-6.642341613769532e-05`
- virtual_update_norm mean: `0.00014336027306853792`
- meta_gradient_norm mean: `0.0005856379803541738`

## LR 3e-05
- valid batches: `30`
- empty protected/authorized: `1` / `0`
- protected_learning_gap mean: `-6.079673767089844e-06`
- authorized_learning_gap mean: `4.19775644938151e-05`
- meta_selectivity mean: `-4.805723826090495e-05`
- clean_query_delta mean: `-0.00019749005635579428`
- virtual_update_norm mean: `0.00043008081168712425`
- meta_gradient_norm mean: `0.0017568330959572146`

## LR 0.0001
- valid batches: `30`
- empty protected/authorized: `1` / `0`
- protected_learning_gap mean: `-2.0333131154378255e-05`
- authorized_learning_gap mean: `0.00013992786407470703`
- meta_selectivity mean: `-0.00016026099522908527`
- clean_query_delta mean: `-0.0006552060445149739`
- virtual_update_norm mean: `0.0014336027554236353`
- meta_gradient_norm mean: `0.005854885932058096`

## LR 0.0003
- valid batches: `30`
- empty protected/authorized: `1` / `0`
- protected_learning_gap mean: `-6.376902262369791e-05`
- authorized_learning_gap mean: `0.000407870610555013`
- meta_selectivity mean: `-0.00047163963317871095`
- clean_query_delta mean: `-0.0020613988240559896`
- virtual_update_norm mean: `0.0043008086814855535`
- meta_gradient_norm mean: `0.017544110802312694`

## LR 0.001
- valid batches: `30`
- empty protected/authorized: `1` / `0`
- protected_learning_gap mean: `-0.00043203035990397134`
- authorized_learning_gap mean: `0.0013739109039306641`
- meta_selectivity mean: `-0.0018059412638346355`
- clean_query_delta mean: `-0.006187661488850912`
- virtual_update_norm mean: `0.014336028601974249`
- meta_gradient_norm mean: `0.05810240305339297`

## LR 0.003
- valid batches: `30`
- empty protected/authorized: `1` / `0`
- protected_learning_gap mean: `-0.001160407066345215`
- authorized_learning_gap mean: `0.0031547307968139648`
- meta_selectivity mean: `-0.00431513786315918`
- clean_query_delta mean: `-0.010196900367736817`
- virtual_update_norm mean: `0.04300808347761631`
- meta_gradient_norm mean: `0.17676770612597464`
