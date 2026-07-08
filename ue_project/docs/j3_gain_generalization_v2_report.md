# J3-v2 Trajectory Generalization and Gain-Scale Diagnosis

## Root Cause Audit
- held-out negative D mainly small denominator: `True`
- raw protected gain gap reversed: `True`
- train protected-valid ratio: `0.4`
- held-out protected-valid ratio: `0.43333333333333335`

## Objective-v2
- robust scales: `{'authorized': 0.0005912614822387696, 'authorized_min': 0.00036435872316360475, 'epsilon': 0.0001, 'protected': 0.0011415315628051758, 'protected_min': 0.0008287429809570314, 'quantile': 0.5, 'shared': 0.00017918477058410644, 'shared_min': 0.0001, 'source_count': 30}`
- fixed v2 held-out Delta_t/abs(Delta_a)/abs(Delta_s)/S: `0.7269086958345724` / `2.9610440891236065` / `6.581683662347496` / `-8.683663536297779`
- online v2 held-out Delta_t/abs(Delta_a)/abs(Delta_s)/S: `0.6682381709106266` / `5.666019243995349` / `6.7299760173618175` / `-11.585576080779235`

## Online Resampling
- online accepted valid ratio: `0.7333333333333333`
- trajectory repetition rate: `0.22727272727272727`
- online best held-out step/score: `{'best_score': -6.811017654702785e-05, 'best_step': 0}`

## Engineering Checks
- finite difference: `{'fixed': {'direction': 'raw', 'gradient_norm': 22.57768440246582, 'loss_after': 0.7414948344230652, 'loss_before': 0.7453290820121765, 'ok': True, 'step_size': 3e-06}, 'online': {'direction': 'raw', 'gradient_norm': 19.240352630615234, 'loss_after': 0.13073161244392395, 'loss_before': 0.7441774010658264, 'ok': True, 'step_size': 0.01}}`
- parameter leak max: `0.0`
- memory allocated first/last: `[149097472, 220105728]`

## Gradient Conflict
- mean protected-authorized cosine: `0.007863177273135919`
- mean protected-shared cosine: `-0.022239730335198916`
- mean authorized-shared cosine: `0.03666551880395183`
- mean gradient norms: `{'authorized': 4.092907934234693, 'protected': 2.058480274404561, 'shared': 7.5874990694797955}`

## Decision
- `C. online resampling helps protected gain, but authorized/shared gradient conflict remains severe.`