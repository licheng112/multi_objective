# 优化收敛对比实验报告

## 实验设置

- 代理模型：KAN，输入 22 个关键工艺参数，输出 IV、DEG、CTA。
- 对比方法：Knowledge-constrained NSGA-II、NSGA-II、NSGA-III、MOEA/D。
- 迭代范围：0-500 代；独立重复次数：30；随机种子范围：42-71。
- 目标函数：F1=|Y_IV-50|，F2=|Y_DEG-1.37|，F3=|Y_CTA-51|。
- 正文收敛图直接使用逐代当前代非支配前沿均值及当前代 HV/IGD；曲线为 mean，阴影为 95% CI；不进行累计最优、事件抽点、阶梯化或后处理扰动。
- MOEA/D 参数由 pilot 参数敏感性实验选出，最终配置为 `stable_decomposition`；选择规则：Lowest mean rank across final Average objective, HV, and IGD on pilot seeds.
- 本文方法在 0-80 代保留较强全局探索，80-160 代逐步增强知识引导，160 代后进入稳定开发阶段。

## 关键代数对比（当前代口径）

- 第 0 代 Average objective：本文方法 0.3223，NSGA-II 0.4984，NSGA-III 0.4984，MOEA/D 0.4984。
- 第 100 代 Average objective：本文方法 0.1579，NSGA-II 0.2223，NSGA-III 0.2219，MOEA/D 0.1551。
- 第 500 代 Average objective：本文方法 0.1573，NSGA-II 0.1436，NSGA-III 0.1401，MOEA/D 0.1355。
- 第 0 代 HV：本文方法 0.5602，NSGA-II 0.2545，NSGA-III 0.2545，MOEA/D 0.2545。
- 第 500 代 HV：本文方法 0.8387，NSGA-II 0.7762，NSGA-III 0.7711，MOEA/D 0.7275。
- 第 100 代 IGD：本文方法 0.0873，NSGA-II 0.1478，NSGA-III 0.1467，MOEA/D 0.1912。
- 第 500 代 IGD：本文方法 0.0673，NSGA-II 0.0971，NSGA-III 0.0968，MOEA/D 0.1390。

## 最终指标

| method                        |   front_points_mean |   front_points_std |   Final_F1_IV_mean |   Final_F1_IV_std |   Final_F2_DEG_mean |   Final_F2_DEG_std |   Final_F3_CTA_mean |   Final_F3_CTA_std |   Average_objective_mean |   Average_objective_std |   Raw_average_objective_mean |   Raw_average_objective_std |   HV_mean |   HV_std |   IGD_mean |   IGD_std |   KCSR_mean |   KCSR_std |   PAM_mean |   PAM_std |   Convergence_Generation_mean |   Convergence_Generation_std |
|:------------------------------|--------------------:|-------------------:|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|-------------------------:|------------------------:|-----------------------------:|----------------------------:|----------:|---------:|-----------:|----------:|------------:|-----------:|-----------:|----------:|------------------------------:|-----------------------------:|
| Knowledge-constrained NSGA-II |             136.3   |             9.2518 |             0.0718 |            0.0211 |              0.0094 |             0.0034 |              0.2777 |             0.0599 |                   0.1196 |                  0.02   |                       0.1196 |                      0.02   |    0.8387 |   0.0195 |     0.0673 |    0.0146 |           1 |          0 |     0.0676 |    0.0099 |                       93.0667 |                      68.0045 |
| NSGA-II                       |             135     |             0      |             0.1099 |            0.0145 |              0.0119 |             0.0013 |              0.2599 |             0.0425 |                   0.1272 |                  0.0163 |                       0.1272 |                      0.0163 |    0.7762 |   0.0216 |     0.0971 |    0.009  |           1 |          0 |     0.1021 |    0.0103 |                      189.1    |                      57.6364 |
| NSGA-III                      |             134.667 |             1.373  |             0.105  |            0.0161 |              0.012  |             0.0016 |              0.2706 |             0.0389 |                   0.1292 |                  0.0156 |                       0.1292 |                      0.0156 |    0.7711 |   0.0222 |     0.0968 |    0.0095 |           1 |          0 |     0.1046 |    0.0106 |                      177.267  |                      65.7723 |
| MOEA/D                        |             138.667 |             1.8815 |             0.1368 |            0.0228 |              0.0253 |             0.0081 |              0.2641 |             0.0615 |                   0.1421 |                  0.0273 |                       0.1421 |                      0.0273 |    0.7275 |   0.0498 |     0.139  |    0.0564 |           1 |          0 |     0.1301 |    0.0228 |                       87.1333 |                      98.2301 |

## 图间一致性检查

| metric            | direction   | best_method                                              |   ours_value |   best_value | supports_main_claim   | note                                                |
|:------------------|:------------|:---------------------------------------------------------|-------------:|-------------:|:----------------------|:----------------------------------------------------|
| F1-IV             | min         | Knowledge-constrained NSGA-II                            |       0.0718 |       0.0718 | True                  | consistent                                          |
| F2-DEG            | min         | Knowledge-constrained NSGA-II                            |       0.0094 |       0.0094 | True                  | consistent                                          |
| F3-CTA            | min         | NSGA-II                                                  |       0.2777 |       0.2599 | False                 | local single-metric advantage; discuss as trade-off |
| Average objective | min         | Knowledge-constrained NSGA-II                            |       0.1196 |       0.1196 | True                  | consistent                                          |
| HV                | max         | Knowledge-constrained NSGA-II                            |       0.8387 |       0.8387 | True                  | consistent                                          |
| IGD               | min         | Knowledge-constrained NSGA-II                            |       0.0673 |       0.0673 | True                  | consistent                                          |
| KCSR              | max         | Knowledge-constrained NSGA-II; NSGA-II; NSGA-III; MOEA/D |       1      |       1      | False                 | tie; report as feasibility evidence only            |
| PAM               | min         | Knowledge-constrained NSGA-II                            |       0.0676 |       0.0676 | True                  | consistent                                          |

## 统计检验与平均排名

Wilcoxon 成对检验采用相同 seed 对齐，p 值经 Holm 校正；若校正后不显著，则只作为趋势性差异讨论。

| metric            | direction   | comparison                                |   runs |   ours_mean |   baseline_mean |   ours_better_rate |   paired_median_advantage |   cliffs_delta_raw |   p_value |   holm_p_value | significant_0_05   |
|:------------------|:------------|:------------------------------------------|-------:|------------:|----------------:|-------------------:|--------------------------:|-------------------:|----------:|---------------:|:-------------------|
| Final_F1_IV       | min         | Knowledge-constrained NSGA-II vs NSGA-II  |     30 |      0.0718 |          0.1099 |             0.9000 |                    0.0406 |            -0.8022 |    0.0000 |         0.0000 | True               |
| Final_F1_IV       | min         | Knowledge-constrained NSGA-II vs NSGA-III |     30 |      0.0718 |          0.1050 |             0.8667 |                    0.0389 |            -0.7933 |    0.0000 |         0.0000 | True               |
| Final_F1_IV       | min         | Knowledge-constrained NSGA-II vs MOEA/D   |     30 |      0.0718 |          0.1368 |             0.9667 |                    0.0695 |            -0.9333 |    0.0000 |         0.0000 | True               |
| Final_F2_DEG      | min         | Knowledge-constrained NSGA-II vs NSGA-II  |     30 |      0.0094 |          0.0119 |             0.8667 |                    0.0027 |            -0.7356 |    0.0000 |         0.0000 | True               |
| Final_F2_DEG      | min         | Knowledge-constrained NSGA-II vs NSGA-III |     30 |      0.0094 |          0.0120 |             0.8667 |                    0.0032 |            -0.7222 |    0.0000 |         0.0000 | True               |
| Final_F2_DEG      | min         | Knowledge-constrained NSGA-II vs MOEA/D   |     30 |      0.0094 |          0.0253 |             0.9667 |                    0.0137 |            -0.9556 |    0.0000 |         0.0000 | True               |
| Final_F3_CTA      | min         | Knowledge-constrained NSGA-II vs NSGA-II  |     30 |      0.2777 |          0.2599 |             0.4667 |                   -0.0052 |             0.2200 |    0.2367 |         0.7100 | False              |
| Final_F3_CTA      | min         | Knowledge-constrained NSGA-II vs NSGA-III |     30 |      0.2777 |          0.2706 |             0.4333 |                   -0.0146 |             0.0667 |    0.5028 |         0.7100 | False              |
| Final_F3_CTA      | min         | Knowledge-constrained NSGA-II vs MOEA/D   |     30 |      0.2777 |          0.2641 |             0.4333 |                   -0.0110 |             0.1400 |    0.3492 |         0.7100 | False              |
| Average_objective | min         | Knowledge-constrained NSGA-II vs NSGA-II  |     30 |      0.1196 |          0.1272 |             0.6667 |                    0.0121 |            -0.3378 |    0.1048 |         0.1048 | False              |
| Average_objective | min         | Knowledge-constrained NSGA-II vs NSGA-III |     30 |      0.1196 |          0.1292 |             0.7000 |                    0.0115 |            -0.3911 |    0.0208 |         0.0417 | True               |
| Average_objective | min         | Knowledge-constrained NSGA-II vs MOEA/D   |     30 |      0.1196 |          0.1421 |             0.6667 |                    0.0240 |            -0.5289 |    0.0010 |         0.0029 | True               |
| HV                | max         | Knowledge-constrained NSGA-II vs NSGA-II  |     30 |      0.8387 |          0.7762 |             0.9667 |                   -0.0692 |             0.9633 |    0.0000 |         0.0000 | True               |
| HV                | max         | Knowledge-constrained NSGA-II vs NSGA-III |     30 |      0.8387 |          0.7711 |             1.0000 |                   -0.0700 |             0.9778 |    0.0000 |         0.0000 | True               |
| HV                | max         | Knowledge-constrained NSGA-II vs MOEA/D   |     30 |      0.8387 |          0.7275 |             0.9667 |                   -0.1000 |             0.9889 |    0.0000 |         0.0000 | True               |
| IGD               | min         | Knowledge-constrained NSGA-II vs NSGA-II  |     30 |      0.0673 |          0.0971 |             0.9333 |                    0.0332 |            -0.9022 |    0.0000 |         0.0000 | True               |
| IGD               | min         | Knowledge-constrained NSGA-II vs NSGA-III |     30 |      0.0673 |          0.0968 |             0.9667 |                    0.0318 |            -0.9000 |    0.0000 |         0.0000 | True               |
| IGD               | min         | Knowledge-constrained NSGA-II vs MOEA/D   |     30 |      0.0673 |          0.1390 |             0.9667 |                    0.0608 |            -0.9711 |    0.0000 |         0.0000 | True               |
| PAM               | min         | Knowledge-constrained NSGA-II vs NSGA-II  |     30 |      0.0676 |          0.1021 |             1.0000 |                    0.0324 |            -1.0000 |    0.0000 |         0.0000 | True               |
| PAM               | min         | Knowledge-constrained NSGA-II vs NSGA-III |     30 |      0.0676 |          0.1046 |             1.0000 |                    0.0338 |            -1.0000 |    0.0000 |         0.0000 | True               |
| PAM               | min         | Knowledge-constrained NSGA-II vs MOEA/D   |     30 |      0.0676 |          0.1301 |             1.0000 |                    0.0598 |            -1.0000 |    0.0000 |         0.0000 | True               |

| metric            | direction   | method                        |   average_rank |   rank_std |
|:------------------|:------------|:------------------------------|---------------:|-----------:|
| Final_F1_IV       | min         | Knowledge-constrained NSGA-II |         1.2667 |     0.6915 |
| Final_F1_IV       | min         | NSGA-II                       |         2.5667 |     0.7279 |
| Final_F1_IV       | min         | NSGA-III                      |         2.4667 |     0.7761 |
| Final_F1_IV       | min         | MOEA/D                        |         3.7000 |     0.7022 |
| Final_F2_DEG      | min         | Knowledge-constrained NSGA-II |         1.3000 |     0.7497 |
| Final_F2_DEG      | min         | NSGA-II                       |         2.3333 |     0.6065 |
| Final_F2_DEG      | min         | NSGA-III                      |         2.4000 |     0.6747 |
| Final_F2_DEG      | min         | MOEA/D                        |         3.9667 |     0.1826 |
| Final_F3_CTA      | min         | Knowledge-constrained NSGA-II |         2.6667 |     1.1244 |
| Final_F3_CTA      | min         | NSGA-II                       |         2.3667 |     1.0981 |
| Final_F3_CTA      | min         | NSGA-III                      |         2.5333 |     1.1666 |
| Final_F3_CTA      | min         | MOEA/D                        |         2.4333 |     1.1351 |
| Average_objective | min         | Knowledge-constrained NSGA-II |         1.9667 |     1.0981 |
| Average_objective | min         | NSGA-II                       |         2.4000 |     0.9685 |
| Average_objective | min         | NSGA-III                      |         2.6333 |     1.0981 |
| Average_objective | min         | MOEA/D                        |         3.0000 |     1.1142 |
| HV                | max         | Knowledge-constrained NSGA-II |         1.0500 |     0.2013 |
| HV                | max         | NSGA-II                       |         2.6000 |     0.7120 |
| HV                | max         | NSGA-III                      |         2.7500 |     0.5981 |
| HV                | max         | MOEA/D                        |         3.6000 |     0.8137 |
| IGD               | min         | Knowledge-constrained NSGA-II |         1.1333 |     0.5074 |
| IGD               | min         | NSGA-II                       |         2.7667 |     0.7279 |
| IGD               | min         | NSGA-III                      |         2.5333 |     0.7303 |
| IGD               | min         | MOEA/D                        |         3.5667 |     0.8172 |
| PAM               | min         | Knowledge-constrained NSGA-II |         1.0000 |     0.0000 |
| PAM               | min         | NSGA-II                       |         2.5667 |     0.6261 |
| PAM               | min         | NSGA-III                      |         2.7000 |     0.6513 |
| PAM               | min         | MOEA/D                        |         3.7333 |     0.6397 |

## 结果分析

正文曲线展示当前代搜索状态，因此早期可能出现局部震荡或短暂退化；这是早期探索阶段的自然结果，而不是后处理平滑或人为扰动。

本文方法的结论应表述为整体稳健优势，而不是所有单目标全面最优。本轮 30 次独立重复中，本文方法在 HV、IGD、PAM 以及 IV/DEG 偏差上具有更稳定优势；但 F3-CTA 的最优均值来自 NSGA-II，MOEA/D 在 CTA 上也保持局部竞争力，并且平均收敛代数较早。这些现象应作为多目标权衡保留并讨论。

由于 KCSR 在四种方法上均为 1.0，该指标只能说明最终前沿均满足先验约束，不能作为区分本文方法优势的核心证据。最终判断应以集合质量、参数调整幅度、统计检验和单目标 trade-off 共同支撑。

## Baseline start consistency

NSGA-II, NSGA-III, and MOEA/D share the same global initial population; their generation-0 spread should stay within one estimated y-axis tick.

| metric                  |   baseline_start_min |   baseline_start_max |   baseline_start_spread |   estimated_one_tick | within_one_tick   |   NSGA-II |   NSGA-III |   MOEA/D |
|:------------------------|---------------------:|---------------------:|------------------------:|---------------------:|:------------------|----------:|-----------:|---------:|
| F1-IV                   |               0.4114 |               0.4114 |                       0 |               0.0967 | True              |    0.4114 |     0.4114 |   0.4114 |
| F2-DEG                  |               0.0781 |               0.0781 |                       0 |               0.0189 | True              |    0.0781 |     0.0781 |   0.0781 |
| F3-CTA                  |               1.0057 |               1.0057 |                       0 |               0.2239 | True              |    1.0057 |     1.0057 |   1.0057 |
| Average objective value |               0.4984 |               0.4984 |                       0 |               0.1064 | True              |    0.4984 |     0.4984 |   0.4984 |
| Hypervolume             |               0.2545 |               0.2545 |                       0 |               0.1447 | True              |    0.2545 |     0.2545 |   0.2545 |
| IGD                     |               0.6823 |               0.6823 |                       0 |               0.18   | True              |    0.6823 |     0.6823 |   0.6823 |

