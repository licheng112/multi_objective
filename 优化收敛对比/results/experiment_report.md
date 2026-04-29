# 优化收敛对比实验报告

## 实验设置

- 代理模型：KAN，输入 22 个关键工艺参数，输出 IV、DEG、CTA。
- 对比方法：Knowledge-constrained NSGA-II、NSGA-II、NSGA-III、MOEA/D。
- 迭代范围：0-500 代；随机种子重复次数：3。
- 目标函数：F1=|Y_IV-50|，F2=|Y_DEG-1.37|，F3=|Y_CTA-51|。
- 正文收敛图直接使用逐代当前代非支配前沿均值及当前代 HV/IGD；不再进行累计最优、事件抽点、阶梯化或后处理扰动。
- 本文方法在 0-80 代保留较强全局探索，80-160 代逐步增强知识引导，160 代后进入稳定开发阶段。

## 关键代数对比（当前代口径）

- 第 0 代 Average objective：本文方法 0.3133，NSGA-II 0.5406，NSGA-III 0.5406，MOEA/D 0.5406。
- 第 100 代 Average objective：本文方法 0.1399，NSGA-II 0.2297，NSGA-III 0.2226，MOEA/D 0.1477。
- 第 500 代 Average objective：本文方法 0.1493，NSGA-II 0.1431，NSGA-III 0.1239，MOEA/D 0.1269。
- 第 0 代 HV：本文方法 0.5403，NSGA-II 0.2303，NSGA-III 0.2303，MOEA/D 0.2303。
- 第 500 代 HV：本文方法 0.8342，NSGA-II 0.7792，NSGA-III 0.7831，MOEA/D 0.7414。
- 第 100 代 IGD：本文方法 0.0238，NSGA-II 0.1397，NSGA-III 0.1136，MOEA/D 0.1488。
- 第 500 代 IGD：本文方法 0.0104，NSGA-II 0.0332，NSGA-III 0.0347，MOEA/D 0.0899。

## 最终指标

| method                        |   front_points_mean |   front_points_std |   Final_F1_IV_mean |   Final_F1_IV_std |   Final_F2_DEG_mean |   Final_F2_DEG_std |   Final_F3_CTA_mean |   Final_F3_CTA_std |   Average_objective_mean |   Average_objective_std |   Raw_average_objective_mean |   Raw_average_objective_std |   HV_mean |   HV_std |   IGD_mean |   IGD_std |   KCSR_mean |   KCSR_std |   PAM_mean |   PAM_std |   Convergence_Generation_mean |   Convergence_Generation_std |
|:------------------------------|--------------------:|-------------------:|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|-------------------------:|------------------------:|-----------------------------:|----------------------------:|----------:|---------:|-----------:|----------:|------------:|-----------:|-----------:|----------:|------------------------------:|-----------------------------:|
| Knowledge-constrained NSGA-II |             140     |             0      |             0.072  |            0.006  |              0.0093 |             0.0009 |              0.2861 |             0.0381 |                   0.1225 |                  0.0122 |                       0.1225 |                      0.0122 |    0.8342 |   0.0227 |     0.0104 |    0.0065 |           1 |          0 |     0.0758 |    0.0037 |                       64.3333 |                      31.3422 |
| NSGA-II                       |             135     |             0      |             0.1084 |            0.013  |              0.0119 |             0.0005 |              0.2533 |             0.0059 |                   0.1245 |                  0.0027 |                       0.1245 |                      0.0027 |    0.7792 |   0.0113 |     0.0332 |    0.0072 |           1 |          0 |     0.1009 |    0.0041 |                      195      |                      54.111  |
| NSGA-III                      |             132.667 |             4.0415 |             0.0959 |            0.0132 |              0.0117 |             0.0018 |              0.2422 |             0.0327 |                   0.1166 |                  0.0146 |                       0.1166 |                      0.0146 |    0.7831 |   0.0251 |     0.0347 |    0.0155 |           1 |          0 |     0.1023 |    0.0055 |                      201      |                      79.6178 |
| MOEA/D                        |             126     |             4.3589 |             0.1218 |            0.0201 |              0.0177 |             0.0052 |              0.2472 |             0.0286 |                   0.1289 |                  0.0143 |                       0.1289 |                      0.0143 |    0.7414 |   0.0184 |     0.0899 |    0.036  |           1 |          0 |     0.1253 |    0.0111 |                      118.333  |                     121.5    |

## 图间一致性检查

| metric            | direction   | best_method                   |   ours_value |   best_value | supports_main_claim   | note                                                        |
|:------------------|:------------|:------------------------------|-------------:|-------------:|:----------------------|:------------------------------------------------------------|
| F1-IV             | min         | Knowledge-constrained NSGA-II |       0.072  |       0.072  | True                  | consistent                                                  |
| F2-DEG            | min         | Knowledge-constrained NSGA-II |       0.0093 |       0.0093 | True                  | consistent                                                  |
| F3-CTA            | min         | NSGA-III                      |       0.2861 |       0.2422 | False                 | local single-metric advantage; discuss with HV/IGD/KCSR/PAM |
| Average objective | min         | NSGA-III                      |       0.1225 |       0.1166 | False                 | local single-metric advantage; discuss with HV/IGD/KCSR/PAM |
| HV                | max         | Knowledge-constrained NSGA-II |       0.8342 |       0.8342 | True                  | consistent                                                  |
| IGD               | min         | Knowledge-constrained NSGA-II |       0.0104 |       0.0104 | True                  | consistent                                                  |
| KCSR              | max         | Knowledge-constrained NSGA-II |       1      |       1      | True                  | consistent                                                  |
| PAM               | min         | Knowledge-constrained NSGA-II |       0.0758 |       0.0758 | True                  | consistent                                                  |

## 结果分析

正文曲线展示当前代搜索状态，因此早期可能出现局部震荡或短暂退化；这是早期探索阶段的自然结果，而不是后处理平滑或人为扰动。

本文方法在 IV、DEG、HV、IGD 上整体更优，但 CTA 存在局部折中，说明知识引导并非对所有目标无条件最优。最终结论应结合 Average objective、HV、IGD、KCSR 和 PAM 共同判断，而不是只看单个目标。

## Baseline start consistency

NSGA-II, NSGA-III, and MOEA/D share the same global initial population; their generation-0 spread should stay within one estimated y-axis tick.

| metric                  |   baseline_start_min |   baseline_start_max |   baseline_start_spread |   estimated_one_tick | within_one_tick   |   NSGA-II |   NSGA-III |   MOEA/D |
|:------------------------|---------------------:|---------------------:|------------------------:|---------------------:|:------------------|----------:|-----------:|---------:|
| F1-IV                   |               0.4287 |               0.4287 |                       0 |               0.0915 | True              |    0.4287 |     0.4287 |   0.4287 |
| F2-DEG                  |               0.0766 |               0.0766 |                       0 |               0.0177 | True              |    0.0766 |     0.0766 |   0.0766 |
| F3-CTA                  |               1.1165 |               1.1165 |                       0 |               0.2159 | True              |    1.1165 |     1.1165 |   1.1165 |
| Average objective value |               0.5406 |               0.5406 |                       0 |               0.0958 | True              |    0.5406 |     0.5406 |   0.5406 |
| Hypervolume             |               0.2303 |               0.2303 |                       0 |               0.1415 | True              |    0.2303 |     0.2303 |   0.2303 |
| IGD                     |               0.7538 |               0.7538 |                       0 |               0.1824 | True              |    0.7538 |     0.7538 |   0.7538 |

