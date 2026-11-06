# IEEE TrafficFlowBench 提交包

| 任务 | 本次采用的版本 |
| --- | --- |
| Task1 | Oscar 的严重缺失增强混合方案。公开训练集二月开发分 (S_{state}=0.94581962)。|
| Task2 | Liu 仓库的旧版 0.827225403 代理 IoU 方案。保留其两期共 174,000 行正式模板预测。|
| Task3 | 评测器直接使用本包的 Task1 速度和流量。二月相邻目标车辆积累变化 RMSE 为 1.81864846。|
| Task4 | Oscar 保存的冻结 OD 预测，两期共 70,708 行。最新的 OD 混合研究保存在相邻目录。|

以上指标来自各自的开发口径。本组合的官方总分待评测。

## 构建完整提交

在 Windows 使用 Python 3.12 和 Task1 原包的依赖，准备从 Kaggle 获得的 `kaggle_public`。原始数据与构建产物保存在本地。

1. 按 [Task1 基础模型说明](experiments/task1/best_road_combination_20261026/reproduction/README_完整复现.md) 中的 `pipeline.py submit` 生成基础状态预测。使用仓库附带的基础权重。把生成的 `submissions` 目录传给下一步。
2. 在本仓库运行：

   ```powershell
   python -X utf8 tools/build_submission.py --data-root D:/data/kaggle_public --base-state-dir D:/work/base/submissions
   ```

   命令按道路和 split 应用 Oscar 权重，复用仓库内的旧 Task2 预测和冻结 Task4。已有 Oscar 校正好的状态 CSV 时可以传 `--state-dir`。
3. 将生成的 `artifacts/submission.zip` 用于 Kaggle 提交。ZIP 只有 `submission.csv`，共 **6,980,503** 行。构建器检查模板键、OD 区域、二值队列、流量合法性和 ZIP 完整性。验收与哈希记录在 `artifacts/build_acceptance.json`。

如果后续确认换用新的 Task4 文件，传入 `--task4 路径`，按相同键重新组包。

## 复现来源

- Oscar 主分支 [08e4921](https://github.com/OscarZH2005030/25-day-a-Big-Dream/tree/08e49214061115367fd664ef92c50a289153e526/experiments/trafficflowbench_local_20261102)。校正权重、推理代码和冻结 Task4 CSV 已对照该提交的 Git blob。
- Liu 仓库 [d4a4810](https://github.com/chengweiliu669-pixel/25-day-a-Big-Dream-submission/tree/d4a4810cf4add1cec039878eb87b9ae48c684b73) 是旧 Task2 导出及 Task1 基础模型的来源。若要从原始观测重跑 Task2，先获取 [R127 父模型包](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-r127-repro-20261027)，然后按 `tools/export_queue.py --help` 执行。
- [完整提交验收与 ZIP 指纹](provenance/build_acceptance.json) · [旧 Task2 导出验收](provenance/task2_export_acceptance.json) · [Oscar 原状态验收](provenance/oscar_state_export_reference.json)
- [Oscar 最新 Task4 研究](experiments/trafficflowbench_local_20261102/task4/CYCLE_FLOW_BLEND_DIRECTION.md) · [此前 Task4 本地研究结果](docs/task4-research)
