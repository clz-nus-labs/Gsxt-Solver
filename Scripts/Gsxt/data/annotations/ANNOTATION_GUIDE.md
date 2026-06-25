# GSXT simple annotation guide

请填写 `gsxt_100_simple_annotation.xlsx`。这是给人工快速标注用的简版表格。

如果 Excel 打不开或下拉选项不可用，可以改填 `gsxt_100_simple_annotation.csv`，字段完全一致。

## 字段说明

每一行是一张图片。

- `image`: 图片文件名，不用改。
- `usable`: 默认 `yes`。图片重复、损坏、完全看不懂时改成 `no`。
- `task_type`: 从下拉框选择：
  - `char`: 汉字题
  - `icon`: 图标题
  - `mixed`: 混合题或不确定但包含两类
- `instruction_text`: 从下拉框选择：
  - `请在下图依次点击`
  - `请按语序依次点击`
  - `select in this order`
- `order_mode`: 从下拉框选择：
  - `given_order`: 按题头/提示给定顺序点击
  - `semantic_order`: 按语义顺序点击，例如“请按语序依次点击”
- `target1`, `target2`, `target3`: 按最终点击顺序填写三个目标。
- `note`: 可选说明。

## target 怎么写

直接写你看到的目标即可：

- 汉字题：写具体汉字或词，例如 `古罗马`、`烧花鸭`。
- icon 题：写中文描述，例如 `杯子`、`体温计`、`蝗虫`。
- 看不出来不用硬猜，可以写：
  - `不确定，像门把手`
  - `看不出`
  - `疑似工具`

顺序很重要：`target1 -> target2 -> target3` 就代表最终点击顺序。

暂时不用手工标坐标框。后续会用模型预标和可视化复查，再把这份表转换成训练/评测用 JSON。
