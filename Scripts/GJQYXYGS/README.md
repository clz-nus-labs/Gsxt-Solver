# 国家企业信用信息公示系统信息下载助手

这个目录放置面向国家企业信用信息公示系统的半自动下载脚本。当前默认入口使用 `https://shiming.gsxt.gov.cn/index.html`。

脚本只做浏览器自动化：读取 `data/任务清单.xlsx`，进入企业详情页，点击页面上的“更多”以及“信息下载/信息打印”，勾选需要的模块并保存下载文件。登录、验证码、企业搜索结果选择由人工在浏览器里完成。

## 目录

```text
Scripts/GJQYXYGS/
  gsxt_info_download.py       主脚本
  create_task_template.py     生成任务清单模板
  config.json                 默认配置
  config.example.json         配置示例
  requirements.txt            Python 依赖
  data/                       任务清单和浏览器状态
  downloads/                  默认下载目录
  logs/                       运行日志
```

## 安装依赖

```powershell
pip install -r Scripts\GJQYXYGS\requirements.txt
python -m playwright install chromium
```

## 生成任务清单

```powershell
python Scripts\GJQYXYGS\create_task_template.py
```

生成的文件是：

```text
Scripts\GJQYXYGS\data\任务清单.xlsx
```

建议优先填写 `详情页URL`。如果只填写 `企业名称或统一社会信用代码`，脚本会打开首页并暂停，等待你手动完成搜索、验证码和详情页选择。

## 运行

```powershell
python Scripts\GJQYXYGS\gsxt_info_download.py
```

自定义配置路径：

```powershell
python Scripts\GJQYXYGS\gsxt_info_download.py --config Scripts\GJQYXYGS\config.json
```

只生成模板不执行：

```powershell
python Scripts\GJQYXYGS\gsxt_info_download.py --init-template
```

记录一次人工操作里的关键网络请求：

```powershell
python Scripts\GJQYXYGS\capture_network.py
```

打开浏览器后，手动完成搜索、验证码、进入详情页、点击信息打印和确定。脚本会把请求记录到 `Scripts/GJQYXYGS/logs/network_capture_*.jsonl`。

脚本默认使用 Microsoft Edge。配置项是：

```json
"browser_channel": "msedge"
```

抓包脚本也默认使用 Edge；如需显式指定：

```powershell
python Scripts\GJQYXYGS\capture_network.py --channel msedge
```

如果启动后页面空白，可以让脚本只打开 Edge、不自动访问首页，然后你手动输入地址：

```powershell
python Scripts\GJQYXYGS\capture_network.py --no-goto
```

如果 Playwright 启动的 Edge 仍然空白，可以改用外部 Edge 调试端口。先关闭所有 Edge 窗口，再运行：

```powershell
.\Scripts\GJQYXYGS\start_edge_cdp.ps1
```

如果国信站点在自动打开时容易白屏，推荐只启动空白 Edge 调试窗口，然后手动输入网址和登录：

```powershell
.\Scripts\GJQYXYGS\start_edge_cdp_blank.ps1
```

登录后保持页面在首页/搜索页，再运行：

```powershell
python Scripts\GJQYXYGS\gsxt_info_download.py
```

页面打开后，再运行：

```powershell
python Scripts\GJQYXYGS\capture_network_cdp.py
```

也可以让抓包脚本自动启动外部 Edge：

```powershell
python Scripts\GJQYXYGS\capture_network_cdp.py --launch-edge
```

如果看到 `connect ECONNREFUSED 127.0.0.1:9222`，表示 9222 调试端口没有启动。先关闭所有 Edge 窗口，再运行上面的 `--launch-edge` 命令。

摘要已经保存的抓包日志：

```powershell
python Scripts\GJQYXYGS\summarize_network_log.py Scripts\GJQYXYGS\logs\network_cdp_20260610_140321.jsonl
```

分析抓包脚本退出时保存的页面快照：

```powershell
python Scripts\GJQYXYGS\analyze_page_snapshot.py Scripts\GJQYXYGS\logs\page_snapshot_YYYYMMDD_HHMMSS.html
```

## 桌面级 Edge 自动化

如果 CDP/Playwright 触发查询后出现空白页，可以改用桌面级自动化。它不读取网页 DOM，只模拟鼠标键盘操作当前正常打开的 Microsoft Edge。

安装额外依赖：

```powershell
pip install pyautogui pyperclip
```

第一次先校准坐标：

```powershell
python Scripts\GJQYXYGS\desktop_edge_download.py --calibrate
```

按提示依次把鼠标移到搜索框、查询按钮、详情页 `更多`、`发送报告`、弹窗 `发送` 的位置，然后在命令行按 Enter 记录坐标。

校准后运行：

```powershell
python Scripts\GJQYXYGS\desktop_edge_download.py
```

运行前请手动打开正常 Edge，登录并停在国信首页。验证码出现时脚本会暂停，手动完成验证码；如果企业名准确，页面应直接进入详情页，随后回命令行按 Enter 继续发送报告。

## Chrome/Edge 扩展方案

如果坐标自动化不稳定，推荐使用本地 Chrome/Edge 扩展。它运行在正常浏览器页面内部，直接操作页面 DOM，不走 Playwright/CDP，也不依赖屏幕坐标。验证码部分会截图当前验证码区域，发送给本地 `gsxt_solver` 服务识别，再按返回坐标点击。

先启动本地验证码服务：

```powershell
& C:\Users\31912\.conda\envs\paddlex_cv\python.exe .\Scripts\GJQYXYGS\server.py
```

默认读取仓库内的模型目录：

```text
dist/models/gsxt-models-v0.1.0
```

如果模型目录不存在，先按 `Gsxt-Solver` 主 README 下载并合并模型权重。

安装：

1. Chrome 打开 `chrome://extensions/`，或 Edge 打开 `edge://extensions/`
2. 开启“开发人员模式”
3. 点击“加载解压缩的扩展”
4. 选择目录：`Scripts/GJQYXYGS/edge_extension`

使用：

1. 正常打开国信网站并登录。
2. 页面右侧会出现 `GSXT 发送报告助手` 面板。
3. 粘贴企业名称或统一社会信用代码，一行一个。
4. 点击“开始”。
5. 验证码出现时扩展会自动截图并调用本地 `gsxt_solver` 服务；如果自动识别失败，再手动完成验证码，脚本会继续等待详情页、点击 `更多`、点击 `发送报告`、等待验证码/弹窗并点击 `发送`。

调试：每次调用本地验证码服务时，服务端都会把实际收到的裁剪 PNG 和结果 JSON 保存到：

```text
Scripts/GJQYXYGS/logs/captcha_debug
```

如果面板显示识别失败，优先打开最近的 PNG 看裁剪是否正确。若 PNG 为空、偏移或只截到一部分，说明是网页渲染/定位问题；若 PNG 正确但结果失败，才是模型识别问题。

扩展不会在页面加载或刷新后自动开始。只有点击面板里的 `开始` 或 `识别并执行一步`，才会操作当前页面。

## 任务清单字段

- `启用`: `是` 或 `否`
- `企业名称或统一社会信用代码`: 没有详情页 URL 时用于人工搜索
- `详情页URL`: 推荐填写，脚本会直接打开
- `下载目录`: 可选；留空则使用 `config.json` 里的 `download_root`
- `下载模块`: `全部`、`基础信息`、`企业自行公示信息`，或用英文逗号分隔多个页面标签文本
- `文件名前缀`: 可选；用于保存下载文件
- `备注`: 可选

## 配置下载目录

编辑 `config.json` 里的 `download_root`：

```json
{
  "download_root": "Scripts/GJQYXYGS/downloads"
}
```

也可以在任务清单每一行的 `下载目录` 单独指定。

## 注意

- 验证码结果如 `lot_number/pass_token/gen_time/captcha_output` 是一次性或短时效数据，不建议写死到任务清单里。
- 如果页面结构变化，优先调整 `config.json` 里的文字选择器。
- 批量访问请控制频率，并遵守目标网站规则。

## 需要补充的页面信息

为了把搜索和结果点击做得更稳，请从浏览器 DevTools 里贴这些 HTML 片段给我：

1. 首页搜索框的 HTML，以及旁边红色搜索按钮的 HTML。
2. 搜索结果列表中单条企业结果的 HTML，尤其是企业名称链接或详情按钮。
3. 详情页右上角“更多”按钮及展开菜单里“信息下载/信息打印”的 HTML。
4. “信息打印模块”区域里 `全选`、`基础信息`、`企业自行公示信息`、`确定` 的 HTML。
5. 如果点击“确定”后不是浏览器下载，而是打开新页面或接口返回文件，请提供该请求的 Network 信息：URL、method、query/body、response headers。

对应配置在 `config.json`：

```json
{
  "captcha_mode": "manual",
  "selectors": {
    "search_input_css": "#keyword",
    "search_button_css": "#btn_query",
    "download_menu_css": "#btn_print",
    "first_result_css": "",
    "detail_ready_text": "统一社会信用代码"
  }
}
```

如果你能提供稳定的 CSS 选择器，我会直接填进配置，让脚本少停顿。
