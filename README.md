# cv_maplescreen

从冒险岛公屏截图提取聊天记录。每条发言对应一个 JSON 对象，续行合并到同一条正文。

## 安装

在项目目录打开 PowerShell。以下命令使用 `cv_maplescreen` Conda 环境的 Python，依赖安装在项目内的 `.deps`。

```powershell
$cvPython = 'D:\conda\envs\cv_maplescreen\python.exe'
& $cvPython -m pip install --target .deps -r requirements.txt
```

## 识别截图

如果截图还包含游戏画面，打开后用鼠标拖出聊天区，按 Enter 确认或按 C 取消。

```powershell
& $cvPython .\extract_chat.py '你的截图.png' --select-roi -o .\messages.json
```

已裁好的聊天图可以省略 `--select-roi`。也可以用 `--roi X Y 宽 高` 传入图片像素坐标，原点在左上角。省略 `-o` 时，程序把 JSON 打印到终端。

## 实时监控

打开游戏并露出聊天区，然后在项目目录运行以下命令。

```powershell
& 'D:\conda\envs\cv_maplescreen\python.exe' .\monitor_chat.py
```

1. 在桌面预览里框住聊天文字区域，按 Enter 开始。按 Esc 取消框选。
2. 在浏览器打开 [实时公屏](http://127.0.0.1:8765/)。上方会显示当前选区的画面、坐标和尺寸，下方会自动追加识别结果。保持选区在屏幕上可见，把浏览器放在选区旁边或另一块屏幕上。
3. 在 PowerShell 按 Ctrl+C 停止采集。程序会处理完已入队的截图。

默认识别框选后出现的新消息。加 `--include-existing` 可以把框选时已经可见的消息也读出来。窗口移动后重新启动并框选。页面只在本机开放，启动时从 SQLite 读取最近 500 条，之前会话的记录标记为历史。

数据库默认是 `outputs/maplescreen.sqlite3`，保存采集会话、选区坐标和逐条消息。`--database` 可指定其他路径。消息也会追加到 `outputs/live_messages.jsonl`，可以直接打开 `demo.html` 导入查看。

屏幕采集以 30 FPS 为目标，网页预览最多每秒更新 5 次，识别在后台进行。页面上的待识别数量表示还有多少批新行在等候。采集缺口表示画面跳变导致跟踪中断，或积压达到上限时有批次未入队。

## 输出字段

| 字段 | 内容 |
| --- | --- |
| `medal` | 含尖括号的勋章名，例如 `<任务狂人勋章>` |
| `player_name` | 玩家名，例如 `Stephen.Chow` |
| `channel` | 公屏上显示的频道，例如 `CH09` |
| `text` | 分隔冒号后的正文，游戏界面自动折出的续行直接拼接 |
| `message_type` | `megaphone`（喇叭）、`local`（当前频道）、`system`（系统提示）或 `unknown`（版式不确定） |

缺失或未识别的字段为 `null`。记录按截图从上到下排列，重复发言分别保留。静态版的徽章定位以示例截图的配色为基准，保留原图分辨率有助于识别小字。
