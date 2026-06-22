# World Model Learning & WorldArena Challenge

项目主页：[https://cjgogo27.github.io/worldarena-wm/](https://cjgogo27.github.io/worldarena-wm/)

世界模型学习与 WorldArena 实验进展。三方对比 **ABot-PhysWorld**、**Vanilla Wan2.1**、**SFT-Wan2.1 (VideoX-Fun LoRA)** 在机器人视频生成上的表现。

## 快速开始

### 1. 提取视频文件

```bash
bash extract_videos.sh
```

这会把所有展示视频拷贝到 `videos/` 目录（约 10 个文件，共 ~3MB）。

### 2. 本地预览

推荐启动本地静态服务器预览，避免浏览器对本地视频和相对路径的限制：

```bash
python3 -m http.server 8000
```

然后打开 `http://localhost:8000/`。如果需要给远程用户看，用 Kimaki tunnel 暴露 8000 端口。

### 3. 部署到 GitHub Pages

```bash
# 1. 使用当前仓库（如 cjgogo27/worldarena-wm）
# 2. 将当前目录初始化为 git 仓库
git init
git add .
git commit -m "Initial commit: project showcase page"
git remote add origin https://github.com/cjgogo27/worldarena-wm.git
git push -u origin main

# 3. 在 GitHub 仓库 Settings → Pages 中：
#    - Source: Deploy from a branch
#    - Branch: main, / (root)
#    - Save
```

等待 1-2 分钟后即可访问 `https://cjgogo27.github.io/worldarena-wm/`

## 重要文档

| 文档 | 用途 |
|---|---|
| `docs/code_run_flow.md` | 中文代码 / 运行流程说明：目录结构、环境、数据、训练、推理、SeedVR、评测、网站预览 |
| `docs/reproduction_guide.md` | 英文复现指南：ABot、Wan2.1、VideoX-Fun、WorldArena、SeedVR 端到端流程 |

## 网站结构

```
├── index.html              # 主页面
├── css/
│   └── style.css           # 样式表
├── js/                     # (预留 JS)
├── docs/
│   ├── code_run_flow.md     # 中文代码/运行流程说明
│   └── reproduction_guide.md # 英文复现指南
├── videos/                 # 展示视频（运行 extract_videos.sh 生成）
│   ├── episode10_abot.mp4
│   ├── episode10_wan.mp4
│   ├── episode10_sft.mp4
│   ├── episode106_abot.mp4
│   ├── episode106_wan.mp4
│   ├── episode106_sft.mp4
│   ├── episode1_wan_fail.mp4
│   ├── episode100_abot_fail.mp4
│   ├── episode10_wan_fail.mp4
│   ├── episode105_wan_raw.mp4
│   └── episode105_wan_seedvr.mp4
├── extract_videos.sh       # 视频提取脚本
└── README.md
```

## 展示内容

- **三方对比**: Case 10 和 Case 106 的 ABot vs Wan2.1 vs SFT-Wan2.1 并排视频
- **失败案例**: Wan2.1 的人手 hallucination、ABot 的动作不完整
- **SeedVR 增强**: 原始视频 vs SeedVR 后处理对比
- **评测指标**: WorldArena Track 1 标准指标的对比表格
- **方法见解**: 6 个核心观察和结论

## 修改网站

直接编辑 `index.html`（内容）和 `css/style.css`（样式）即可。所有 CSS 自定义属性在 `:root` 中定义，便于统一调色。

## License

Videos and data © 2026. All Rights Reserved.
