# EMO.LAB - 情绪燃料实验室 (EMO.SOUL)

> **"将情感信号转化为电信号，将情绪波动铸造成数字能源。"**

EMO.LAB 是一个沉浸式的心理觉察与生成式音乐实验室。通过 10 轮深度 AI 咨询对话，系统实时分析用户的情感波动（Sentiment Analysis），并以此驱动生成式音乐引擎（Tone.js）构建动态演进的复合和弦。最终，所有的情感能量将被转化为数字“燃料”并可进行 Token 铸造。

![EMO.LAB UI](https://img.shields.io/badge/Status-POC_v2-blueviolet)
![Vercel Deployment](https://img.shields.io/badge/Deployment-Vercel-black)

## 🌟 核心特性

- **10 轮深度觉察对话**：AI 咨询师通过关键词感知与情感分析（TextBlob），引导用户探索内心世界。
- **演进式生成音乐**：基于 `Tone.js` 的 PolySynth 与 MonoSynth，每一轮对话都会在当前音轨中加入一组新的和弦，形成 10 层叠加的情感织体。
- **沉浸式视觉环境**：
  - **动态天气系统**：支持“晴朗”、“阴雨”、“飘雪”实时渲染。
  - **全球情绪负荷图**：可视化展示全球各地区的集体情感能量分布。
  - **高保真 VLOG 背景**：精选视觉素材营造实验室氛围。
- **能源转化系统**：每一句对话的情感极性都会被量化为 `kW` (千瓦) 能量，并在完成 10 轮实验后进行 Token 铸造。

## 🛠️ 技术栈

- **Frontend**: HTML5 Canvas, Tailwind CSS, Tone.js, Lucide Icons.
- **Backend**: Python (Flask), TextBlob (NLP), SQLite.
- **Infrastructure**: Vercel Serverless Functions.

## 🚀 快速启动

1. **克隆仓库**
   ```bash
   git clone https://github.com/Iamquitesmart/emo-fuel-poc.git
   cd emo-fuel-poc
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行开发服务器**
   ```bash
   python backend/app.py
   ```

4. **访问**
   打开浏览器访问 `http://localhost:5001`

## 📦 Vercel 部署注意事项

本项目已针对 Vercel Serverless 环境进行优化：
- **只读文件系统**：SQLite 数据库与 NLTK 模型数据自动指向 `/tmp` 目录。
- **冷启动优化**：采用 `Lazy Initialization` 模式，确保 Vercel 不会因下载资源而超时。

---
Designed for **EMO.SOUL Project**. Created with ❤️ for your interview demonstration.
