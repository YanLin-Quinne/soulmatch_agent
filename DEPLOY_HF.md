# HuggingFace Spaces 部署指南

## 步骤1：创建新的Space

1. 访问 https://huggingface.co/new-space
2. 填写信息：
   - **Owner**: Quinnnnnne
   - **Space name**: soulmatch-agent
   - **License**: MIT
   - **Select the Space SDK**: Docker
   - **Space hardware**: CPU basic (免费) 或 CPU upgrade (更快)

## 步骤2：配置环境变量

在Space的Settings → Variables and secrets中添加：

```
OPENAI_API_KEY=你的OpenAI密钥
GEMINI_API_KEY=你的Gemini密钥
DEEPSEEK_API_KEY=你的DeepSeek密钥
```

至少配置一个LLM provider的密钥。

## 步骤3：推送代码

```bash
cd /Users/quinne/Desktop/soulmatch_agent_test

# 添加HuggingFace remote
git remote add hf https://huggingface.co/spaces/Quinnnnnne/soulmatch-agent

# 推送代码
git push hf main
```

## 步骤4：等待构建

HuggingFace会自动：
1. 读取Dockerfile
2. 构建Docker镜像（约5-10分钟）
3. 启动容器
4. 在端口7860上运行应用

## 步骤5：访问应用

构建完成后，访问：
https://huggingface.co/spaces/Quinnnnnne/soulmatch-agent

## 故障排查

### 构建失败
- 检查Dockerfile语法
- 查看Build logs
- 确保requirements.txt包含所有依赖

### 运行时错误
- 检查Environment variables是否正确设置
- 查看Container logs
- 确保至少一个LLM API密钥有效

### 前端无法加载
- 确保frontend/dist目录存在
- 检查Dockerfile中的npm build步骤
- 查看静态文件路径是否正确

## 本地测试Docker构建

```bash
# 构建镜像
docker build -t soulmatch-agent .

# 运行容器
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e GEMINI_API_KEY=your_key \
  -e DEEPSEEK_API_KEY=your_key \
  soulmatch-agent

# 访问 http://localhost:7860
```

## 更新Space

每次推送到main分支，HuggingFace会自动重新构建：

```bash
git add .
git commit -m "Update: 描述你的更改"
git push hf main
```

## 性能优化

### 免费CPU basic tier限制：
- 2 vCPU
- 16GB RAM
- 50GB disk
- 适合demo和测试

### 升级到CPU upgrade ($0.03/hour)：
- 8 vCPU
- 32GB RAM
- 更快的响应速度
- 适合生产环境

## 注意事项

1. **API密钥安全**：永远不要在代码中硬编码密钥，使用环境变量
2. **CORS配置**：已在main.py中配置允许所有来源
3. **WebSocket支持**：HuggingFace Spaces支持WebSocket
4. **持久化存储**：容器重启后数据会丢失，ChromaDB使用内存模式
5. **日志查看**：在Space页面点击"Logs"查看运行日志
