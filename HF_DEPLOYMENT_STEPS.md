# HuggingFace Spaces 部署步骤

## 1. 创建HuggingFace账号
1. 访问 https://huggingface.co/join
2. 注册账号(可以用GitHub登录)

## 2. 创建新的Space
1. 访问 https://huggingface.co/new-space
2. 填写信息:
   - Space name: `soulmatch-agent` (或你喜欢的名字)
   - License: MIT
   - Select SDK: **Docker**
   - Space hardware: CPU basic (免费) 或 CPU upgrade ($5/月,推荐)
   - 勾选 "Private" 如果你想保持私有

3. 点击 "Create Space"

## 3. 配置Git Remote
在本地项目目录执行:

```bash
cd /Users/quinne/soulmatch_agent
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/soulmatch-agent
```

替换 `YOUR_USERNAME` 为你的HuggingFace用户名

## 4. 配置API Keys (重要!)
在HuggingFace Space页面:
1. 点击 "Settings" 标签
2. 滚动到 "Repository secrets" 部分
3. 添加以下Secrets (点击 "New secret"):

```
Name: ANTHROPIC_API_KEY
Value: 你的Claude API Key

Name: OPENAI_API_KEY  
Value: 你的OpenAI API Key

Name: GOOGLE_API_KEY
Value: 你的Google Gemini API Key

Name: QWEN_API_KEY
Value: 你的Qwen API Key

Name: DEEPSEEK_API_KEY
Value: 你的DeepSeek API Key
```

**注意**: 至少需要配置1个API Key才能运行,推荐配置全部5个以启用智能路由

## 5. 配置持久化存储 (可选但推荐)
1. 在Space Settings中找到 "Persistent storage"
2. 点击 "Enable persistent storage"
3. 选择存储大小:
   - 免费版: 5GB
   - Pro版: 50GB
4. 挂载路径设为: `/data`

## 6. 推送代码到HuggingFace
```bash
cd /Users/quinne/soulmatch_agent
git push hf main
```

如果提示需要认证:
1. 访问 https://huggingface.co/settings/tokens
2. 创建新的Access Token (Write权限)
3. 用户名输入你的HF用户名
4. 密码输入刚创建的Token

## 7. 等待构建完成
1. 返回Space页面
2. 查看 "Logs" 标签,等待Docker构建完成(约5-10分钟)
3. 看到 "Application startup complete" 表示成功

## 8. 访问你的应用
Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/soulmatch-agent`

## 9. 测试API端点
```bash
# 替换为你的Space URL
SPACE_URL="https://YOUR_USERNAME-soulmatch-agent.hf.space"

# 健康检查
curl $SPACE_URL/health

# 开始画像模式
curl -X POST $SPACE_URL/api/profiling/start \
  -H "Content-Type: application/json" \
  -d '{"persona_id": 0}'
```

## 故障排查

### 问题1: 构建失败
- 检查Logs标签查看错误信息
- 确保Dockerfile语法正确
- 确保requirements.txt中的依赖可安装

### 问题2: API返回500错误
- 检查是否配置了至少1个API Key
- 查看Logs标签的运行时错误
- 确认API Key有效且有余额

### 问题3: 持久化存储未生效
- 确认已在Settings中启用Persistent storage
- 重启Space: Settings → Factory reboot

### 问题4: 前端显示"Frontend not built"
- 检查Dockerfile中的frontend构建步骤
- 确保npm install和npm run build成功执行

## 更新部署
每次本地修改后:
```bash
git add -A
git commit -m "Update: 描述你的修改"
git push hf main
```

HuggingFace会自动重新构建和部署。

## 成本估算
- CPU basic (免费): 适合演示和测试
- CPU upgrade ($5/月): 推荐,响应更快
- Persistent storage (免费5GB): 推荐启用
- API调用成本: 根据使用量,优先使用Qwen/DeepSeek降低成本

## 下一步
部署成功后,你可以:
1. 分享Space URL给朋友测试
2. 在README中添加Space badge
3. 监控API使用量和成本
4. 根据用户反馈迭代改进
