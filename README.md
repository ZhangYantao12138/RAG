# RAG原型验证系统

基于检索增强生成(RAG)技术的原型验证系统，用于验证本地文档知识库与大语言模型结合的技术可行性。

## 环境要求

- Python 3.8+
- pip包管理器
- 有效的DeepSeek API密钥（用于对话生成）
- Qdrant Cloud账号

## 快速开始

### 1. 克隆项目

```bash
git clone <项目地址>
cd RAG
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

系统使用本地向量模型（sentence-transformers）进行文本向量化，无需额外配置。

### 3. 配置API密钥

1. 复制配置文件模板：
```bash
cp config.example.py config_values.py
```

2. 编辑 `config_values.py`，填入以下信息：
   - DEEPSEEK_API_KEY：从DeepSeek平台获取（用于对话生成）
   - QDRANT_URL：从Qdrant Cloud获取
   - QDRANT_API_KEY：从Qdrant Cloud获取

### 4. 准备测试文档

将需要处理的Word文档(.docx格式)放在项目根目录下，命名为 `script.docx`，或使用自定义路径。

### 5. 运行系统

1. 启动系统：
```bash
python start.py
```

2. 如果向量数据不存在，系统会自动提示上传文档：
```bash
python upload_script.py
```

3. 开始问答测试：
```bash
python query_chat.py
```

## 使用说明

### 文档上传

- 使用 `upload_script.py` 上传文档：
```bash
python upload_script.py --file <文档路径>
```

- 查看数据库信息：
```bash
python upload_script.py --info
```

- 清空现有数据：
```bash
python upload_script.py --clear
```

### 问答交互

在问答界面中可以使用以下命令：

- 直接输入问题：进行问答查询
- `/help`：显示帮助信息
- `/info`：显示数据库信息
- `/history`：显示对话历史
- `/context`：切换上下文显示模式
- `/clear`：清空对话历史
- `/quit` 或 `/exit`：退出程序

## 常见问题

1. 配置错误
   - 确保 `config_values.py` 中的API密钥正确
   - 检查网络连接是否正常

2. 文档处理失败
   - 确保文档格式为.docx
   - 检查文档是否包含有效文本内容

3. 问答失败
   - 确保向量数据库中有数据
   - 检查API调用是否成功

## 项目结构

```
RAG/
├── src/                    # 源代码目录
├── script/                 # 脚本文件
├── config_values.py        # 配置文件
├── requirements.txt        # 依赖列表
├── start.py               # 启动脚本
├── upload_script.py       # 文档上传脚本
└── query_chat.py          # 问答交互脚本
```

## 注意事项

1. 本项目仅用于技术验证，不建议用于生产环境
2. 请妥善保管API密钥，不要泄露
3. 建议使用虚拟环境运行项目
4. 文档大小建议控制在50MB以内

## 技术支持

如有问题，请提交Issue或联系技术支持。 