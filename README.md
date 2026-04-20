# LightRAG 知识图谱问答系统


基于 [LightRAG](https://github.com/HKUDS/LightRAG) 构建的知识图谱问答系统，支持从文档自动构建知识图谱，并提供自然语言查询和可视化功能。


## 项目概述


### 核心功能


- **文档处理**：自动从文档中提取实体和关系，构建知识图谱
- **智能问答**：支持多种查询模式（naive/local/global/hybrid）
- **图谱可视化**：交互式知识图谱展示
- **图谱搜索**：按关键词、实体类型、关系类型搜索
- **多数据源支持**：支持 TXT、Markdown、PDF、Word、HTML、JSON 等格式
<img width="1383" height="716" alt="1" src="https://github.com/user-attachments/assets/5a1f34f6-70db-4d2f-9c82-feee371d8e54" />
<img width="1370" height="816" alt="2" src="https://github.com/user-attachments/assets/2127235f-a8be-4a4f-8ec2-19432edde00f" />
<img width="1351" height="722" alt="3" src="https://github.com/user-attachments/assets/a0ea589c-b486-4cd2-9dfc-3e4feff2d561" />
<img width="1349" height="830" alt="4" src="https://github.com/user-attachments/assets/38fff3ff-4ecc-4b94-805c-06e0d0f82b67" />
<img width="1338" height="835" alt="5" src="https://github.com/user-attachments/assets/cceb9b98-bc17-4769-916f-bf2ea7a8e367" />
<img width="1349" height="838" alt="6" src="https://github.com/user-attachments/assets/75ee9dce-d144-4a69-957a-0aa38e03b1a3" />
<img width="1388" height="815" alt="7" src="https://github.com/user-attachments/assets/d771f85e-4e57-4bde-b023-5c938d946b20" />


### 技术栈


| 组件 | 支持选项 |
|------|----------|
| **LLM 提供商** | OpenAI、Ollama、Anthropic、Google Gemini、AWS Bedrock、智谱 AI |
| **向量数据库** | NanoVectorDB（内置）、Qdrant、Milvus、PostgreSQL/pgvector |
| **图数据库** | Neo4j、NetworkX、PostgreSQL、MongoDB |
| **嵌入模型** | OpenAI、Ollama、Sentence-Transformers、Jina、Gemini |
| **交互方式** | Web UI（Gradio）、命令行（CLI） |


## 项目结构


```
LightRAG-Demo/
├── main.py                    # 主入口文件
├── config/
│   ├── config.yaml            # 主配置文件
│   └── prompts.yaml           # 自定义提示词配置
├── src/
│   ├── core/
│   │   ├── config_loader.py   # 配置加载器
│   │   ├── rag_engine.py      # RAG 引擎核心
│   │   ├── entity_validator.py # 实体类型验证器
│   │   └── graph_search.py    # 图谱搜索服务
│   ├── factories/
│   │   ├── llm_factory.py     # LLM 工厂（创建模型函数）
│   │   └── embedding_factory.py # 嵌入模型工厂
│   ├── cli/
│   │   └── cli.py             # 命令行接口
│   ├── webui/
│   │   ├── webui.py           # Web UI 主模块
│   │   └── visualization.py   # 图谱可视化
│   └── utils/
│       └── helpers.py         # 工具函数
├── documents/                  # 待处理文档目录
├── rag_storage/               # 知识图谱存储目录
├── logs/                      # 日志目录
├── lib/                       # 前端库（vis.js 等）
├── tests/                     # 测试文件
├── docker-compose.yml         # Docker Compose 配置
├── Dockerfile                 # Docker 镜像构建文件
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量示例
└── README.md                  # 本文档
```


## 快速开始


### 环境要求


- Python 3.11+
- Neo4j 5.15+（图数据库）
- Qdrant（向量数据库，可选，默认使用内置 NanoVectorDB）
- Ollama（本地 LLM，可选）


### 安装步骤


1. **克隆项目**


```bash
git clone <repository-url>
cd LightRAG-Demo
```


2. **创建虚拟环境**


```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate   # Windows
```


3. **安装依赖**


```bash
pip install -r requirements.txt
```


4. **配置环境变量**


```bash
cp .env.example .env
# 编辑 .env 文件，填入您的 API 密钥和数据库配置
```


5. **启动服务**


```bash
# Web UI 模式
python main.py --mode webui


# 或 CLI 模式
# Interactive 模式
python main.py --mode cli interactive
```


### Docker 部署（推荐）


使用 Docker Compose 一键启动所有服务：


```bash
# 配置环境变量
cp .env.example .env
# 编辑 .env 文件


# 启动服务
docker-compose up -d


# 查看日志
docker-compose logs -f lightrag
```


服务启动后：
- Web UI: http://localhost:7860
- Neo4j Browser: http://localhost:7474
- Qdrant Dashboard: http://localhost:6333/dashboard


## 使用方法


### Web UI 模式


```bash
python main.py --mode webui
```


访问 http://localhost:7860，您可以：


1. **文档处理**：上传文档构建知识图谱
2. **智能问答**：自然语言提问获取答案
3. **图谱可视化**：交互式查看知识图谱
4. **图谱搜索**：按关键词、实体类型、关系筛选
5. **统计导出**：查看统计信息并导出图谱


**支持的文件类型:**
- Plain text (.txt)
- Markdown (.md)
- PDF (.pdf)
- Word (.docx)
- HTML (.html)
- JSON (.json)


### CLI 模式


```bash
# 构建知识图谱
python main.py --mode cli build --docs ./documents


# 单次查询
python main.py --mode cli query --query "什么是机器学习？"


# 交互模式
python main.py --mode cli interactive


# 导出图谱
python main.py --mode cli export --format json --output graph.json
```


---


## 📊 测试功能




```bash
pytest tests/test_basic.py -v
```


**测试覆盖率：**
- ✅ 配置加载与验证
- ✅ 辅助工具（文件处理、数据清理）
- ✅ 工厂导入与基本初始化
- ✅ 模块结构完整性


### 测试功能检查表


- [x] Web UI launches successfully
- [x] CLI commands execute without errors
- [x] Document processing completes
- [x] Query returns valid answers
- [x] Graph visualization renders
- [x] Export functions work
- [x] Configuration validation catches errors
- [x] Environment variables are resolved
- [x] Logs are written correctly


---


## 🔒 安全与最佳实践
- ✅ API密钥存储在`.env`文件中（而非代码中） 
- ✅ 敏感数据的环境变量
- ✅ `.gitignore` 排除敏感信息
- ✅ 输入验证（文件大小、格式）
- ✅ 全程错误处理
- ✅ 审计日志记录
- ✅ Web UI 可选认证


## 配置说明


配置文件位于 `config/config.yaml`，主要配置项如下：


### LLM 提供商配置


切换不同的 LLM 提供商只需修改 `llm` 配置块：


#### OpenAI


```yaml
llm:
 provider: "openai"
 model_name: "gpt-4o-mini"  # 或 gpt-4o, gpt-4-turbo
 api_key: "${OPENAI_API_KEY}"
 base_url: "https://api.openai.com/v1"  # 可选，用于代理
 temperature: 0.7
 max_tokens: 4096
```


#### Ollama（本地部署）


```yaml
llm:
 provider: "ollama"
 model_name: "qwen2.5:7b"  # 或 llama3:8b, deepseek-r1:32b
 base_url: "http://localhost:11434"
 temperature: 0.3
 max_tokens: 4096
 timeout: 600
```


#### Google Gemini


```yaml
llm:
 provider: "gemini"
 model_name: "gemini-2.0-flash"  # 或 gemini-1.5-pro
 api_key: "${GEMINI_API_KEY}"
 temperature: 0.7
 max_tokens: 8192
```


#### Anthropic Claude


```yaml
llm:
 provider: "anthropic"
 model_name: "claude-3-sonnet-20240229"
 api_key: "${ANTHROPIC_API_KEY}"
 temperature: 0.7
 max_tokens: 4096
```


#### 智谱 AI


```yaml
llm:
 provider: "zhipu"
 model_name: "glm-4"
 api_key: "${ZHIPUAI_API_KEY}"
 temperature: 0.7
```


### 嵌入模型配置


```yaml
embedding:
 provider: "ollama"           # openai, ollama, sentence-transformers, jina, gemini
 model_name: "bge-m3:latest"
 embedding_dim: 1024          # 向量维度
 batch_size: 100
```


### 数据库配置


#### Neo4j 图数据库


```yaml
graph_store:
 backend: "neo4j"
 uri: "${NEO4J_URI}"          # bolt://localhost:7687
 username: "${NEO4J_USERNAME}"
 password: "${NEO4J_PASSWORD}"
 workspace: "default"
```


#### Qdrant 向量数据库


```yaml
vector_store:
 backend: "qdrant"
 url: "${QDRANT_URL}"         # http://localhost:6333
 collection_name: "lightrag_vectors"
```


### RAG 引擎配置


```yaml
rag:
 chunk_size: 1200             # 文本分块大小（tokens）
 chunk_overlap: 100           # 分块重叠
 top_k: 60                    # 检索实体数量
 query_mode: "hybrid"         # 查询模式
 language: "Chinese"          # 语言设置
 entity_types:                # 实体类型
   - "Person"
   - "Organization"
   - "Location"
   - "Technology"
   - "Concept"
```


## 查询模式说明


| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **naive** | 简单向量搜索 | 快速查询，无需图谱遍历 |
| **local** | 基于实体的图遍历 | 查询具体实体及其直接关系 |
| **global** | 基于社区的总结 | 查询整体趋势、总结性信息 |
| **hybrid** | 混合模式 | 综合查询，推荐使用 |


## 环境变量


在 `.env` 文件中配置以下环境变量：


```bash
# LLM API Keys
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GEMINI_API_KEY=xxx
ZHIPUAI_API_KEY=xxx


# 数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
QDRANT_URL=http://localhost:6333


# Web UI
WEBUI_PASSWORD=your-password
```


## 常见问题


### 1. Ollama 模型响应慢或超时


- 增大 `timeout` 配置值
- 减小 `llm_max_async` 并发数
- 使用更小的模型（如 qwen2.5:7b）


### 2. Neo4j 连接失败


- 确认 Neo4j 服务已启动
- 检查密码配置是否正确
- 确认端口 7687 未被占用


### 3. 查询返回 "no-context"


- 检查文档是否已正确处理
- 尝试降低 `cosine_threshold` 值
- 尝试使用 `hybrid` 查询模式


### 4. 返回 "Out of memory"


减低 `config/config.yaml`的batch_size:
```yaml
document_processing:
 batch_size: 10  # Reduce from 50


performance:
 llm_max_async: 4  # Reduce from 16
 embedding_max_async: 8  # Reduce from 32
```


## 许可证


MIT License


## 致谢


- [LightRAG](https://github.com/HKUDS/LightRAG) - 核心框架
- [Gradio](https://gradio.app/) - Web UI 框架
- [Neo4j](https://neo4j.com/) - 图数据库
- [Qdrant](https://qdrant.tech/) - 向量数据库
