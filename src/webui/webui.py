"""
Gradio Web UI Module
Provides a user-friendly web interface for the RAG system
"""

import gradio as gr
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 启用嵌套事件循环支持，解决 Gradio 中的 asyncio.run() 冲突
import nest_asyncio
nest_asyncio.apply()

from src.core.config_loader import load_config
from src.core.rag_engine import RAGEngine
from src.core.graph_search import GraphSearchService
from src.webui.visualization import create_interactive_graph


# Global variables
engine: RAGEngine = None
config = load_config("config/config.yaml")
graph_search: GraphSearchService = None
_engine_lock = threading.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """获取或创建事件循环，确保在整个应用中使用同一个循环"""
    global _event_loop
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        pass

    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
    return _event_loop


def run_async(coro):
    """安全地运行异步协程，处理事件循环"""
    try:
        loop = asyncio.get_running_loop()
        # 如果已经在事件循环中，使用 nest_asyncio 允许嵌套
        return asyncio.run(coro)
    except RuntimeError:
        # 没有运行中的循环，创建新的
        loop = get_or_create_event_loop()
        return loop.run_until_complete(coro)


async def initialize_engine():
    """Initialize RAG engine (called once)"""
    global engine, config
    with _engine_lock:
        if engine is None:
            config = load_config("config/config.yaml")
            engine = RAGEngine(config)
            await engine.initialize()
    return engine


async def process_documents_async(docs_folder: str, progress=gr.Progress()):
    """Process documents with progress updates"""
    global engine
    await initialize_engine()
    
    progress(0, desc="Starting document processing...")
    
    def update_progress(ratio, desc):
        progress(ratio, desc=desc)
    
    stats = await engine.process_documents(docs_folder, progress_callback=update_progress)
    
    # Format statistics for display
    result = f"""
## ✅ Processing Complete!

### 📊 Statistics
- **Processed Files**: {stats.get('processed_files', 0)}
- **Failed Files**: {stats.get('failed_files', 0)}
- **Total Entities**: {stats.get('total_entities', 0)}
- **Total Relationships**: {stats.get('total_relationships', 0)}
- **Graph Backend**: {stats.get('graph_backend', 'N/A')}
- **Vector Backend**: {stats.get('vector_backend', 'N/A')}
"""
    
    return result, stats


def process_documents_sync(docs_folder: str, progress=gr.Progress()):
    """Synchronous wrapper for Gradio"""
    return run_async(process_documents_async(docs_folder, progress))


async def query_async(question: str, mode: str):
    """Query knowledge graph"""
    global engine
    await initialize_engine()

    if not question.strip():
        return "Please enter a question.", ""

    result = await engine.query(question, mode=mode)

    # Format answer
    answer_md = f"""
### 💡 Answer
{result['answer']}

---
**Query Mode**: {result['mode']}
"""

    # 提取上下文信息
    context_parts = []

    # 从结果中提取上下文（如果可用）
    if 'context' in result and result['context']:
        context_parts.append("### 📄 相关上下文\n")
        context_parts.append(str(result['context'])[:2000])  # 限制长度

    # 提取实体信息
    if 'entities' in result and result['entities']:
        context_parts.append("\n\n### 🏷️ 相关实体\n")
        for entity in result['entities'][:10]:  # 最多显示10个
            if isinstance(entity, dict):
                context_parts.append(f"- **{entity.get('name', 'Unknown')}**: {entity.get('description', '')[:100]}\n")
            else:
                context_parts.append(f"- {entity}\n")

    # 提取关系信息
    if 'relationships' in result and result['relationships']:
        context_parts.append("\n\n### 🔗 相关关系\n")
        for rel in result['relationships'][:10]:
            if isinstance(rel, dict):
                src = rel.get('source', '')
                tgt = rel.get('target', '')
                desc = rel.get('description', '')[:80]
                context_parts.append(f"- {src} → {tgt}: {desc}\n")
            else:
                context_parts.append(f"- {rel}\n")

    # 提取来源块信息
    if 'chunks' in result and result['chunks']:
        context_parts.append("\n\n### 📚 来源文档片段\n")
        for i, chunk in enumerate(result['chunks'][:5], 1):
            if isinstance(chunk, dict):
                content = chunk.get('content', str(chunk))[:200]
            else:
                content = str(chunk)[:200]
            context_parts.append(f"**[{i}]** {content}...\n\n")

    context_info = "".join(context_parts) if context_parts else "未找到相关上下文信息。"

    return answer_md, context_info


def query_sync(question: str, mode: str):
    """Synchronous wrapper for Gradio"""
    return run_async(query_async(question, mode))


async def get_stats_async():
    """Get knowledge graph statistics"""
    global engine
    await initialize_engine()
    
    stats = await engine.get_statistics()
    
    stats_md = f"""
## 📊 Knowledge Graph Statistics

- **Total Entities**: {stats.get('total_entities', 0)}
- **Total Relationships**: {stats.get('total_relationships', 0)}
- **Graph Backend**: {stats.get('graph_backend', 'N/A')}
- **Vector Backend**: {stats.get('vector_backend', 'N/A')}
"""
    
    return stats_md


def get_stats_sync():
    """Synchronous wrapper for Gradio"""
    return run_async(get_stats_async())


async def export_graph_async(format: str):
    """Export knowledge graph"""
    global engine
    await initialize_engine()

    data = await engine.export_graph(format=format)

    filename = f"knowledge_graph.{format}"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(data)

    return filename, f"✅ Exported to {filename}"


def export_graph_sync(format: str):
    """Synchronous wrapper for Gradio"""
    return run_async(export_graph_async(format))


async def visualize_graph_async():
    """Generate interactive graph visualization"""
    global engine
    await initialize_engine()

    try:
        import os
        from neo4j import AsyncGraphDatabase

        # 直接使用 Neo4j 连接获取数据，不依赖 LightRAG 内部结构
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")

        nodes = []
        edges = []
        node_ids_set = set()

        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        try:
            async with driver.session(database="neo4j") as session:
                # 获取节点 - 使用更灵活的查询，检查所有可能的属性名
                node_query = """
                MATCH (n)
                RETURN
                    coalesce(n.entity_id, n.id, n.name, toString(id(n))) as id,
                    coalesce(n.entity_name, n.name, n.entity_id, n.id) as entity_name,
                    coalesce(n.entity_type, head(labels(n)), 'Other') as entity_type,
                    coalesce(n.description, n.desc, '') as description
                LIMIT 200
                """
                node_result = await session.run(node_query)
                async for record in node_result:
                    node_id = record['id']
                    if node_id and node_id not in node_ids_set:
                        node_ids_set.add(node_id)
                        nodes.append({
                            'id': node_id,
                            'entity_name': record['entity_name'] or node_id,
                            'entity_type': record['entity_type'] or 'Other',
                            'description': record['description'] or ''
                        })

                # 获取边 - 同样使用灵活查询
                edge_query = """
                MATCH (a)-[r]->(b)
                RETURN
                    coalesce(a.entity_id, a.id, a.name, toString(id(a))) as source,
                    coalesce(b.entity_id, b.id, b.name, toString(id(b))) as target,
                    coalesce(r.keywords, r.keyword, type(r), 'related') as relation,
                    coalesce(r.description, r.desc, '') as description
                LIMIT 500
                """
                edge_result = await session.run(edge_query)
                async for record in edge_result:
                    source = record['source']
                    target = record['target']
                    if source and target and source in node_ids_set and target in node_ids_set:
                        edges.append({
                            'source_id': source,
                            'target_id': target,
                            'keywords': record['relation'] or 'related',
                            'description': record['description'] or ''
                        })
        finally:
            await driver.close()

        # 如果 Neo4j 中没有边，尝试从 KV 存储读取
        if not edges and nodes:
            edges = await _get_edges_from_kv_storage(node_ids_set)

        if not nodes:
            return """<div style='padding: 20px; color: white; background: #333;'>
                <h3>📭 图谱为空</h3>
                <p>未找到实体节点。请先在"文档处理"标签页处理文档。</p>
            </div>"""

        html = create_interactive_graph(
            nodes, edges,
            top_n_per_type=config.webui.get('visualization',{}).get('top_n_per_type',3)
        )
        return html

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""<div style='color: #ff6b6b; padding: 20px; background: #333;'>
            <h3>❌ 可视化错误</h3>
            <pre style='overflow-x: auto;'>{error_details}</pre>
        </div>"""


def visualize_graph_sync():
    """Synchronous wrapper for Gradio"""
    return run_async(visualize_graph_async())


async def _get_edges_from_kv_storage(node_ids_set: set) -> list:
    """
    从 KV 存储文件中读取关系数据作为回退方案
    当 Neo4j 图存储的边查询返回空时使用
    """
    import json

    edges = []
    # 尝试读取 kv_store_relation_chunks.json
    relation_chunks_path = Path("./rag_storage/default/kv_store_relation_chunks.json")

    if relation_chunks_path.exists():
        try:
            with open(relation_chunks_path, 'r', encoding='utf-8') as f:
                relation_data = json.load(f)

            print(f"[KV回退] 从 KV 存储读取到 {len(relation_data)} 条关系")

            for relation_key, rel_info in relation_data.items():
                # 关系键格式: "源实体<SEP>目标实体"
                if "<SEP>" in relation_key:
                    parts = relation_key.split("<SEP>")
                    if len(parts) >= 2:
                        source = parts[0].strip()
                        target = parts[1].strip()

                        # 只添加两端节点都存在的边
                        if source in node_ids_set and target in node_ids_set:
                            edges.append({
                                "source_id": source,
                                "target_id": target,
                                "keywords": "related",
                                "description": f"关系: {source} → {target}"
                            })

            print(f"[KV回退] 成功解析 {len(edges)} 条有效边")

        except Exception as e:
            print(f"[KV回退] 读取 KV 关系存储失败: {e}")

    return edges


# ==================== 图谱搜索功能 ====================

async def init_graph_search():
    """Initialize graph search service"""
    global graph_search
    if graph_search is None:
        graph_search = GraphSearchService()
    return graph_search


async def search_by_keywords_async(keywords: str):
    """Search graph by keywords"""
    await init_graph_search()
    if not keywords.strip():
        return "<div style='color: #888; padding: 20px;'>请输入搜索关键词</div>", []

    result = await graph_search.search_by_keywords(keywords)
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    if not nodes:
        return f"<div style='color: #888; padding: 20px;'>未找到与 '{keywords}' 相关的实体</div>", []

    html = create_interactive_graph(
        nodes, edges,
        top_n_per_type=config.webui.get('visualization', {}).get('top_n_per_type', 3)
    )

    # 创建结果表格
    table_data = [[n['entity_name'], n['entity_type'], n['description'][:100]] for n in nodes[:20]]

    return html, table_data


def search_by_keywords_sync(keywords: str):
    """Sync wrapper for keyword search"""
    return run_async(search_by_keywords_async(keywords))


async def search_by_type_async(entity_type: str, name_filter: str):
    """Search graph by entity type"""
    await init_graph_search()

    result = await graph_search.search_by_entity_type(entity_type or "", name_filter or "")
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    if not nodes:
        return "<div style='color: #888; padding: 20px;'>未找到符合条件的实体</div>", []

    html = create_interactive_graph(
        nodes, edges,
        top_n_per_type=config.webui.get('visualization', {}).get('top_n_per_type', 3)
    )
    table_data = [[n['entity_name'], n['entity_type'], n['description'][:100]] for n in nodes[:20]]

    return html, table_data


def search_by_type_sync(entity_type: str, name_filter: str):
    """Sync wrapper for type search"""
    return run_async(search_by_type_async(entity_type, name_filter))


async def search_by_relation_async(relation_type: str, source_filter: str, target_filter: str):
    """Search graph by relationship"""
    await init_graph_search()

    result = await graph_search.search_by_relation(
        relation_type or "", source_filter or "", target_filter or ""
    )
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    if not edges:
        return "<div style='color: #888; padding: 20px;'>未找到符合条件的关系</div>", []

    html = create_interactive_graph(
        nodes, edges,
        top_n_per_type=config.webui.get('visualization', {}).get('top_n_per_type', 3)
    )
    table_data = [[e['source_id'], e['keywords'], e['target_id']] for e in edges[:20]]

    return html, table_data


def search_by_relation_sync(relation_type: str, source_filter: str, target_filter: str):
    """Sync wrapper for relation search"""
    return run_async(search_by_relation_async(relation_type, source_filter, target_filter))


async def get_entity_types_async():
    """Get all entity types"""
    await init_graph_search()
    types = await graph_search.get_entity_types()
    return gr.update(choices=[""] + types)


def get_entity_types_sync():
    """Sync wrapper"""
    return run_async(get_entity_types_async())


async def get_relation_types_async():
    """Get all relation types"""
    await init_graph_search()
    types = await graph_search.get_relation_types()
    return gr.update(choices=[""] + types)


def get_relation_types_sync():
    """Sync wrapper"""
    return run_async(get_relation_types_async())


def create_web_interface():
    """Create Gradio web interface"""

    # Create interface with Chinese language setting
    with gr.Blocks(
        title="LightRAG 知识图谱问答系统",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { font-family: 'Microsoft YaHei', 'SimHei', 'Segoe UI', sans-serif; }
        """
    ) as demo:

        gr.Markdown("""
        # 🧠 LightRAG 知识图谱问答系统
        ### 从文档构建知识图谱，并使用自然语言进行查询
        """)
        
        with gr.Tabs():
            
            # Tab 1: Document Processing
            with gr.Tab("📚 文档处理"):
                gr.Markdown("### 上传文档以构建知识图谱")

                with gr.Row():
                    docs_folder_input = gr.Textbox(
                        label="文档文件夹路径",
                        placeholder="./documents",
                        value="./documents",
                        interactive=True
                    )

                    def browse_folder():
                        """Return the absolute path of documents folder"""
                        from pathlib import Path
                        docs_path = Path("./documents")
                        if docs_path.exists():
                            return str(docs_path.absolute())
                        return "./documents"

                    browse_btn = gr.Button("📁 浏览", size="sm", variant="secondary")
                    browse_btn.click(
                        fn=browse_folder,
                        outputs=docs_folder_input
                    )

                process_btn = gr.Button("🚀 构建知识图谱", variant="primary", size="lg")

                process_output = gr.Markdown(label="处理结果")
                process_stats = gr.JSON(label="详细统计")
                
                process_btn.click(
                    fn=process_documents_sync,
                    inputs=[docs_folder_input],
                    outputs=[process_output, process_stats]
                )
            
            # Tab 2: Question Answering
            with gr.Tab("💬 问答"):
                gr.Markdown("### 提问关于您的文档")

                with gr.Row():
                    with gr.Column(scale=4):
                        question_input = gr.Textbox(
                            label="您的问题",
                            placeholder="什么是机器学习？",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        mode_dropdown = gr.Dropdown(
                            choices=["naive", "local", "global", "hybrid"],
                            value="hybrid",
                            label="查询模式",
                            info="推荐使用混合模式"
                        )

                query_btn = gr.Button("🔍 搜索", variant="primary", size="lg")

                with gr.Row():
                    answer_output = gr.Markdown(label="答案")
                    context_output = gr.Textbox(label="上下文和来源", lines=10)

                query_btn.click(
                    fn=query_sync,
                    inputs=[question_input, mode_dropdown],
                    outputs=[answer_output, context_output]
                )

                # Example questions
                gr.Examples(
                    examples=[
                        ["这些文档的主要主题是什么？", "global"],
                        ["解释关键概念之间的关系", "local"],
                        ["总结主要发现", "hybrid"]
                    ],
                    inputs=[question_input, mode_dropdown]
                )

            # Tab 3: Knowledge Graph Visualization
            with gr.Tab("🕸️ 图谱可视化"):
                gr.Markdown("### 交互式知识图谱可视化")

                visualize_btn = gr.Button("🎨 生成可视化", variant="primary")

                graph_html = gr.HTML(label="交互式图谱")

                visualize_btn.click(
                    fn=visualize_graph_sync,
                    outputs=graph_html
                )

                gr.Markdown("""
                **操作说明：**
                - 鼠标滚轮放大/缩小
                - 拖动节点重新排列
                - 点击节点查看详情
                - 不同颜色代表不同的实体类型
                - 单击节点展开子节点
                - 双击节点收起子节点
                """)

            # Tab 4: Graph Search (新增)
            with gr.Tab("🔍 图谱搜索"):
                gr.Markdown("### 自然语言知识图谱搜索")

                with gr.Tabs():
                    # 子标签页1: 关键词搜索
                    with gr.Tab("🔤 关键词搜索"):
                        gr.Markdown("输入关键词搜索相关实体和关系")

                        keyword_input = gr.Textbox(
                            label="搜索关键词",
                            placeholder="例如：张三、人工智能、北京...",
                            lines=1
                        )
                        keyword_search_btn = gr.Button("🔍 搜索", variant="primary")

                        keyword_graph = gr.HTML(label="搜索结果图谱")
                        keyword_table = gr.Dataframe(
                            headers=["实体名称", "类型", "描述"],
                            label="匹配的实体列表"
                        )

                        keyword_search_btn.click(
                            fn=search_by_keywords_sync,
                            inputs=[keyword_input],
                            outputs=[keyword_graph, keyword_table]
                        )

                    # 子标签页2: 实体类型筛选
                    with gr.Tab("📋 实体筛选"):
                        gr.Markdown("按实体类型和名称筛选")

                        with gr.Row():
                            entity_type_dropdown = gr.Dropdown(
                                label="实体类型",
                                choices=[""],
                                value="",
                                allow_custom_value=True
                            )
                            refresh_types_btn = gr.Button("🔄 刷新类型", size="sm")

                        entity_name_filter = gr.Textbox(
                            label="名称关键词（可选）",
                            placeholder="输入实体名称关键词...",
                            lines=1
                        )
                        entity_search_btn = gr.Button("🔍 筛选", variant="primary")

                        entity_graph = gr.HTML(label="筛选结果图谱")
                        entity_table = gr.Dataframe(
                            headers=["实体名称", "类型", "描述"],
                            label="筛选的实体列表"
                        )

                        refresh_types_btn.click(
                            fn=get_entity_types_sync,
                            outputs=[entity_type_dropdown]
                        )

                        entity_search_btn.click(
                            fn=search_by_type_sync,
                            inputs=[entity_type_dropdown, entity_name_filter],
                            outputs=[entity_graph, entity_table]
                        )

                    # 子标签页3: 关系筛选
                    with gr.Tab("🔗 关系筛选"):
                        gr.Markdown("按关系类型和源/目标实体筛选")

                        with gr.Row():
                            relation_type_dropdown = gr.Dropdown(
                                label="关系类型",
                                choices=[""],
                                value="",
                                allow_custom_value=True
                            )
                            refresh_relations_btn = gr.Button("🔄 刷新类型", size="sm")

                        with gr.Row():
                            source_filter = gr.Textbox(
                                label="源实体关键词",
                                placeholder="源实体名称...",
                                lines=1
                            )
                            target_filter = gr.Textbox(
                                label="目标实体关键词",
                                placeholder="目标实体名称...",
                                lines=1
                            )

                        relation_search_btn = gr.Button("🔍 筛选", variant="primary")

                        relation_graph = gr.HTML(label="筛选结果图谱")
                        relation_table = gr.Dataframe(
                            headers=["源实体", "关系", "目标实体"],
                            label="筛选的关系列表"
                        )

                        refresh_relations_btn.click(
                            fn=get_relation_types_sync,
                            outputs=[relation_type_dropdown]
                        )

                        relation_search_btn.click(
                            fn=search_by_relation_sync,
                            inputs=[relation_type_dropdown, source_filter, target_filter],
                            outputs=[relation_graph, relation_table]
                        )

            # Tab 5: Statistics & Export
            with gr.Tab("📊 统计与导出"):
                gr.Markdown("### 查看统计信息并导出知识图谱")

                with gr.Row():
                    stats_btn = gr.Button("📈 刷新统计", variant="secondary")

                stats_output = gr.Markdown(label="统计信息")

                stats_btn.click(
                    fn=get_stats_sync,
                    outputs=stats_output
                )

                gr.Markdown("---")
                gr.Markdown("### 导出知识图谱")

                with gr.Row():
                    export_format = gr.Radio(
                        choices=["json", "graphml"],
                        value="json",
                        label="导出格式"
                    )
                    export_btn = gr.Button("💾 导出", variant="primary")

                export_file = gr.File(label="下载文件")
                export_status = gr.Markdown()

                export_btn.click(
                    fn=export_graph_sync,
                    inputs=[export_format],
                    outputs=[export_file, export_status]
                )

            # Tab 5: Configuration
            with gr.Tab("⚙️ 配置"):
                gr.Markdown("### 当前配置（只读）")

                def load_config_display():
                    global config
                    if config:
                        return {
                            "LLM 提供商": config.llm.provider,
                            "LLM 模型": config.llm.model_name,
                            "嵌入提供商": config.embedding.provider,
                            "向量存储": config.vector_store.backend,
                            "图存储": config.graph_store.backend,
                            "查询模式": config.rag.query_mode,
                            "块大小": config.rag.chunk_size
                        }
                    return {"状态": "配置未加载"}

                config_display = gr.JSON(
                    label="活跃配置",
                    value=load_config_display
                )

                gr.Markdown("""
                **修改配置：**
                1. 编辑 `config/config.yaml`
                2. 重启应用程序

                **环境变量：**
                - `OPENAI_API_KEY`: OpenAI API 密钥
                - `NEO4J_PASSWORD`: Neo4j 数据库密码
                - `QDRANT_URL`: Qdrant 向量数据库 URL
                - `WEBUI_PASSWORD`: Web UI 密码（如果启用身份验证）
                """)

        gr.Markdown("""
        ---
        ### 📖 快速开始指南
        1. **处理文档**: 在标签页 1 上传您的文档
        2. **提问**: 在标签页 2 查询知识图谱
        3. **可视化**: 在标签页 3 查看实体关系
        4. **导出**: 在标签页 4 下载图数据

        **支持的查询模式：**
        - **Naive**: 简单向量搜索（快速）
        - **Local**: 基于实体的图遍历（精确）
        - **Global**: 基于社区的总结（全面）
        - **Hybrid**: 混合方法（推荐）
        """)

    return demo


def launch_webui(host: str = "0.0.0.0", port: int = 7860, share: bool = False, auth: tuple = None):
    """Launch Gradio web interface"""
    demo = create_web_interface()

    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        auth=auth,
        show_error=True
    )


if __name__ == "__main__":
    # Load config for webui settings
    try:
        config = load_config("config/config.yaml")
        webui_config = config.webui

        host = webui_config.get("host", "0.0.0.0")
        port = webui_config.get("port", 7860)
        share = webui_config.get("share", False)

        auth = None
        if webui_config.get("auth_enabled", False):
            auth = (webui_config.get("username", "admin"), webui_config.get("password", ""))

        print(f"🚀 Launching LightRAG Web UI at http://{host}:{port}")
        launch_webui(host=host, port=port, share=share, auth=auth)

    except Exception as e:
        print(f"❌ Failed to launch Web UI: {e}")
        import traceback
        traceback.print_exc()


