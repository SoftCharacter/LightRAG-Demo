"""
RAG Engine Core Module
Main orchestrator for document processing and querying
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from lightrag import LightRAG, QueryParam
from lightrag.prompt import PROMPTS

from ..core.config_loader import Config
from ..core.entity_validator import EntityTypeValidator
from ..factories.llm_factory import create_llm
from ..factories.embedding_factory import create_embedding


class RAGEngine:
    """Core RAG Engine for knowledge graph construction and querying"""

    def __init__(self, config: Config):
        """
        Initialize RAG Engine with configuration

        Args:
            config: Loaded configuration object
        """
        self.config = config
        self.logger = self._setup_logger()
        self.rag: Optional[LightRAG] = None
        self._is_initialized = False

        # Initialize entity type validator
        allowed_types = config.rag.entity_types or [
            "Person", "Organization", "Location", "Technology", "Concept", "Event", "Product"
        ]
        self.entity_validator = EntityTypeValidator(allowed_types, self.logger)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("RAGEngine")
        logger.setLevel(self.config.logging.get("level", "INFO"))
        
        # File handler
        log_path = Path(self.config.logging.get("file_path", "./logs/lightrag.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        if self.config.logging.get("console_output", True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    async def initialize(self):
        """Initialize LightRAG instance with configured components"""
        self.logger.info("Initializing RAG Engine...")

        try:
            # Set up environment variables for storage backends
            self._setup_storage_env_vars()

            # Create LLM function
            llm_func = create_llm(
                provider=self.config.llm.provider,
                model_name=self.config.llm.model_name,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url
            )
            self.logger.info(f"LLM initialized: {self.config.llm.provider}/{self.config.llm.model_name}")

            # Create embedding function
            # 对于 Ollama，需要传递 host 参数
            embedding_kwargs = {}
            if self.config.embedding.provider.lower() == "ollama":
                embedding_kwargs["host"] = self.config.llm.base_url or "http://localhost:11434"
                self.logger.info(f"Ollama Embedding host: {embedding_kwargs['host']}")

            embedding_func = create_embedding(
                provider=self.config.embedding.provider,
                model_name=self.config.embedding.model_name,
                api_key=self.config.embedding.api_key,
                embedding_dim=self.config.embedding.embedding_dim,
                max_token_size=self.config.embedding.max_token_size,
                **embedding_kwargs
            )
            self.logger.info(f"Embedding initialized: {self.config.embedding.provider}/{self.config.embedding.model_name}")

            # Load custom prompts
            self._load_prompts()

            # 构建 llm_model_kwargs（对于 Ollama 必须传递 host 参数）
            llm_model_kwargs = {}
            if self.config.llm.provider.lower() == "ollama":
                llm_model_kwargs = {
                    "host": self.config.llm.base_url or "http://localhost:11434",
                    "options": {"num_ctx": self.config.llm.max_tokens or 8192},
                    "timeout": self.config.llm.timeout or 300,
                }
                self.logger.info(f"Ollama LLM kwargs: host={llm_model_kwargs['host']}, timeout={llm_model_kwargs['timeout']}")

            # 构建 addon_params - 关键配置！传递 language 和 entity_types
            addon_params = {
                "language": self.config.rag.language or "Chinese",  # 默认中文
                "entity_types": self.config.rag.entity_types or [
                    "Person", "Organization", "Location", "Event", "Concept"
                ]
            }
            self.logger.info(f"addon_params: language={addon_params['language']}, entity_types={addon_params['entity_types']}")

            # Initialize LightRAG
            self.rag = LightRAG(
                working_dir=self.config.persistence.get("working_dir", "./rag_storage"),
                llm_model_func=llm_func,
                llm_model_name=self.config.llm.model_name,
                llm_model_kwargs=llm_model_kwargs,  # 关键修复：传递 Ollama host 配置
                embedding_func=embedding_func,

                # Storage backends
                graph_storage=self._get_graph_storage_name(),
                vector_storage=self._get_vector_storage_name(),
                kv_storage="JsonKVStorage",  # Default KV storage

                # RAG parameters
                chunk_token_size=self.config.rag.chunk_size,
                chunk_overlap_token_size=self.config.rag.chunk_overlap,
                top_k=self.config.rag.top_k,
                chunk_top_k=self.config.rag.chunk_top_k,
                entity_extract_max_gleaning=self.config.rag.max_gleaning,

                # Performance settings
                llm_model_max_async=self.config.performance.get("llm_max_async", 16),
                embedding_func_max_async=self.config.performance.get("embedding_max_async", 32),
                embedding_batch_num=self.config.embedding.batch_size,

                # Workspace
                workspace=self.config.graph_store.workspace,

                # 关键修复：传递 language 和 entity_types 配置
                addon_params=addon_params,

                # Vector DB storage kwargs (REQUIRED for all vector backends)
                vector_db_storage_cls_kwargs={
                    "cosine_better_than_threshold": self.config.rag.cosine_threshold
                }
            )

            # Initialize storages
            await self.rag.initialize_storages()
            
            self._is_initialized = True
            self.logger.info("RAG Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Engine: {e}", exc_info=True)
            raise

    def _setup_storage_env_vars(self):
        """Setup environment variables for storage backends"""
        import os

        # Set Qdrant URL if using Qdrant backend
        if self.config.vector_store.backend == "qdrant":
            qdrant_url = self.config.vector_store.url or os.getenv("QDRANT_URL", "http://localhost:6333")
            os.environ["QDRANT_URL"] = qdrant_url
            self.logger.info(f"Set QDRANT_URL: {qdrant_url}")

        # Set Neo4j environment variables if using Neo4j backend
        if self.config.graph_store.backend == "neo4j":
            # Set Neo4j URI
            neo4j_uri = self.config.graph_store.uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            os.environ["NEO4J_URI"] = neo4j_uri
            self.logger.info(f"Set NEO4J_URI: {neo4j_uri}")

            # Set Neo4j Username
            neo4j_username = self.config.graph_store.username or os.getenv("NEO4J_USERNAME", "neo4j")
            os.environ["NEO4J_USERNAME"] = neo4j_username
            self.logger.info(f"Set NEO4J_USERNAME: {neo4j_username}")

            # Set Neo4j Password
            if self.config.graph_store.password:
                os.environ["NEO4J_PASSWORD"] = self.config.graph_store.password
                self.logger.info("Set NEO4J_PASSWORD from config")

        # Set MongoDB URI if using MongoDB backend
        if self.config.graph_store.backend == "mongodb":
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            os.environ["MONGODB_URI"] = mongodb_uri
            self.logger.info(f"Set MONGODB_URI: {mongodb_uri}")

    def _load_prompts(self):
        """Load custom prompts from config"""
        if self.config.prompts:
            for key, value in self.config.prompts.items():
                if isinstance(value, str):
                    PROMPTS[key] = value
            self.logger.info(f"Loaded {len(self.config.prompts)} custom prompts")
    
    def _get_graph_storage_name(self) -> str:
        """Map config backend name to LightRAG storage class name"""
        backend_map = {
            "neo4j": "Neo4JStorage",
            "networkx": "NetworkXStorage",
            "postgres": "PostgresGraphStorage",
            "mongodb": "MongoGraphStorage"
        }
        return backend_map.get(self.config.graph_store.backend, "NetworkXStorage")
    
    def _get_vector_storage_name(self) -> str:
        """Map config backend name to LightRAG storage class name"""
        backend_map = {
            "nanovec": "NanoVectorDBStorage",
            "qdrant": "QdrantVectorDBStorage",      # ✅ Fixed: was QdrantStorage
            "milvus": "MilvusVectorDBStorage",     # ✅ Fixed: was MilvusStorage
            "postgres": "PGVectorStorage",         # ✅ Fixed: was PostgresVectorStorage
            "faiss": "FaissVectorDBStorage",       # ✅ Added
            "mongo": "MongoVectorDBStorage"        # ✅ Added
        }
        return backend_map.get(self.config.vector_store.backend, "NanoVectorDBStorage")
    
    async def process_documents(self, docs_folder: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process all documents in a folder
        
        Args:
            docs_folder: Path to folder containing documents
            progress_callback: Optional callback function for progress updates
        
        Returns:
            Processing statistics dictionary
        """
        if not self._is_initialized:
            await self.initialize()
        
        folder_path = Path(docs_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Documents folder not found: {docs_folder}")
        
        # Scan for supported files
        supported_formats = self.config.document_processing.get("supported_formats", [".txt", ".md"])
        files = []
        for ext in supported_formats:
            files.extend(folder_path.glob(f"**/*{ext}"))
        
        total_files = len(files)
        max_files = self.config.document_processing.get("max_files", 1000)
        
        if total_files > max_files:
            self.logger.warning(f"Found {total_files} files, limiting to {max_files}")
            files = files[:max_files]
        
        self.logger.info(f"Processing {len(files)} documents from {docs_folder}")
        
        # Process documents in batches
        batch_size = self.config.document_processing.get("batch_size", 50)
        processed_count = 0
        failed_count = 0
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            for j, file_path in enumerate(batch):
                try:
                    # Read file content
                    content = self._read_file(file_path)
                    
                    # Insert into RAG
                    await self.rag.ainsert(content, file_paths=[str(file_path)])
                    
                    processed_count += 1
                    current_progress = (i + j + 1) / len(files)
                    
                    if progress_callback:
                        progress_callback(current_progress, f"Processed: {file_path.name}")
                    
                    self.logger.debug(f"Processed: {file_path}")
                    
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Failed to process {file_path}: {e}")
        
        # Get statistics
        stats = await self.get_statistics()

        # 验证并修正实体类型
        validated_stats = await self._validate_and_fix_entity_types()
        stats.update(validated_stats)

        stats.update({
            "processed_files": processed_count,
            "failed_files": failed_count,
            "total_files": total_files
        })

        # 自动导出 GraphML 文件
        try:
            graphml_path = Path(self.config.persistence.get("working_dir", "./rag_storage")) / "knowledge_graph.graphml"
            graphml_data = await self.export_graph(format="graphml")
            with open(graphml_path, 'w', encoding='utf-8') as f:
                f.write(graphml_data)
            self.logger.info(f"GraphML 文件已自动保存到: {graphml_path}")
            stats["graphml_file"] = str(graphml_path)
        except Exception as e:
            self.logger.warning(f"自动导出 GraphML 失败: {e}")

        self.logger.info(f"Processing complete: {processed_count} succeeded, {failed_count} failed")
        return stats

    async def _validate_and_fix_entity_types(self) -> Dict[str, Any]:
        """
        Validate all entities in the graph and fix invalid types

        Returns:
            Statistics about validation results
        """
        import os
        from neo4j import AsyncGraphDatabase

        stats = {
            "entities_validated": 0,
            "entities_reclassified": 0,
            "entities_kept": 0
        }

        if self.config.graph_store.backend != "neo4j":
            self.logger.info("Entity validation only supported for Neo4j backend")
            return stats

        try:
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "")

            driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

            try:
                async with driver.session(database="neo4j") as session:
                    # 获取所有实体及其类型
                    result = await session.run("""
                        MATCH (n)
                        WHERE n.entity_type IS NOT NULL
                        RETURN n.id as id, n.entity_type as entity_type, n.entity_name as name
                    """)

                    entities_to_update = []
                    async for record in result:
                        entity = {
                            "id": record["id"],
                            "entity_type": record["entity_type"],
                            "entity_name": record["name"]
                        }
                        stats["entities_validated"] += 1

                        # 验证实体类型
                        validated = self.entity_validator.validate_entity(entity.copy(), mode="reclassify")
                        if validated and validated["entity_type"] != entity["entity_type"]:
                            entities_to_update.append({
                                "id": entity["id"],
                                "new_type": validated["entity_type"]
                            })
                            stats["entities_reclassified"] += 1
                        else:
                            stats["entities_kept"] += 1

                    # 批量更新实体类型
                    for entity in entities_to_update:
                        await session.run("""
                            MATCH (n {id: $id})
                            SET n.entity_type = $new_type
                        """, id=entity["id"], new_type=entity["new_type"])

                    self.logger.info(
                        f"Entity validation: {stats['entities_validated']} checked, "
                        f"{stats['entities_reclassified']} reclassified, "
                        f"{stats['entities_kept']} kept"
                    )
            finally:
                await driver.close()

        except Exception as e:
            self.logger.error(f"Entity validation failed: {e}")

        return stats

    def _read_file(self, file_path: Path) -> str:
        """Read file content based on extension"""
        ext = file_path.suffix.lower()

        if ext in [".txt", ".md"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == ".pdf":
            return self._read_pdf(file_path)
        elif ext == ".docx":
            return self._read_docx(file_path)
        elif ext == ".html":
            return self._read_html(file_path)
        elif ext == ".json":
            return self._read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _read_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        try:
            import pypdf
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("pypdf not installed. Install with: pip install pypdf")

    def _read_docx(self, file_path: Path) -> str:
        """Extract text from Word document"""
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")

    def _read_html(self, file_path: Path) -> str:
        """Extract text from HTML"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()
        except ImportError:
            raise ImportError("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")

    def _read_json(self, file_path: Path) -> str:
        """Convert JSON to readable text"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)

    async def query(self, question: str, mode: str = None) -> Dict[str, Any]:
        """
        Query the knowledge graph

        Args:
            question: User question
            mode: Query mode (naive/local/global/hybrid), defaults to config

        Returns:
            Query results with answer, sources, entities, relationships
        """
        if not self._is_initialized:
            await self.initialize()

        # --- 核心修复补丁：动态修复 LightRAG 底层模板变量名不匹配的 Bug ---
        # 将所有模板中的 {query_string} 替换为 {query}，适配底层实际传入的参数名
        for prompt_key, prompt_template in PROMPTS.items():
            if isinstance(prompt_template, str) and "{query_string}" in prompt_template:
                PROMPTS[prompt_key] = prompt_template.replace("{query_string}", "{query}")
        # -------------------------------------------------------------

        query_mode = mode or self.config.rag.query_mode
        self.logger.info(f"Querying with mode={query_mode}: {question}")

        context_info = ""

        try:
            # 1. 首先获取上下文信息
            try:
                context_param = QueryParam(
                    mode=query_mode,
                    only_need_context=True,
                )
                # 移除了上个版本错误的 query_string 参数，使用原生调用
                context_result = await self.rag.aquery(question, param=context_param)
                if context_result:
                    context_info = str(context_result)
                    self.logger.info(f"Context retrieved, length: {len(context_info)} chars")
            except Exception as ctx_err:
                self.logger.warning(f"Failed to get context in {query_mode} mode: {ctx_err}")
                # 如果图检索拿不到 Context，降级使用 naive 模式获取作为保底
                if query_mode != "naive":
                    self.logger.info("Falling back to naive mode for context retrieval...")
                    backup_param = QueryParam(mode="naive", only_need_context=True)
                    context_result = await self.rag.aquery(question, param=backup_param)
                    context_info = str(context_result) if context_result else ""

            # 2. 创建查询参数获取完整答案
            query_param = QueryParam(
                mode=query_mode,
                only_need_context=False,
            )

            # Execute query (使用原生调用，Prompt 补丁已在上方生效)
            result = await self.rag.aquery(question, param=query_param)
            self.logger.info(f"Raw query result: {str(result)[:500]}...")

            # 3. Check for no-context error
            if result is None or "[no-context]" in str(result).lower():
                self.logger.warning(f"No context found for query: {question}")
                # 尝试使用 hybrid 模式重试
                if query_mode != "hybrid":
                    self.logger.info("Retrying with hybrid mode...")
                    retry_param = QueryParam(mode="hybrid", only_need_context=False)
                    result = await self.rag.aquery(question, param=retry_param)
                    if result and "[no-context]" not in str(result).lower():
                        self.logger.info("Hybrid mode retry succeeded")
                        query_mode = "hybrid"  # 更新实际成功的模式
                    else:
                        result = "抱歉，未找到与您问题相关的内容。请尝试：\n1. 使用不同的查询模式\n2. 重新处理文档\n3. 使用更具体或更宽泛的问题"

            # 4. 处理 deepseek-r1 模型的思考过程输出（去除 <think>...</think> 标签）
            if result and isinstance(result, str):
                # 移除思考过程标签
                import re
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                result = result.strip()

            self.logger.info(f"Query completed, result length: {len(str(result))} chars")

            return {
                "answer": result or "未能生成回答",
                "mode": query_mode,
                "question": question,
                "context": context_info,
                "entities": [],
                "relationships": [],
                "chunks": []
            }
        except Exception as e:
            self.logger.error(f"Query failed: {e}", exc_info=True)
            return {
                "answer": f"查询出错：{str(e)}",
                "mode": query_mode,
                "question": question,
                "context": context_info,  # 即使失败，也尽量返回已获取到的上下文
                "entities": [],
                "relationships": [],
                "chunks": []
            }

    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        if not self._is_initialized:
            return {}

        try:
            nodes = await self.rag.chunk_entity_relation_graph.get_all_nodes()
            edges = await self.rag.chunk_entity_relation_graph.get_all_edges()

            return {
                "total_entities": len(nodes),
                "total_relationships": len(edges),
                "graph_backend": self.config.graph_store.backend,
                "vector_backend": self.config.vector_store.backend
            }
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}

    async def export_graph(self, format: str = "json") -> Any:
        """
        Export knowledge graph

        Args:
            format: Export format (json, graphml, html)

        Returns:
            Exported graph data
        """
        if not self._is_initialized:
            raise RuntimeError("RAG Engine not initialized")

        nodes = await self.rag.chunk_entity_relation_graph.get_all_nodes()
        edges = await self.rag.chunk_entity_relation_graph.get_all_edges()

        self.logger.info(f"从图存储获取: {len(nodes)} 节点, {len(edges)} 边")

        # 过滤无效节点（ID 为 None 或空）
        valid_nodes = []
        valid_node_ids = set()
        self.logger.debug(f"原始节点数量: {len(nodes)}")

        for i, node in enumerate(nodes):
            if i < 3:
                self.logger.debug(f"节点[{i}] keys: {list(node.keys())}")

            # 尝试多种可能的ID字段名
            node_id = (node.get("id") or node.get("entity_id") or
                      node.get("entity_name") or node.get("name") or
                      node.get("node_id"))
            if node_id:
                node["id"] = str(node_id)
                valid_nodes.append(node)
                valid_node_ids.add(str(node_id))

        self.logger.info(f"有效节点: {len(valid_nodes)}, 节点ID集合: {valid_node_ids}")

        # 总是尝试从 KV 存储读取关系（确保不遗漏边）
        kv_edges = await self._get_edges_from_kv_storage()
        self.logger.info(f"从 KV 存储读取到 {len(kv_edges)} 条边")

        # 合并图存储和 KV 存储的边（去重）
        edge_set = set()  # 用于去重
        all_edges = []

        for edge in list(edges) + kv_edges:
            source = (edge.get("source_id") or edge.get("source") or
                     edge.get("src_id") or edge.get("src") or
                     edge.get("from") or edge.get("head") or
                     edge.get("subject"))
            target = (edge.get("target_id") or edge.get("target") or
                     edge.get("tgt_id") or edge.get("tgt") or
                     edge.get("to") or edge.get("tail") or
                     edge.get("object"))

            if source and target:
                edge_key = (str(source), str(target))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edge["source_id"] = str(source)
                    edge["target_id"] = str(target)
                    all_edges.append(edge)

        self.logger.info(f"合并后共 {len(all_edges)} 条边（去重后）")

        # 过滤无效边（源或目标节点不存在）- 但保留所有边用于完整导出
        valid_edges = []
        for i, edge in enumerate(all_edges):
            if i < 5:
                self.logger.debug(f"边[{i}] data: {edge}")

            source_str = edge.get("source_id")
            target_str = edge.get("target_id")

            if source_str and target_str:
                # 检查节点是否存在
                source_exists = source_str in valid_node_ids
                target_exists = target_str in valid_node_ids

                if source_exists and target_exists:
                    valid_edges.append(edge)
                else:
                    self.logger.debug(f"边被过滤: {source_str}->{target_str}, source存在={source_exists}, target存在={target_exists}")

        self.logger.info(f"导出图谱: {len(valid_nodes)} 个有效节点, {len(valid_edges)} 条有效边")

        if format == "json":
            import json
            return json.dumps({
                "nodes": valid_nodes,
                "edges": valid_edges,
                "metadata": await self.get_statistics()
            }, indent=2, ensure_ascii=False)

        elif format == "graphml":
            import networkx as nx
            G = nx.DiGraph()

            # 添加节点
            added_node_ids = set()
            for node in valid_nodes:
                node_id = node.get("id")
                if node_id:
                    # 清理节点属性，确保所有值都是基本类型
                    attrs = {}
                    for k, v in node.items():
                        if k != "id" and v is not None:
                            attrs[k] = str(v) if not isinstance(v, (int, float, bool)) else v
                    G.add_node(node_id, **attrs)
                    added_node_ids.add(node_id)

            self.logger.info(f"GraphML: 添加了 {len(added_node_ids)} 个节点到图中")

            # 添加边
            edges_added = 0
            edges_skipped = 0
            for edge in valid_edges:
                source = edge.get("source_id")
                target = edge.get("target_id")
                if source and target:
                    if source in added_node_ids and target in added_node_ids:
                        # 清理边属性
                        attrs = {}
                        for k, v in edge.items():
                            if k not in ("source_id", "target_id", "source", "target") and v is not None:
                                attrs[k] = str(v) if not isinstance(v, (int, float, bool)) else v
                        G.add_edge(source, target, **attrs)
                        edges_added += 1
                    else:
                        edges_skipped += 1
                        self.logger.debug(f"GraphML 边跳过: {source}->{target}, source存在={source in added_node_ids}, target存在={target in added_node_ids}")

            self.logger.info(f"GraphML: 添加了 {edges_added} 条边, 跳过 {edges_skipped} 条边")
            self.logger.info(f"GraphML 最终图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

            # 使用 BytesIO 而不是 StringIO（networkx.write_graphml 写入字节）
            from io import BytesIO
            output = BytesIO()
            nx.write_graphml(G, output)
            return output.getvalue().decode('utf-8')

        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.rag:
            await self.rag.finalize_storages()
            self.logger.info("RAG Engine cleaned up")

    def __del__(self):
        """Destructor"""
        if self._is_initialized and self.rag:
            try:
                asyncio.run(self.cleanup())
            except:
                pass

    async def _get_edges_from_kv_storage(self) -> List[Dict[str, Any]]:
        """
        从 KV 存储文件中读取关系数据作为回退方案
        当 Neo4j 图存储的 get_all_edges() 返回空时使用
        """
        import json
        from pathlib import Path

        edges = []
        working_dir = Path(self.config.persistence.get("working_dir", "./rag_storage"))
        workspace = self.config.graph_store.workspace or "default"

        # 尝试读取 kv_store_relation_chunks.json
        relation_chunks_path = working_dir / workspace / "kv_store_relation_chunks.json"

        if relation_chunks_path.exists():
            try:
                with open(relation_chunks_path, 'r', encoding='utf-8') as f:
                    relation_data = json.load(f)

                self.logger.info(f"从 KV 存储读取到 {len(relation_data)} 条关系")

                for relation_key, rel_info in relation_data.items():
                    # 关系键格式: "源实体<SEP>目标实体"
                    if "<SEP>" in relation_key:
                        parts = relation_key.split("<SEP>")
                        if len(parts) >= 2:
                            source = parts[0].strip()
                            target = parts[1].strip()

                            edges.append({
                                "source_id": source,
                                "target_id": target,
                                "source": source,
                                "target": target,
                                "keywords": "related",
                                "description": f"关系: {source} -> {target}",
                                "chunk_ids": rel_info.get("chunk_ids", [])
                            })

                self.logger.info(f"成功解析 {len(edges)} 条边")

            except Exception as e:
                self.logger.error(f"读取 KV 关系存储失败: {e}")
        else:
            self.logger.warning(f"KV 关系存储文件不存在: {relation_chunks_path}")

        return edges

