"""
Graph Search Service Module
Provides natural language and filtered search capabilities for the knowledge graph
"""

import os
import logging
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase


class GraphSearchService:
    """Service for searching the knowledge graph"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("GraphSearchService")
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
    
    async def search_by_keywords(self, keywords: str, limit: int = 50) -> Dict[str, Any]:
        """
        Search entities and relationships by keywords (natural language)

        Args:
            keywords: Search keywords or natural language query
            limit: Maximum results to return

        Returns:
            Dict with nodes and edges matching the search
        """
        driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

        nodes = []
        edges = []
        node_ids = set()

        try:
            async with driver.session(database="neo4j") as session:
                # 使用更灵活的搜索，检查所有可能的属性
                # LightRAG 可能使用 name 作为主要标识符
                node_query = """
                MATCH (n)
                WHERE any(key in keys(n) WHERE toLower(toString(n[key])) CONTAINS toLower($keyword))
                RETURN
                    coalesce(n.entity_id, n.id, n.name, toString(id(n))) as id,
                    coalesce(n.entity_name, n.name, n.entity_id, n.id) as entity_name,
                    coalesce(n.entity_type, head(labels(n)), 'Other') as entity_type,
                    coalesce(n.description, n.desc, '') as description
                LIMIT $limit
                """
                result = await session.run(node_query, keyword=keywords, limit=limit)
                async for record in result:
                    node_id = record['id']
                    if node_id and node_id not in node_ids:
                        node_ids.add(node_id)
                        nodes.append({
                            'id': node_id,
                            'entity_name': record['entity_name'] or node_id,
                            'entity_type': record['entity_type'] or 'Other',
                            'description': record['description'] or ''
                        })
                
                # Get edges between found nodes
                if node_ids:
                    edge_query = """
                    MATCH (a)-[r]->(b)
                    WHERE coalesce(a.id, a.entity_id, toString(id(a))) IN $node_ids
                      AND coalesce(b.id, b.entity_id, toString(id(b))) IN $node_ids
                    RETURN 
                        coalesce(a.id, a.entity_id, toString(id(a))) as source,
                        coalesce(b.id, b.entity_id, toString(id(b))) as target,
                        coalesce(r.keywords, type(r), 'related') as relation,
                        coalesce(r.description, '') as description
                    LIMIT $limit
                    """
                    edge_result = await session.run(edge_query, node_ids=list(node_ids), limit=limit*2)
                    async for record in edge_result:
                        edges.append({
                            'source_id': record['source'],
                            'target_id': record['target'],
                            'keywords': record['relation'],
                            'description': record['description']
                        })
        finally:
            await driver.close()
        
        return {"nodes": nodes, "edges": edges, "query": keywords}
    
    async def search_by_entity_type(self, entity_type: str, name_filter: str = "", limit: int = 50) -> Dict[str, Any]:
        """Search entities by type with optional name filter"""
        driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

        nodes = []
        edges = []
        node_ids = set()

        try:
            async with driver.session(database="neo4j") as session:
                # 使用更灵活的匹配条件，检查所有可能的属性
                if name_filter and name_filter.strip():
                    query = """
                    MATCH (n)
                    WHERE ($entity_type = ''
                           OR toLower(coalesce(n.entity_type, '')) = toLower($entity_type)
                           OR toLower(coalesce(head(labels(n)), '')) = toLower($entity_type))
                      AND any(key in keys(n) WHERE toLower(toString(n[key])) CONTAINS toLower($name_filter))
                    RETURN coalesce(n.entity_id, n.id, n.name, toString(id(n))) as id,
                           coalesce(n.entity_name, n.name, n.entity_id, n.id) as entity_name,
                           coalesce(n.entity_type, head(labels(n)), 'Other') as entity_type,
                           coalesce(n.description, n.desc, '') as description
                    LIMIT $limit
                    """
                else:
                    query = """
                    MATCH (n)
                    WHERE $entity_type = ''
                          OR toLower(coalesce(n.entity_type, '')) = toLower($entity_type)
                          OR toLower(coalesce(head(labels(n)), '')) = toLower($entity_type)
                    RETURN coalesce(n.entity_id, n.id, n.name, toString(id(n))) as id,
                           coalesce(n.entity_name, n.name, n.entity_id, n.id) as entity_name,
                           coalesce(n.entity_type, head(labels(n)), 'Other') as entity_type,
                           coalesce(n.description, n.desc, '') as description
                    LIMIT $limit
                    """

                result = await session.run(
                    query,
                    entity_type=entity_type or "",
                    name_filter=name_filter or "",
                    limit=limit
                )
                async for record in result:
                    node_id = record['id']
                    if node_id:
                        node_ids.add(node_id)
                        nodes.append({
                            'id': node_id,
                            'entity_name': record['entity_name'] or node_id,
                            'entity_type': record['entity_type'],
                            'description': record['description']
                        })

                # Get related edges
                if node_ids:
                    edge_query = """
                    MATCH (a)-[r]->(b)
                    WHERE coalesce(a.entity_id, a.id, a.name, toString(id(a))) IN $ids
                      AND coalesce(b.entity_id, b.id, b.name, toString(id(b))) IN $ids
                    RETURN coalesce(a.entity_id, a.id, a.name, toString(id(a))) as source,
                           coalesce(b.entity_id, b.id, b.name, toString(id(b))) as target,
                           coalesce(r.keywords, r.keyword, type(r)) as relation,
                           coalesce(r.description, r.desc, '') as description
                    LIMIT $limit
                    """
                    edge_result = await session.run(edge_query, ids=list(node_ids), limit=limit*2)
                    async for record in edge_result:
                        edges.append({
                            'source_id': record['source'],
                            'target_id': record['target'],
                            'keywords': record['relation'],
                            'description': record['description']
                        })
        finally:
            await driver.close()

        return {"nodes": nodes, "edges": edges, "filter": {"type": entity_type, "name": name_filter}}

    async def search_by_relation(self, relation_type: str = "", source_filter: str = "",
                                  target_filter: str = "", intermediate_filter: str = "", limit: int = 50) -> Dict[str, Any]:
        """
        Search relationships by type and source/target filters

        Args:
            relation_type: Relationship type to filter
            source_filter: Filter for source entity name
            target_filter: Filter for target entity name
            limit: Maximum results

        Returns:
            Dict with matching nodes and edges
        """
        driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

        nodes = []
        edges = []
        node_ids = set()
        edge_keys = set()  # 专门用于跟踪已添加的边

        try:
            async with driver.session(database="neo4j") as session:
                # 使用更灵活的匹配，检查所有可能的属性
                # query = """
                # MATCH (a)-[r]->(b)
                # WHERE ($rel_type = ''
                #        OR toLower(type(r)) = toLower($rel_type)
                #        OR toLower(coalesce(r.keywords, r.keyword, '')) CONTAINS toLower($rel_type))
                #   AND ($source = ''
                #        OR any(key in keys(a) WHERE toLower(toString(a[key])) CONTAINS toLower($source)))
                #   AND ($target = ''
                #        OR any(key in keys(b) WHERE toLower(toString(b[key])) CONTAINS toLower($target)))
                # RETURN
                #     coalesce(a.entity_id, a.id, a.name, toString(id(a))) as source_id,
                #     coalesce(a.entity_name, a.name, a.entity_id, a.id) as source_name,
                #     coalesce(a.entity_type, head(labels(a)), 'Other') as source_type,
                #     coalesce(b.entity_id, b.id, b.name, toString(id(b))) as target_id,
                #     coalesce(b.entity_name, b.name, b.entity_id, b.id) as target_name,
                #     coalesce(b.entity_type, head(labels(b)), 'Other') as target_type,
                #     coalesce(r.keywords, r.keyword, type(r), 'related') as relation,
                #     coalesce(r.description, r.desc, '') as rel_description,
                #     coalesce(a.description, a.desc, '') as source_desc,
                #     coalesce(b.description, b.desc, '') as target_desc
                # LIMIT $limit
                # """
                query = """
                // 1. 精准锁定源节点 a：只在名称相关属性中查找
                MATCH (a)
                WHERE ($source = '' 
                OR toLower(coalesce(a.entity_id, a.id, a.name, toString(id(a)))) CONTAINS toLower($source))
                
                // 2. 锁定目标节点 b (支持全属性模糊搜索，并确保不是同一个节点)
                MATCH (b)
                WHERE ($target = '' 
                OR toLower(coalesce(b.entity_id, b.id, b.name, toString(id(b)))) CONTAINS toLower($target))
                AND elementId(a) <> elementId(b)
                
                // 3. 寻找 1 到 5 步的路径 p，并过滤关系类型
                MATCH p = (a)-[*1..5]-(b)
                WHERE ($rel_type = '' 
                // 确保路径中的所有关系都符合 $rel_type 的筛选条件
                OR all(r in relationships(p) WHERE toLower(type(r)) = toLower($rel_type) 
                OR toLower(coalesce(r.keywords, r.keyword, '')) CONTAINS toLower($rel_type)))
                
                // 4. 返回结果
                RETURN
                // 源节点 (a)
                coalesce(a.entity_id, a.id, a.name, toString(id(a))) as source_id,
                coalesce(a.entity_name, a.name, a.entity_id, a.id) as source_name,
                coalesce(a.entity_type, head(labels(a)), 'Other') as source_type,
                coalesce(a.description, a.desc, '') as source_desc,
                
                // 目标节点 (b)
                coalesce(b.entity_id, b.id, b.name, toString(id(b))) as target_id,
                coalesce(b.entity_name, b.name, b.entity_id, b.id) as target_name,
                coalesce(b.entity_type, head(labels(b)), 'Other') as target_type,
                coalesce(b.description, b.desc, '') as target_desc,
                
                // --- 【新增部分】中间节点信息列表 ---
                // nodes(p)[1..-1] 取出路径中除去起点和终点的所有中间节点
                [n in nodes(p)[1..-1] | coalesce(n.entity_id, n.id, n.name, toString(id(n)))] as intermediate_id,
                [n in nodes(p)[1..-1] | coalesce(n.entity_name, n.name, n.entity_id, n.id)] as intermediate_names,
                [n in nodes(p)[1..-1] | coalesce(n.entity_type, head(labels(n)), 'Other')] as intermediate_types,
                [n in nodes(p)[1..-1] | coalesce(n.description, n.desc, '')] as intermediate_desc,
                
                // --- 路径关系列表 (保持不变) ---
                [r in relationships(p) | coalesce(r.keywords, r.keyword, type(r), 'related')] as relation,
                [r in relationships(p) | coalesce(r.description, r.desc, '')] as rel_description,
                
                // --- 【新增部分】路径长度 ---
                // 方便直观看到是几跳关系
                length(p) as path_length

                LIMIT $limit
                """


                result = await session.run(
                    query,
                    rel_type=relation_type or "",
                    source=source_filter or "",
                    target=target_filter or "",
                    intermediate=intermediate_filter or "",
                    limit=limit
                )
                async for record in result:
                    # Add source node
                    src_id = record['source_id']
                    if src_id and src_id not in node_ids:
                        node_ids.add(src_id)
                        nodes.append({
                            'id': src_id,
                            'entity_name': record['source_name'] or src_id,
                            'entity_type': record['source_type'],
                            'description': record['source_desc']
                        })

                    # Add target node
                    tgt_id = record['target_id']
                    if tgt_id and tgt_id not in node_ids:
                        node_ids.add(tgt_id)
                        nodes.append({
                            'id': tgt_id,
                            'entity_name': record['target_name'] or tgt_id,
                            'entity_type': record['target_type'],
                            'description': record['target_desc']
                        })

                    # 获取中间节点列表（intermediate_id 现在是列表类型）
                    intermediate_ids = record.get('intermediate_id', []) or []
                    intermediate_names = record.get('intermediate_names', []) or []
                    intermediate_types = record.get('intermediate_types', []) or []
                    intermediate_descs = record.get('intermediate_desc', []) or []

                    # 获取关系列表
                    relations = record.get('relation', []) or []
                    rel_descriptions = record.get('rel_description', []) or []

                    # 添加所有中间节点
                    for i, inter_id in enumerate(intermediate_ids):
                        if inter_id and inter_id not in node_ids:
                            node_ids.add(inter_id)
                            nodes.append({
                                'id': inter_id,
                                'entity_name': intermediate_names[i] if i < len(intermediate_names) else inter_id,
                                'entity_type': intermediate_types[i] if i < len(intermediate_types) else 'Other',
                                'description': intermediate_descs[i] if i < len(intermediate_descs) else ''
                            })

                    # 构建完整的路径节点序列: [源节点, 中间节点1, 中间节点2, ..., 目标节点]
                    path_nodes = [src_id] + list(intermediate_ids) + [tgt_id]

                    # 为路径中相邻的节点对创建边
                    for i in range(len(path_nodes) - 1):
                        from_id = path_nodes[i]
                        to_id = path_nodes[i + 1]

                        # 获取对应的关系信息
                        relation_keyword = relations[i] if i < len(relations) else 'related'
                        relation_desc = rel_descriptions[i] if i < len(rel_descriptions) else ''

                        # 避免重复添加相同的边
                        edge_key = f"{from_id}->{to_id}"
                        if edge_key not in edge_keys:  # 使用专门的 edge_keys 集合
                            edge_keys.add(edge_key)
                            edges.append({
                                'source_id': from_id,
                                'target_id': to_id,
                                'keywords': relation_keyword,
                                'description': relation_desc
                            })
        finally:
            await driver.close()

        return {
            "nodes": nodes,
            "edges": edges,
            "filter": {"relation": relation_type, "source": source_filter, "target": target_filter, "intermediate": intermediate_filter}
        }

        #         async for record in result:
        #             # Add source node
        #             src_id = record['source_id']
        #             if src_id and src_id not in node_ids:
        #                 node_ids.add(src_id)
        #                 nodes.append({
        #                     'id': src_id,
        #                     'entity_name': record['source_name'] or src_id,
        #                     'entity_type': record['source_type'],
        #                     'description': record['source_desc']
        #                 })
        #
        #             # Add target node
        #             tgt_id = record['target_id']
        #             if tgt_id and tgt_id not in node_ids:
        #                 node_ids.add(tgt_id)
        #                 nodes.append({
        #                     'id': tgt_id,
        #                     'entity_name': record['target_name'] or tgt_id,
        #                     'entity_type': record['target_type'],
        #                     'description': record['target_desc']
        #                 })
        #
        #             # Add edge
        #             edges.append({
        #                 'source_id': src_id,
        #                 'target_id': tgt_id,
        #                 'keywords': record['relation'],
        #                 'description': record['rel_description']
        #             })
        # finally:
        #     await driver.close()
        #
        # return {
        #     "nodes": nodes,
        #     "edges": edges,
        #     "filter": {"relation": relation_type, "source": source_filter, "target": target_filter}
        # }

    async def get_entity_types(self) -> List[str]:
        """Get all unique entity types in the graph"""
        driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

        types = []
        try:
            async with driver.session(database="neo4j") as session:
                result = await session.run("""
                    MATCH (n)
                    WHERE n.entity_type IS NOT NULL
                    RETURN DISTINCT n.entity_type as type
                    ORDER BY type
                """)
                async for record in result:
                    if record['type']:
                        types.append(record['type'])
        finally:
            await driver.close()

        return types

    async def get_relation_types(self) -> List[str]:
        """Get all unique relationship types in the graph"""
        driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

        types = []
        try:
            async with driver.session(database="neo4j") as session:
                result = await session.run("""
                    MATCH ()-[r]->()
                    RETURN DISTINCT type(r) as type
                    ORDER BY type
                """)
                async for record in result:
                    if record['type']:
                        types.append(record['type'])
        finally:
            await driver.close()

        return types

