"""
Visualization Module
Creates interactive knowledge graph visualizations using pyvis
"""

from pyvis.network import Network
from typing import List, Dict, Any, Set
import tempfile


# Entity type to color mapping - 支持多种大小写格式
COLOR_MAP = {
    # 首字母大写
    "Person": "#FF6B6B",
    "Organization": "#4ECDC4",
    "Location": "#45B7D1",
    "Technology": "#FFA07A",
    "Concept": "#98D8C8",
    "Event": "#FFD93D",
    "Product": "#A8E6CF",
    "Other": "#CCCCCC",
    "Unknown": "#AAAAAA",
    # 全小写
    "person": "#FF6B6B",
    "organization": "#4ECDC4",
    "location": "#45B7D1",
    "technology": "#FFA07A",
    "concept": "#98D8C8",
    "event": "#FFD93D",
    "product": "#A8E6CF",
    "other": "#CCCCCC",
    "unknown": "#AAAAAA",
    # 全大写
    "PERSON": "#FF6B6B",
    "ORGANIZATION": "#4ECDC4",
    "LOCATION": "#45B7D1",
    "TECHNOLOGY": "#FFA07A",
    "CONCEPT": "#98D8C8",
    "EVENT": "#FFD93D",
    "PRODUCT": "#A8E6CF",
    "OTHER": "#CCCCCC",
    "UNKNOWN": "#AAAAAA",
}

# 标准化类型名称映射（用于图例显示）
TYPE_DISPLAY_NAMES = {
    "person": "人物 (Person)",
    "organization": "组织 (Organization)",
    "location": "地点 (Location)",
    "technology": "技术 (Technology)",
    "concept": "概念 (Concept)",
    "event": "事件 (Event)",
    "product": "产品 (Product)",
    "other": "其他 (Other)",
    "unknown": "未知 (Unknown)",
}


def get_color_for_type(entity_type: str) -> str:
    """获取实体类型对应的颜色"""
    if not entity_type:
        return COLOR_MAP.get("Other", "#CCCCCC")
    return COLOR_MAP.get(entity_type, COLOR_MAP.get(entity_type.lower(), "#CCCCCC"))


def normalize_type(entity_type: str) -> str:
    """标准化实体类型名称（转为小写）"""
    if not entity_type:
        return "other"
    return entity_type.lower()


def create_legend_html(used_types: Set[str]) -> str:
    """创建颜色图例 HTML"""
    if not used_types:
        return ""

    legend_items = []
    # 按标准化类型去重并排序
    normalized_types = sorted(set(normalize_type(t) for t in used_types))

    for norm_type in normalized_types:
        color = get_color_for_type(norm_type)
        display_name = TYPE_DISPLAY_NAMES.get(norm_type, norm_type.title())
        legend_items.append(f'''
            <div style="display: flex; align-items: center; margin: 4px 8px;">
                <div style="width: 16px; height: 16px; border-radius: 50%; background: {color}; margin-right: 8px; border: 2px solid rgba(255,255,255,0.3);"></div>
                <span style="color: #fff; font-size: 12px;">{display_name}</span>
            </div>
        ''')

    return f'''
    <div style="position: absolute; top: 10px; right: 10px; background: rgba(30,30,50,0.9); padding: 12px; border-radius: 8px; z-index: 1000; border: 1px solid rgba(255,255,255,0.2);">
        <div style="color: #fff; font-weight: bold; margin-bottom: 8px; font-size: 14px;">📊 实体类型图例</div>
        <div style="display: flex; flex-direction: column;">
            {"".join(legend_items)}
        </div>
    </div>
    '''


def select_core_nodes(nodes: List[Dict], edges: List[Dict], top_n_per_type: int = 3) -> Set[str]:
    """
    选择核心节点：每种实体类型中度数最高的 Top N 个节点

    Args:
        nodes: 所有节点列表
        edges: 所有边列表
        top_n_per_type: 每种类型显示的最大节点数

    Returns:
        核心节点 ID 集合
    """
    from collections import defaultdict

    # 计算每个节点的度数（入度 + 出度）
    degree_count = defaultdict(int)
    for edge in edges:
        source = str(edge.get("source_id") or edge.get("source") or "")
        target = str(edge.get("target_id") or edge.get("target") or "")
        if source:
            degree_count[source] += 1
        if target:
            degree_count[target] += 1

    # 按实体类型分组节点
    nodes_by_type = defaultdict(list)
    for node in nodes:
        node_id = str(node.get("id") or node.get("entity_id") or node.get("entity_name") or "")
        if not node_id:
            continue
        entity_type = node.get("entity_type") or "Other"
        # 标准化类型名称
        entity_type_lower = entity_type.lower()
        degree = degree_count.get(node_id, 0)
        nodes_by_type[entity_type_lower].append((node_id, degree))

    # 每种类型选择度数最高的 Top N 个节点
    core_node_ids = set()
    for entity_type, node_list in nodes_by_type.items():
        # 按度数降序排序
        sorted_nodes = sorted(node_list, key=lambda x: x[1], reverse=True)
        # 选择 Top N
        for node_id, degree in sorted_nodes[:top_n_per_type]:
            core_node_ids.add(node_id)

    # 如果核心节点太少，至少保留一些节点
    if len(core_node_ids) < 3 and nodes:
        # 添加度数最高的节点
        all_nodes_sorted = sorted(
            [(str(n.get("id") or ""), degree_count.get(str(n.get("id") or ""), 0)) for n in nodes if n.get("id")],
            key=lambda x: x[1],
            reverse=True
        )
        for node_id, _ in all_nodes_sorted[:5]:
            core_node_ids.add(node_id)

    return core_node_ids


def create_interactive_graph(nodes: List[Dict], edges: List[Dict],
                            max_nodes: int = 200, height: str = "600px",
                            top_n_per_type: int = 3) -> str:
    """
    Create interactive HTML graph visualization with color legend and click-to-expand

    采用"核心节点优先"策略：初始只显示每种类型中度数最高的 Top N 个节点，
    其他节点可通过点击展开查看。

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        max_nodes: Maximum nodes to display
        height: Height of visualization
        top_n_per_type: 每种实体类型初始显示的最大节点数（默认3）

    Returns:
        HTML string for embedding
    """
    # 过滤掉无效节点（ID 为 None 或空）
    valid_nodes = []
    for node in nodes:
        node_id = node.get("id") or node.get("entity_id") or node.get("entity_name")
        if node_id:
            node["id"] = str(node_id)  # 确保 ID 是字符串
            valid_nodes.append(node)

    nodes = valid_nodes

    if not nodes:
        return """<div style='padding: 40px; text-align: center; color: #888; background: #2a2a2a; border-radius: 8px;'>
            <h3>📭 无可视化数据</h3>
            <p>未找到有效的实体节点</p>
        </div>"""

    # Limit nodes for performance
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]

    # 创建有效节点 ID 集合
    node_ids = {node.get("id") for node in nodes if node.get("id")}

    # 过滤边，确保源和目标都存在且有效
    valid_edges = []
    for e in edges:
        source = e.get("source_id") or e.get("source")
        target = e.get("target_id") or e.get("target")
        if source and target:
            source = str(source)
            target = str(target)
            if source in node_ids and target in node_ids:
                e["source_id"] = source
                e["target_id"] = target
                valid_edges.append(e)

    edges = valid_edges

    # ========== 核心节点优先策略 ==========
    # 选择每种类型中度数最高的 Top N 个节点作为初始显示
    core_node_ids = select_core_nodes(nodes, edges, top_n_per_type)

    # 分离核心节点和隐藏节点
    initial_nodes = []  # 初始显示的节点
    hidden_nodes = []   # 隐藏的节点（可通过展开查看）

    for node in nodes:
        node_id = node.get("id")
        if node_id in core_node_ids:
            initial_nodes.append(node)
        else:
            hidden_nodes.append(node)

    # 筛选初始显示的边（两端都是核心节点的边）
    initial_edges = []
    hidden_edges = []

    for edge in edges:
        source = edge.get("source_id")
        target = edge.get("target_id")
        if source in core_node_ids and target in core_node_ids:
            initial_edges.append(edge)
        else:
            hidden_edges.append(edge)

    print(f"📊 核心节点优先策略: 初始显示 {len(initial_nodes)}/{len(nodes)} 个节点, {len(initial_edges)}/{len(edges)} 条边")

    # 收集使用的实体类型（用于图例）- 基于所有节点
    used_types = set()
    for node in nodes:
        entity_type = node.get("entity_type") or "Other"
        used_types.add(entity_type)

    # Create network
    net = Network(
        height=height,
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        notebook=False,
        cdn_resources='remote'  # 使用远程 CDN 资源
    )

    # Configure physics - 注意不要在这里设置 click 事件，稍后通过 JavaScript 添加
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75,
        "maxVelocity": 50,
        "solver": "barnesHut",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 100
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": {
          "enabled": true
        }
      },
      "nodes": {
        "font": {
          "size": 14,
          "color": "white"
        },
        "borderWidth": 2,
        "shadow": {
          "enabled": true,
          "size": 10
        }
      },
      "edges": {
        "font": {
          "size": 10,
          "align": "middle"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "smooth": {
          "enabled": true,
          "type": "continuous"
        },
        "shadow": {
          "enabled": false
        }
      }
    }
    """)
    
    # Add nodes - 只添加核心节点（初始显示）
    added_nodes = set()
    for node in initial_nodes:  # 只遍历核心节点
        node_id = node.get("id")
        if not node_id or node_id in added_nodes:
            continue

        node_id = str(node_id)  # 确保是字符串
        added_nodes.add(node_id)

        entity_name = node.get("entity_name") or node.get("name") or node_id
        entity_type = node.get("entity_type") or "Other"
        description = node.get("description") or "无描述"

        # 确保 entity_name 不为 None
        if not entity_name:
            entity_name = node_id

        # 使用新的颜色函数（支持多种大小写格式）
        color = get_color_for_type(entity_type)

        # Calculate node size based on name length (simple heuristic)
        size = min(20 + len(str(entity_name)) * 2, 45)

        # 安全处理描述
        desc_text = str(description)[:100] if description else ""
        title = f"<b>{entity_name}</b><br>类型: {entity_type}<br>{desc_text}<br><i>🔍 单击展开关联节点 | 双击收起</i>"

        try:
            net.add_node(
                node_id,
                label=str(entity_name)[:30],  # 限制标签长度
                title=title,
                color=color,
                size=size,
                shape="dot"
            )
        except Exception as e:
            print(f"添加节点失败 {node_id}: {e}")
            continue

    # Add edges - 只添加核心节点之间的边（初始显示）
    for edge in initial_edges:  # 只遍历核心边
        source = edge.get("source_id") or edge.get("source")
        target = edge.get("target_id") or edge.get("target")

        if not source or not target:
            continue

        source = str(source)
        target = str(target)

        # 确保两端节点都已添加
        if source not in added_nodes or target not in added_nodes:
            continue

        keywords = edge.get("keywords") or edge.get("relation") or "related"
        description = edge.get("description") or ""

        # 安全处理
        keywords = str(keywords)[:30]
        desc_text = str(description)[:80] if description else ""
        title = f"{keywords}<br>{desc_text}"

        try:
            net.add_edge(
                source,
                target,
                title=title,
                label=keywords[:15],
                color="#888888",
                width=2
            )
        except Exception as e:
            print(f"添加边失败 {source}->{target}: {e}")
            continue

    # 检查是否有节点被添加
    if not added_nodes:
        return """<div style='padding: 40px; text-align: center; color: #888; background: #2a2a2e; border-radius: 8px;'>
            <h3>📭 无有效节点</h3>
            <p>所有节点 ID 均无效</p>
        </div>"""

    # 生成颜色图例 HTML
    legend_html = create_legend_html(used_types)

    # 构建完整的边数据用于点击展开功能
    import json
    all_edges_json = json.dumps(edges, ensure_ascii=False)
    all_nodes_json = json.dumps(nodes, ensure_ascii=False)

    # Generate HTML - 使用临时文件方式（pyvis 标准方法）
    try:
        import os
        import html as html_module

        # 创建临时文件保存图谱
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8')
        temp_path = temp_file.name
        temp_file.close()

        # 保存图谱到临时文件
        net.save_graph(temp_path)

        # 读取生成的 HTML
        with open(temp_path, 'r', encoding='utf-8') as html_file:
            raw_html = html_file.read()

        # 清理临时文件
        try:
            os.unlink(temp_path)
        except:
            pass

        # 验证 HTML 有效性
        if raw_html and len(raw_html) > 100 and '<script' in raw_html:
            # 注入点击展开功能的 JavaScript 代码
            # 注意：不能使用 f-string，因为 JSON 字符串中的花括号会被误解析
            click_expand_script = '''
            <script type="text/javascript">
            // 存储完整的边和节点数据
            var allEdgesData = ''' + all_edges_json + ''';
            var allNodesData = ''' + all_nodes_json + ''';
            var expandedNodes = new Set();
            var hiddenNodes = new Set();
            var hiddenEdges = new Set();

            // 节点颜色映射
            var colorMap = {
                "person": "#FF6B6B",
                "organization": "#4ECDC4",
                "location": "#45B7D1",
                "technology": "#FFA07A",
                "concept": "#98D8C8",
                "event": "#FFD93D",
                "product": "#A8E6CF",
                "other": "#CCCCCC",
                "unknown": "#AAAAAA"
            };

            function getColorForType(entityType) {
                if (!entityType) return "#CCCCCC";
                return colorMap[entityType.toLowerCase()] || "#CCCCCC";
            }

            // 存储 vis.js DataSet 引用
            var nodesDataSet = null;
            var edgesDataSet = null;
            // 记录初始显示的节点和边（用于收起时恢复）
            var initialNodeIds = new Set();
            var initialEdgeKeys = new Set();
            // 记录每个展开操作添加的节点和边
            var addedByExpand = new Map(); // nodeId -> {nodes: [], edges: []}

            // 初始化函数：获取 DataSet 引用并绑定事件
            function initExpandFeature() {
                if (typeof network === 'undefined') {
                    return false;
                }

                // 方法1: 直接使用全局变量 nodes/edges (pyvis 创建的)
                if (typeof nodes !== 'undefined' && nodes.add) {
                    nodesDataSet = nodes;
                }
                if (typeof edges !== 'undefined' && edges.add) {
                    edgesDataSet = edges;
                }

                // 方法2: 从 network.body.data 获取
                if (!nodesDataSet && network.body && network.body.data && network.body.data.nodes) {
                    nodesDataSet = network.body.data.nodes;
                }
                if (!edgesDataSet && network.body && network.body.data && network.body.data.edges) {
                    edgesDataSet = network.body.data.edges;
                }

                if (!nodesDataSet || !edgesDataSet) {
                    console.log("⚠️ 无法获取 DataSet 引用");
                    return false;
                }

                // 记录初始节点和边
                nodesDataSet.get().forEach(function(node) {
                    initialNodeIds.add(node.id);
                });
                edgesDataSet.get().forEach(function(edge) {
                    initialEdgeKeys.add(edge.from + "->" + edge.to);
                });
                console.log("📊 初始图谱: " + initialNodeIds.size + " 个节点, " + initialEdgeKeys.size + " 条边");

                // 绑定单击事件 - 展开节点
                network.on("click", function(params) {
                    if (params.nodes.length > 0) {
                        var clickedNodeId = params.nodes[0];
                        expandNode(clickedNodeId);
                    }
                });

                // 绑定双击事件 - 收起节点
                network.on("doubleClick", function(params) {
                    if (params.nodes.length > 0) {
                        var clickedNodeId = params.nodes[0];
                        collapseNode(clickedNodeId);
                    }
                });

                console.log("✅ 点击展开/双击收起功能已启用");
                return true;
            }

            // 多次尝试初始化
            var initAttempts = 0;
            function tryInit() {
                if (initExpandFeature()) return;
                initAttempts++;
                if (initAttempts < 15) {
                    setTimeout(tryInit, 300);
                }
            }
            setTimeout(tryInit, 500);

            // 收起节点功能
            function collapseNode(nodeId) {
                if (!expandedNodes.has(nodeId)) {
                    console.log("节点未展开，无需收起: " + nodeId);
                    return;
                }

                // 获取该节点展开时添加的内容
                var addedContent = addedByExpand.get(nodeId);
                if (!addedContent) {
                    console.log("无记录的展开内容: " + nodeId);
                    expandedNodes.delete(nodeId);
                    return;
                }

                expandedNodes.delete(nodeId);
                addedByExpand.delete(nodeId);

                // 检查哪些节点/边仍被其他展开节点使用
                var stillNeededNodes = new Set();
                var stillNeededEdges = new Set();
                addedByExpand.forEach(function(content, expNodeId) {
                    content.nodes.forEach(function(nid) { stillNeededNodes.add(nid); });
                    content.edges.forEach(function(eid) { stillNeededEdges.add(eid); });
                });

                // 移除不再需要的节点和边
                var nodesToRemove = addedContent.nodes.filter(function(nid) {
                    return !stillNeededNodes.has(nid) && !initialNodeIds.has(nid);
                });
                var edgesToRemove = addedContent.edges.filter(function(eid) {
                    return !stillNeededEdges.has(eid);
                });

                if (edgesToRemove.length > 0) edgesDataSet.remove(edgesToRemove);
                if (nodesToRemove.length > 0) nodesDataSet.remove(nodesToRemove);

                console.log("已收起 " + nodeId + ": 移除 " + nodesToRemove.length + " 个节点, " + edgesToRemove.length + " 条边");
            }

            // 展开节点功能
            function expandNode(nodeId) {
                if (expandedNodes.has(nodeId)) {
                    console.log("节点已展开，双击可收起: " + nodeId);
                    return;
                }
                expandedNodes.add(nodeId);

                var relatedEdges = [];
                var relatedNodeIds = new Set();

                // 查找所有与该节点相关的边
                allEdgesData.forEach(function(edge) {
                    var source = edge.source_id || edge.source;
                    var target = edge.target_id || edge.target;
                    if (source === nodeId || target === nodeId) {
                        relatedEdges.push(edge);
                        if (source !== nodeId) relatedNodeIds.add(source);
                        if (target !== nodeId) relatedNodeIds.add(target);
                    }
                });

                console.log("展开节点 " + nodeId + ": 找到 " + relatedEdges.length + " 条边, " + relatedNodeIds.size + " 个关联节点");

                // 使用 DataSet.get() 获取当前已有的节点ID
                var existingNodeIds = new Set();
                nodesDataSet.get().forEach(function(node) {
                    existingNodeIds.add(node.id);
                });

                // 获取已有的边（用于去重）
                var existingEdgeKeys = new Set();
                edgesDataSet.get().forEach(function(edge) {
                    existingEdgeKeys.add(edge.from + "->" + edge.to);
                });

                // 添加新节点
                var newNodes = [];
                var addedNodeIds = [];
                relatedNodeIds.forEach(function(relNodeId) {
                    if (!existingNodeIds.has(relNodeId)) {
                        var nodeInfo = allNodesData.find(function(n) {
                            return (n.id || n.entity_id || n.entity_name) === relNodeId;
                        });

                        if (nodeInfo) {
                            var entityType = nodeInfo.entity_type || "Other";
                            var entityName = nodeInfo.entity_name || nodeInfo.name || String(relNodeId);
                            var description = nodeInfo.description || "";

                            newNodes.push({
                                id: relNodeId,
                                label: entityName.substring(0, 30),
                                title: "<b>" + entityName + "</b><br>类型: " + entityType + "<br>" + description.substring(0, 100) + "<br><i>单击展开，双击收起</i>",
                                color: getColorForType(entityType),
                                size: Math.min(20 + entityName.length * 2, 45),
                                shape: "dot"
                            });
                        } else {
                            newNodes.push({
                                id: relNodeId,
                                label: String(relNodeId).substring(0, 30),
                                title: String(relNodeId),
                                color: "#CCCCCC",
                                size: 25,
                                shape: "dot"
                            });
                        }
                        existingNodeIds.add(relNodeId);
                        addedNodeIds.push(relNodeId);
                    }
                });

                // 添加新边（为边生成唯一ID）
                var newEdges = [];
                var addedEdgeIds = [];
                var edgeCounter = Date.now();
                relatedEdges.forEach(function(edge) {
                    var source = edge.source_id || edge.source;
                    var target = edge.target_id || edge.target;
                    var edgeKey = source + "->" + target;

                    if (!existingEdgeKeys.has(edgeKey) && existingNodeIds.has(source) && existingNodeIds.has(target)) {
                        var keywords = edge.keywords || edge.relation || "related";
                        var edgeId = "edge_" + edgeCounter++;
                        newEdges.push({
                            id: edgeId,
                            from: source,
                            to: target,
                            label: String(keywords).substring(0, 15),
                            title: String(keywords),
                            color: "#888888",
                            width: 2,
                            arrows: "to"
                        });
                        existingEdgeKeys.add(edgeKey);
                        addedEdgeIds.push(edgeId);
                    }
                });

                // 使用 DataSet.add() 更新图谱
                if (newNodes.length > 0) nodesDataSet.add(newNodes);
                if (newEdges.length > 0) edgesDataSet.add(newEdges);

                if (newNodes.length === 0 && newEdges.length === 0) {
                    // 没有添加任何新内容，不标记为已展开
                    console.log("⚠️ 该节点的所有关联节点都已在图中显示，无新内容可展开");
                    expandedNodes.delete(nodeId);  // 移除展开标记
                } else {
                    // 记录此次展开添加的内容（用于收起时移除）
                    addedByExpand.set(nodeId, {
                        nodes: addedNodeIds,
                        edges: addedEdgeIds
                    });
                    console.log("已展开: 添加 " + newNodes.length + " 个节点, " + newEdges.length + " 条边");
                }
            }
            </script>
            '''

            # 在 </body> 前注入脚本
            if '</body>' in raw_html:
                raw_html = raw_html.replace('</body>', click_expand_script + '</body>')

            # 在 <body> 后注入图例
            legend_div = f'''
            <div id="legend-container" style="position: fixed; top: 10px; right: 10px; background: rgba(30,30,50,0.95); padding: 12px; border-radius: 8px; z-index: 9999; border: 1px solid rgba(255,255,255,0.2); max-width: 200px;">
                <div style="color: #fff; font-weight: bold; margin-bottom: 8px; font-size: 13px;">📊 实体类型图例</div>
                <div style="display: flex; flex-direction: column; font-size: 11px;">
            '''
            for norm_type in sorted(set(normalize_type(t) for t in used_types)):
                color = get_color_for_type(norm_type)
                display_name = TYPE_DISPLAY_NAMES.get(norm_type, norm_type.title())
                legend_div += f'''
                    <div style="display: flex; align-items: center; margin: 3px 0;">
                        <div style="width: 14px; height: 14px; border-radius: 50%; background: {color}; margin-right: 8px; border: 1px solid rgba(255,255,255,0.3); flex-shrink: 0;"></div>
                        <span style="color: #fff;">{display_name}</span>
                    </div>
                '''
            legend_div += '''
                </div>
                <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.2); font-size: 10px; color: #aaa;">
                    💡 点击节点展开关联实体
                </div>
            </div>
            '''

            if '<body>' in raw_html:
                raw_html = raw_html.replace('<body>', '<body>' + legend_div)

            # 使用 iframe + srcdoc 方式嵌入完整 HTML 文档
            escaped_html = html_module.escape(raw_html)
            iframe_html = f'''<iframe
                srcdoc="{escaped_html}"
                style="width: 100%; height: 700px; border: none; border-radius: 8px;"
                sandbox="allow-scripts allow-same-origin"
                loading="lazy"
            ></iframe>'''
            return iframe_html
        else:
            # 回退到简单 HTML
            return create_simple_graph_html(nodes, edges, used_types)

    except Exception as e:
        print(f"生成图谱 HTML 失败: {e}")
        import traceback
        traceback.print_exc()
        # 回退到简单 HTML
        return create_simple_graph_html(nodes, edges, used_types)


def create_simple_graph_html(nodes: List[Dict], edges: List[Dict], used_types: Set[str] = None) -> str:
    """
    Create a simple static graph visualization (fallback)

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        used_types: Set of entity types used (for legend)

    Returns:
        Simple HTML representation
    """
    # 生成图例样式
    legend_style = ""
    if used_types:
        legend_items = []
        for norm_type in sorted(set(normalize_type(t) for t in used_types)):
            color = get_color_for_type(norm_type)
            display_name = TYPE_DISPLAY_NAMES.get(norm_type, norm_type.title())
            legend_items.append(f'''
                <div style="display: flex; align-items: center; margin: 4px 0;">
                    <div style="width: 16px; height: 16px; border-radius: 50%; background: {color}; margin-right: 8px;"></div>
                    <span>{display_name}</span>
                </div>
            ''')
        legend_style = f'''
            <div style="position: fixed; top: 10px; right: 10px; background: rgba(50,50,70,0.95); padding: 12px; border-radius: 8px; border: 1px solid #555;">
                <div style="font-weight: bold; margin-bottom: 8px;">📊 实体类型图例</div>
                {"".join(legend_items)}
            </div>
        '''

    html = f"""
    <html>
    <head>
        <style>
            body {{ background: #222; color: #fff; font-family: Arial; padding: 20px; }}
            .node {{ margin: 10px; padding: 10px; background: #333; border-radius: 5px; border-left: 4px solid #ccc; }}
            .edge {{ margin: 5px 20px; padding: 5px; background: #444; border-left: 3px solid #4ECDC4; }}
        </style>
    </head>
    <body>
        {legend_style}
        <h2>Knowledge Graph</h2>
        <h3>Entities ({len(nodes)} nodes)</h3>
    """

    for node in nodes[:50]:  # Limit display
        entity_type = node.get('entity_type', 'Other')
        color = get_color_for_type(entity_type)
        html += f"""
        <div class="node" style="border-left-color: {color};">
            <b>{node.get('entity_name', 'Unknown')}</b>
            <span style="color: {color};">({entity_type})</span><br>
            <small>{node.get('description', 'No description')[:100]}</small>
        </div>
        """

    html += f"<h3>Relationships ({len(edges)} edges)</h3>"

    for edge in edges[:50]:  # Limit display
        html += f"""
        <div class="edge">
            {edge.get('source_id', '?')} → {edge.get('target_id', '?')}:
            {edge.get('keywords', 'related')}
        </div>
        """

    html += "</body></html>"
    return html

