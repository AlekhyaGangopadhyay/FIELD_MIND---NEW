"""
Expedition Knowledge Graph (EKG) - Graph Store
===============================================
NetworkX-backed property graph with JSON persistence.
Provides typed node/edge creation, label-based queries,
temporal window searches, and full serialisation.
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import networkx as nx


class MineKnowledgeGraph:
    """
    In-memory property graph for the underground mine.
    
    Nodes carry a mandatory 'label' attribute (e.g. 'TunnelSegment',
    'BlastEvent') plus arbitrary key-value properties.
    
    Edges carry a mandatory 'rel_type' attribute (e.g. 'CAUSED_BY')
    plus optional properties.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._label_index = defaultdict(set)  # label -> {node_id, ...}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, label: str, properties: dict = None):
        """Add a typed node to the graph."""
        props = dict(properties) if properties else {}
        props["label"] = label
        self.G.add_node(node_id, **props)
        self._label_index[label].add(node_id)

    def get_node(self, node_id: str) -> dict | None:
        """Return properties of a node, or None if it doesn't exist."""
        if node_id in self.G:
            return dict(self.G.nodes[node_id])
        return None

    def update_node(self, node_id: str, properties: dict):
        """Merge properties into an existing node."""
        if node_id in self.G:
            self.G.nodes[node_id].update(properties)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, from_id: str, to_id: str, rel_type: str, properties: dict = None):
        """Add a directed, typed edge."""
        props = dict(properties) if properties else {}
        props["rel_type"] = rel_type
        self.G.add_edge(from_id, to_id, **props)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query_by_label(self, label: str, filters: dict = None) -> list[dict]:
        """Return all nodes matching a label and optional property filters."""
        results = []
        for nid in self._label_index.get(label, set()):
            props = self.G.nodes[nid]
            if filters:
                if all(props.get(k) == v for k, v in filters.items()):
                    results.append({"node_id": nid, **dict(props)})
            else:
                results.append({"node_id": nid, **dict(props)})
        return results

    def query_neighbors(self, node_id: str, rel_type: str = None, depth: int = 1) -> list[dict]:
        """BFS traversal returning neighbours up to `depth` hops."""
        if node_id not in self.G:
            return []

        visited = set()
        frontier = [node_id]
        results = []

        for d in range(depth):
            next_frontier = []
            for nid in frontier:
                for successor in self.G.successors(nid):
                    edge_data = self.G.edges[nid, successor]
                    if rel_type and edge_data.get("rel_type") != rel_type:
                        continue
                    if successor not in visited:
                        visited.add(successor)
                        next_frontier.append(successor)
                        results.append({
                            "node_id": successor,
                            "depth": d + 1,
                            "via_rel": edge_data.get("rel_type"),
                            **dict(self.G.nodes[successor]),
                        })
                # Also check predecessors (incoming edges)
                for predecessor in self.G.predecessors(nid):
                    edge_data = self.G.edges[predecessor, nid]
                    if rel_type and edge_data.get("rel_type") != rel_type:
                        continue
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_frontier.append(predecessor)
                        results.append({
                            "node_id": predecessor,
                            "depth": d + 1,
                            "via_rel": edge_data.get("rel_type"),
                            **dict(self.G.nodes[predecessor]),
                        })
            frontier = next_frontier

        return results

    def query_temporal_window(self, label: str, start_ts: float, end_ts: float) -> list[dict]:
        """Return all nodes of a label whose 'timestamp' falls in [start_ts, end_ts]."""
        results = []
        for nid in self._label_index.get(label, set()):
            props = self.G.nodes[nid]
            ts = props.get("timestamp", 0.0)
            if start_ts <= ts <= end_ts:
                results.append({"node_id": nid, **dict(props)})
        return sorted(results, key=lambda x: x.get("timestamp", 0.0))

    def get_segment_history(self, segment_id: str) -> dict:
        """
        Return all events linked to a tunnel segment via OCCURRED_IN edges.
        Groups results by event label.
        """
        history = defaultdict(list)
        for predecessor in self.G.predecessors(segment_id):
            edge_data = self.G.edges[predecessor, segment_id]
            if edge_data.get("rel_type") == "OCCURRED_IN":
                node_data = dict(self.G.nodes[predecessor])
                label = node_data.get("label", "Unknown")
                history[label].append({"node_id": predecessor, **node_data})

        # Also include sensors via LOCATED_IN
        for predecessor in self.G.predecessors(segment_id):
            edge_data = self.G.edges[predecessor, segment_id]
            if edge_data.get("rel_type") == "LOCATED_IN":
                node_data = dict(self.G.nodes[predecessor])
                label = node_data.get("label", "Unknown")
                history[label].append({"node_id": predecessor, **node_data})

        return dict(history)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return node and edge counts by type."""
        node_counts = defaultdict(int)
        for nid in self.G.nodes:
            label = self.G.nodes[nid].get("label", "Unknown")
            node_counts[label] += 1

        edge_counts = defaultdict(int)
        for u, v in self.G.edges:
            rel = self.G.edges[u, v].get("rel_type", "Unknown")
            edge_counts[rel] += 1

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "nodes_by_label": dict(node_counts),
            "edges_by_type": dict(edge_counts),
        }

    # ------------------------------------------------------------------
    # Persistence (JSON)
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Serialise the entire graph to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "total_nodes": self.G.number_of_nodes(),
                "total_edges": self.G.number_of_edges(),
            },
            "nodes": [],
            "edges": [],
        }

        for nid in self.G.nodes:
            props = dict(self.G.nodes[nid])
            # Convert any non-serialisable values
            clean_props = {}
            for k, v in props.items():
                try:
                    json.dumps(v)
                    clean_props[k] = v
                except (TypeError, ValueError):
                    clean_props[k] = str(v)
            data["nodes"].append({"id": nid, "properties": clean_props})

        for u, v in self.G.edges:
            props = dict(self.G.edges[u, v])
            clean_props = {}
            for k, val in props.items():
                try:
                    json.dumps(val)
                    clean_props[k] = val
                except (TypeError, ValueError):
                    clean_props[k] = str(val)
            data["edges"].append({"from": u, "to": v, "properties": clean_props})

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[EKG] Graph saved to {filepath} ({self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges)")

    def load(self, filepath: str):
        """Deserialise the graph from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.G.clear()
        self._label_index.clear()

        for node_entry in data.get("nodes", []):
            nid = node_entry["id"]
            props = node_entry["properties"]
            label = props.get("label", "Unknown")
            self.G.add_node(nid, **props)
            self._label_index[label].add(nid)

        for edge_entry in data.get("edges", []):
            self.G.add_edge(edge_entry["from"], edge_entry["to"], **edge_entry["properties"])

        meta = data.get("metadata", {})
        print(f"[EKG] Graph loaded from {filepath} (saved: {meta.get('saved_at', '?')})")
        print(f"  Restored {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
