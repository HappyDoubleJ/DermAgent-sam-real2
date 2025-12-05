"""
Ontology Utility Functions for Derm1M Dataset

온톨로지 트리 구조를 활용한 유틸리티 함수들:
- 경로 추출
- 거리 계산
- LCA (Lowest Common Ancestor) 찾기
- 노드 유효성 검증
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import deque


def get_default_ontology_path() -> Optional[str]:
    """
    온톨로지 파일의 기본 경로를 찾습니다.
    여러 가능한 위치를 시도합니다.

    Returns:
        ontology.json 파일의 절대 경로, 찾지 못하면 None
    """
    # 1. project_path.py 사용 시도
    try:
        # 프로젝트 루트 찾기 (2단계 상위 디렉토리)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent

        # project_path.py 임포트 시도
        sys.path.insert(0, str(project_root))
        import project_path
        sys.path.pop(0)

        if hasattr(project_path, 'ONTOLOGY_PATH'):
            if os.path.exists(project_path.ONTOLOGY_PATH):
                return project_path.ONTOLOGY_PATH
    except Exception:
        pass

    # 2. 상대 경로로 찾기
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        ontology_path = project_root / "dataset" / "Derm1M" / "ontology.json"
        if ontology_path.exists():
            return str(ontology_path)
    except Exception:
        pass

    # 3. 현재 디렉토리 기준
    possible_paths = [
        "ontology.json",
        "../../../dataset/Derm1M/ontology.json",
        "../../dataset/Derm1M/ontology.json",
        "./dataset/Derm1M/ontology.json",
    ]

    for path in possible_paths:
        abs_path = Path(path).resolve()
        if abs_path.exists():
            return str(abs_path)

    return None


class OntologyTree:
    """온톨로지 트리 구조를 관리하는 클래스"""

    def __init__(self, ontology_path: Optional[str] = None):
        """
        Args:
            ontology_path: ontology.json 파일 경로
                          None이면 자동으로 찾음
        """
        if ontology_path is None:
            ontology_path = get_default_ontology_path()
            if ontology_path is None:
                raise FileNotFoundError(
                    "ontology.json 파일을 찾을 수 없습니다. "
                    "경로를 명시적으로 지정하거나 프로젝트 구조를 확인하세요."
                )

        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)

        self.ontology_path = ontology_path
        
        # 부모 노드 매핑 구축 (자식 -> 부모)
        self.parent_map: Dict[str, str] = {}
        self._build_parent_map()
        
        # 모든 유효 노드 집합 (root 제외)
        self.valid_nodes: Set[str] = set(self.ontology.keys()) - {'root'}
        # 평가 시 특수 케이스로 허용하는 라벨 추가
        self.valid_nodes.add("no definitive diagnosis")
        
        # 정규화된 노드 이름 매핑 (소문자 -> 원본)
        self.normalized_map: Dict[str, str] = {}
        self._build_normalized_map()
        
        # 각 노드의 깊이 캐싱
        self._depth_cache: Dict[str, int] = {}
        
        # 각 노드의 경로 캐싱
        self._path_cache: Dict[str, List[str]] = {}
    
    def _build_parent_map(self):
        """자식 -> 부모 매핑 구축"""
        for parent, children in self.ontology.items():
            for child in children:
                self.parent_map[child] = parent
    
    def _build_normalized_map(self):
        """정규화된 노드 이름 매핑 구축"""
        for node in self.valid_nodes:
            normalized = self.normalize_label(node)
            self.normalized_map[normalized] = node
            
            # 쉼표로 구분된 별칭도 추가
            if ',' in node:
                parts = [p.strip() for p in node.split(',')]
                for part in parts:
                    norm_part = self.normalize_label(part)
                    if norm_part not in self.normalized_map:
                        self.normalized_map[norm_part] = node
    
    @staticmethod
    def normalize_label(label: str) -> str:
        """라벨 정규화 (소문자, 공백/구두점 정리)"""
        text = str(label).lower().strip()
        text = text.replace('"', '').replace("'", "")  # 따옴표 제거
        text = text.rstrip(" .,:;")                    # 흔한 말미 구두점 제거
        text = re.sub(r"\s+", " ", text)               # 공백 축소
        return text
    
    def is_valid_node(self, node: str) -> bool:
        """노드가 온톨로지에 존재하는지 확인"""
        if node in self.valid_nodes:
            return True
        # 정규화된 이름으로도 확인
        normalized = self.normalize_label(node)
        return normalized in self.normalized_map
    
    def get_canonical_name(self, node: str) -> Optional[str]:
        """노드의 정식 이름 반환 (정규화된 입력도 처리)"""
        if node in self.valid_nodes:
            return node
        normalized = self.normalize_label(node)
        return self.normalized_map.get(normalized)
    
    def get_path_to_root(self, node: str) -> List[str]:
        """
        노드에서 루트까지의 경로 반환 (노드 포함, 루트 포함)
        
        예: "Tinea corporis" -> ["Tinea corporis", "fungal", "infectious", "inflammatory", "root"]
        """
        canonical = self.get_canonical_name(node)
        if canonical is None:
            return []
        
        if canonical in self._path_cache:
            return self._path_cache[canonical].copy()
        
        path = [canonical]
        current = canonical
        
        while current in self.parent_map:
            parent = self.parent_map[current]
            path.append(parent)
            current = parent
        
        self._path_cache[canonical] = path
        return path.copy()
    
    def get_depth(self, node: str) -> int:
        """
        노드의 깊이 반환 (root = 0, root의 자식 = 1, ...)
        유효하지 않은 노드는 -1 반환
        """
        canonical = self.get_canonical_name(node)
        if canonical is None:
            return -1
        
        if canonical in self._depth_cache:
            return self._depth_cache[canonical]
        
        path = self.get_path_to_root(canonical)
        depth = len(path) - 1  # root 제외
        
        self._depth_cache[canonical] = depth
        return depth
    
    def get_ancestors(self, node: str) -> Set[str]:
        """노드의 모든 조상 반환 (자기 자신 제외)"""
        path = self.get_path_to_root(node)
        if len(path) <= 1:
            return set()
        return set(path[1:])
    
    def get_lca(self, node1: str, node2: str) -> Optional[str]:
        """
        두 노드의 최소 공통 조상 (Lowest Common Ancestor) 반환
        """
        path1 = self.get_path_to_root(node1)
        path2 = self.get_path_to_root(node2)
        
        if not path1 or not path2:
            return None
        
        # path1의 노드들을 집합으로 변환
        ancestors1 = set(path1)
        
        # path2를 따라가면서 첫 번째로 공통되는 조상 찾기
        for ancestor in path2:
            if ancestor in ancestors1:
                return ancestor
        
        return None
    
    def get_hierarchical_distance(self, node1: str, node2: str) -> int:
        """
        두 노드 간의 계층적 거리 계산
        
        거리 = (node1에서 LCA까지 거리) + (node2에서 LCA까지 거리)
        
        유효하지 않은 노드가 있으면 -1 반환
        """
        path1 = self.get_path_to_root(node1)
        path2 = self.get_path_to_root(node2)
        
        if not path1 or not path2:
            return -1
        
        lca = self.get_lca(node1, node2)
        if lca is None:
            return -1
        
        # 각 노드에서 LCA까지의 거리
        dist1 = path1.index(lca)
        dist2 = path2.index(lca)
        
        return dist1 + dist2
    
    def get_level_labels(self, node: str) -> Dict[int, str]:
        """
        노드의 각 레벨별 라벨 반환
        
        Returns:
            {1: "inflammatory", 2: "infectious", 3: "fungal", 4: "Tinea corporis"}
        """
        path = self.get_path_to_root(node)
        if not path:
            return {}
        
        # path는 [node, parent, grandparent, ..., root] 형태
        # 레벨은 root부터 1로 시작
        result = {}
        path_reversed = list(reversed(path))
        
        for i, label in enumerate(path_reversed):
            if label != 'root':
                result[i] = label  # root=0, level1=1, ...
        
        return result
    
    def get_children(self, node: str) -> List[str]:
        """노드의 직계 자식 노드들 반환"""
        canonical = self.get_canonical_name(node)
        if canonical is None:
            return []
        return self.ontology.get(canonical, [])
    
    def get_all_descendants(self, node: str) -> Set[str]:
        """노드의 모든 자손 노드들 반환 (BFS)"""
        canonical = self.get_canonical_name(node)
        if canonical is None:
            return set()
        
        descendants = set()
        queue = deque([canonical])
        
        while queue:
            current = queue.popleft()
            children = self.ontology.get(current, [])
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
        
        return descendants
    
    def get_siblings(self, node: str) -> List[str]:
        """노드의 형제 노드들 반환 (자기 자신 제외)"""
        canonical = self.get_canonical_name(node)
        if canonical is None:
            return []
        
        parent = self.parent_map.get(canonical)
        if parent is None:
            return []
        
        siblings = self.ontology.get(parent, [])
        return [s for s in siblings if s != canonical]
    
    def get_max_depth(self) -> int:
        """트리의 최대 깊이 반환"""
        max_depth = 0
        for node in self.valid_nodes:
            depth = self.get_depth(node)
            max_depth = max(max_depth, depth)
        return max_depth
    
    def get_nodes_at_depth(self, depth: int) -> List[str]:
        """특정 깊이의 모든 노드 반환"""
        return [node for node in self.valid_nodes if self.get_depth(node) == depth]
    
    def filter_valid_labels(self, labels: List[str]) -> List[str]:
        """유효한 라벨만 필터링하여 반환"""
        valid = []
        for label in labels:
            if label is None:
                continue
            cleaned = str(label).strip()
            if not cleaned:
                continue
            # 흔한 구두점/공백 잡음 제거
            cleaned = cleaned.rstrip(" .,:;")

            # Special case: "no definitive diagnosis" is valid without ontology check
            if cleaned.lower() == "no definitive diagnosis":
                valid.append("no definitive diagnosis")
                continue

            canonical = self.get_canonical_name(cleaned)
            if canonical is not None:
                valid.append(canonical)
        return valid
    
    def print_tree(self, node: str = 'root', indent: int = 0, max_depth: int = 3):
        """트리 구조 출력 (디버깅용)"""
        if indent > max_depth * 2:
            return
        
        prefix = "  " * indent
        print(f"{prefix}{node}")
        
        children = self.ontology.get(node, [])
        for child in children:
            self.print_tree(child, indent + 1, max_depth)
    
    def get_stats(self) -> Dict:
        """온톨로지 통계 반환"""
        depths = [self.get_depth(node) for node in self.valid_nodes]
        leaf_nodes = [node for node in self.valid_nodes if not self.ontology.get(node)]
        
        return {
            'total_nodes': len(self.valid_nodes),
            'max_depth': max(depths) if depths else 0,
            'leaf_nodes': len(leaf_nodes),
            'internal_nodes': len(self.valid_nodes) - len(leaf_nodes),
            'root_children': len(self.ontology.get('root', [])),
        }


def demo():
    """데모 함수"""
    try:
        # 자동으로 ontology.json 찾기
        tree = OntologyTree()
        print(f"Loaded ontology from: {tree.ontology_path}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print("=== Ontology Statistics ===")
    stats = tree.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Example: Tinea corporis ===")
    node = "Tinea corporis"
    print(f"  Valid: {tree.is_valid_node(node)}")
    print(f"  Depth: {tree.get_depth(node)}")
    print(f"  Path to root: {tree.get_path_to_root(node)}")
    print(f"  Level labels: {tree.get_level_labels(node)}")
    print(f"  Siblings: {tree.get_siblings(node)}")
    
    print("\n=== Example: Distance between Tinea corporis and Tinea pedis ===")
    node1, node2 = "Tinea corporis", "Tinea pedis"
    print(f"  LCA: {tree.get_lca(node1, node2)}")
    print(f"  Distance: {tree.get_hierarchical_distance(node1, node2)}")
    
    print("\n=== Example: Distance between Tinea corporis and Psoriasis ===")
    node1, node2 = "Tinea corporis", "Psoriasis"
    print(f"  LCA: {tree.get_lca(node1, node2)}")
    print(f"  Distance: {tree.get_hierarchical_distance(node1, node2)}")
    
    print("\n=== Normalized label lookup ===")
    test_labels = ["tinea corporis", "PSORIASIS", "eczema", "invalid_disease"]
    for label in test_labels:
        canonical = tree.get_canonical_name(label)
        print(f"  '{label}' -> '{canonical}'")


if __name__ == "__main__":
    demo()
