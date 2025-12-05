"""
Hierarchical Evaluation Metrics for Derm1M Dataset

온톨로지 트리 구조를 활용한 평가 메트릭:
- Exact Match
- Hierarchical Distance Score
- Level-wise Accuracy
- Partial Credit Score
- Ancestor Match Score
"""

import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from ontology_utils import OntologyTree


@dataclass
class PredictionResult:
    """단일 예측 결과"""
    ground_truth: List[str]  # 정답 라벨들 (온톨로지에 있는 것만)
    prediction: List[str]    # 예측 라벨들 (온톨로지에 있는 것만)
    
    # 원본 (필터링 전)
    raw_ground_truth: List[str] = field(default_factory=list)
    raw_prediction: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """전체 평가 결과"""
    # 기본 메트릭
    exact_match: float = 0.0
    partial_match: float = 0.0

    # 계층적 메트릭
    hierarchical_precision: float = 0.0
    hierarchical_recall: float = 0.0
    hierarchical_f1: float = 0.0
    avg_hierarchical_distance: float = 0.0

    # 레벨별 정확도
    level_accuracy: Dict[int, float] = field(default_factory=dict)

    # 부분 점수
    avg_partial_credit: float = 0.0

    # Top-K 정확도 (다중 예측 지원)
    top_k_accuracy: Dict[int, float] = field(default_factory=dict)  # {k: accuracy}
    top_k_hierarchical_f1: Dict[int, float] = field(default_factory=dict)  # {k: h_f1}

    # 상세 통계
    total_samples: int = 0
    valid_samples: int = 0  # GT와 Pred 모두 유효한 샘플 수
    skipped_samples: int = 0

    # 추가 정보
    details: Dict[str, Any] = field(default_factory=dict)


class HierarchicalEvaluator:
    """계층적 평가를 수행하는 클래스"""

    def __init__(self, ontology_path: Optional[str] = None):
        """
        Args:
            ontology_path: ontology.json 파일 경로
                          None이면 자동으로 찾음
        """
        self.tree = OntologyTree(ontology_path)
        self.max_depth = self.tree.get_max_depth()
    
    def preprocess_labels(self, labels: List[str]) -> List[str]:
        """라벨 전처리: 유효한 라벨만 필터링하고 정규화"""
        # 공백/대소문자/구두점에 강건하도록 정리 후 필터링
        cleaned = []
        for label in labels:
            if label is None:
                continue
            text = str(label).strip().rstrip(" .,:;")
            if not text:
                continue
            # 쉼표로 구분된 멀티 라벨 분해
            parts = [p.strip() for p in text.split(",") if p.strip()]
            cleaned.extend(parts if parts else [text])
        return self.tree.filter_valid_labels(cleaned)
    
    # ============ 기본 메트릭 ============
    
    def exact_match(self, gt_labels: List[str], pred_labels: List[str]) -> bool:
        """
        정확히 일치하는지 확인 (순서 무관)
        
        최소 하나의 GT 라벨이 예측에 포함되어 있으면 True
        """
        gt_set = set(gt_labels)
        pred_set = set(pred_labels)
        return len(gt_set & pred_set) > 0
    
    def partial_match_ratio(self, gt_labels: List[str], pred_labels: List[str]) -> float:
        """
        부분 일치 비율: GT 중 몇 개가 예측에 있는지
        
        Returns:
            0.0 ~ 1.0
        """
        if not gt_labels:
            return 0.0
        gt_set = set(gt_labels)
        pred_set = set(pred_labels)
        return len(gt_set & pred_set) / len(gt_set)
    
    # ============ 계층적 메트릭 ============
    
    def hierarchical_similarity(self, label1: str, label2: str) -> float:
        """
        두 라벨 간의 계층적 유사도 (0.0 ~ 1.0)

        Jaccard 유사도 기반 (root 제외):
        Similarity = |Ancestors(A) ∩ Ancestors(B)| / |Ancestors(A) ∪ Ancestors(B)|
        """
        if label1 == label2:
            return 1.0

        # 조상 집합 구하기 (root 제외)
        ancestors1 = set(self.tree.get_path_to_root(label1)) - {'root'}
        ancestors2 = set(self.tree.get_path_to_root(label2)) - {'root'}

        if not ancestors1 or not ancestors2:
            return 0.0

        intersection = ancestors1 & ancestors2
        union = ancestors1 | ancestors2

        if not union:
            return 0.0

        return len(intersection) / len(union)
    
    def best_hierarchical_match(self, gt_label: str, pred_labels: List[str]) -> Tuple[Optional[str], float]:
        """
        GT 라벨과 가장 유사한 예측 라벨 찾기
        
        Returns:
            (best_match, similarity_score)
        """
        if not pred_labels:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for pred in pred_labels:
            sim = self.hierarchical_similarity(gt_label, pred)
            if sim > best_similarity:
                best_similarity = sim
                best_match = pred
        
        return best_match, best_similarity
    
    def hierarchical_precision_recall(
        self, 
        gt_labels: List[str], 
        pred_labels: List[str]
    ) -> Tuple[float, float, float]:
        """
        계층적 Precision, Recall, F1 계산
        
        Precision: 예측 중 GT와 유사한 것의 비율 (가중 평균)
        Recall: GT 중 예측과 유사한 것의 비율 (가중 평균)
        
        Returns:
            (precision, recall, f1)
        """
        if not gt_labels or not pred_labels:
            return 0.0, 0.0, 0.0
        
        # Precision: 각 예측에 대해 가장 유사한 GT와의 유사도
        precision_scores = []
        for pred in pred_labels:
            _, best_sim = self.best_hierarchical_match(pred, gt_labels)
            precision_scores.append(best_sim)
        precision = np.mean(precision_scores) if precision_scores else 0.0
        
        # Recall: 각 GT에 대해 가장 유사한 예측과의 유사도
        recall_scores = []
        for gt in gt_labels:
            _, best_sim = self.best_hierarchical_match(gt, pred_labels)
            recall_scores.append(best_sim)
        recall = np.mean(recall_scores) if recall_scores else 0.0
        
        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return precision, recall, f1
    
    def avg_min_distance(self, gt_labels: List[str], pred_labels: List[str]) -> float:
        """
        평균 최소 거리: 각 GT에 대해 가장 가까운 예측까지의 거리 평균
        
        낮을수록 좋음
        """
        if not gt_labels or not pred_labels:
            return float('inf')
        
        distances = []
        for gt in gt_labels:
            min_dist = float('inf')
            for pred in pred_labels:
                dist = self.tree.get_hierarchical_distance(gt, pred)
                if dist >= 0:
                    min_dist = min(min_dist, dist)
            if min_dist < float('inf'):
                distances.append(min_dist)
        
        return np.mean(distances) if distances else float('inf')
    
    # ============ 레벨별 메트릭 ============
    
    def level_match(self, gt_labels: List[str], pred_labels: List[str], level: int) -> bool:
        """
        특정 레벨에서 일치하는지 확인
        
        GT와 Pred 중 하나라도 해당 레벨에서 같은 조상을 가지면 True
        """
        gt_level_labels = set()
        pred_level_labels = set()
        
        for gt in gt_labels:
            level_labels = self.tree.get_level_labels(gt)
            if level in level_labels:
                gt_level_labels.add(level_labels[level])
        
        for pred in pred_labels:
            level_labels = self.tree.get_level_labels(pred)
            if level in level_labels:
                pred_level_labels.add(level_labels[level])
        
        return len(gt_level_labels & pred_level_labels) > 0
    
    def compute_level_accuracy(
        self, 
        results: List[PredictionResult]
    ) -> Dict[int, float]:
        """
        각 레벨별 정확도 계산
        
        Returns:
            {level: accuracy}
        """
        level_correct = defaultdict(int)
        level_total = defaultdict(int)
        
        for result in results:
            if not result.ground_truth or not result.prediction:
                continue
            
            for level in range(1, self.max_depth + 1):
                # 해당 레벨에 GT나 Pred가 있는 경우에만 카운트
                gt_has_level = any(
                    level in self.tree.get_level_labels(gt) 
                    for gt in result.ground_truth
                )
                pred_has_level = any(
                    level in self.tree.get_level_labels(pred) 
                    for pred in result.prediction
                )
                
                if gt_has_level or pred_has_level:
                    level_total[level] += 1
                    if self.level_match(result.ground_truth, result.prediction, level):
                        level_correct[level] += 1
        
        return {
            level: level_correct[level] / level_total[level] if level_total[level] > 0 else 0.0
            for level in range(1, self.max_depth + 1)
        }
    
    # ============ 부분 점수 메트릭 ============
    
    def partial_credit_score(self, gt_labels: List[str], pred_labels: List[str]) -> float:
        """
        부분 점수 계산
        
        각 레벨에서 맞으면 해당 레벨의 가중치만큼 점수 획득
        더 깊은 레벨이 맞으면 더 높은 점수
        
        Returns:
            0.0 ~ 1.0
        """
        if not gt_labels or not pred_labels:
            return 0.0
        
        max_score = 0.0
        
        for gt in gt_labels:
            gt_path = self.tree.get_path_to_root(gt)
            if not gt_path:
                continue
            
            # 각 예측에 대해 공통 경로 깊이 계산
            for pred in pred_labels:
                pred_path = self.tree.get_path_to_root(pred)
                if not pred_path:
                    continue
                
                # 공통 조상까지의 경로 길이
                gt_set = set(gt_path)
                common_depth = 0
                for i, node in enumerate(pred_path):
                    if node in gt_set:
                        # LCA 찾음
                        lca_depth_from_root = len(gt_path) - gt_path.index(node) - 1
                        common_depth = lca_depth_from_root
                        break
                
                # 정확히 일치하면 만점
                if gt == pred:
                    score = 1.0
                else:
                    # 공통 깊이에 비례한 점수
                    gt_depth = len(gt_path) - 1
                    score = common_depth / gt_depth if gt_depth > 0 else 0.0
                
                max_score = max(max_score, score)
        
        return max_score
    
    # ============ 조상 기반 메트릭 ============
    
    def ancestor_match_score(self, gt_labels: List[str], pred_labels: List[str]) -> float:
        """
        조상 일치 점수: 예측이 GT의 조상이거나 자손이면 부분 점수
        
        - 정확히 일치: 1.0
        - 예측이 GT의 조상: 0.5 ~ 0.9 (거리에 따라)
        - 예측이 GT의 자손: 0.5 ~ 0.9 (거리에 따라)
        - 형제 노드: 0.3 ~ 0.5
        """
        if not gt_labels or not pred_labels:
            return 0.0
        
        max_score = 0.0
        
        for gt in gt_labels:
            gt_ancestors = self.tree.get_ancestors(gt)
            gt_descendants = self.tree.get_all_descendants(gt)
            gt_siblings = set(self.tree.get_siblings(gt))
            
            for pred in pred_labels:
                if gt == pred:
                    score = 1.0
                elif pred in gt_ancestors:
                    # 예측이 GT의 조상
                    dist = self.tree.get_hierarchical_distance(gt, pred)
                    score = max(0.5, 0.9 - dist * 0.1)
                elif pred in gt_descendants:
                    # 예측이 GT의 자손 (더 구체적)
                    dist = self.tree.get_hierarchical_distance(gt, pred)
                    score = max(0.5, 0.9 - dist * 0.1)
                elif pred in gt_siblings:
                    # 형제 노드
                    score = 0.4
                else:
                    # 그 외: 계층적 유사도 기반
                    score = self.hierarchical_similarity(gt, pred) * 0.5
                
                max_score = max(max_score, score)
        
        return max_score
    
    # ============ Top-K 메트릭 ============

    def top_k_exact_match(self, gt_labels: List[str], pred_labels: List[str], k: int) -> bool:
        """
        Top-K Exact Match: GT 중 하나라도 상위 K개 예측에 포함되면 True

        Args:
            gt_labels: 정답 라벨들
            pred_labels: 예측 라벨들 (순서대로 Top-1, Top-2, ...)
            k: 상위 몇 개까지 확인할지

        Returns:
            True if any GT label is in top-K predictions
        """
        gt_set = set(gt_labels)
        top_k_preds = set(pred_labels[:k]) if len(pred_labels) >= k else set(pred_labels)
        return len(gt_set & top_k_preds) > 0

    def top_k_hierarchical_match(
        self,
        gt_labels: List[str],
        pred_labels: List[str],
        k: int,
        threshold: float = 0.5
    ) -> bool:
        """
        Top-K Hierarchical Match: 상위 K개 예측 중 하나라도 GT와 계층적으로 유사하면 True

        Args:
            gt_labels: 정답 라벨들
            pred_labels: 예측 라벨들 (순서대로)
            k: 상위 몇 개까지 확인할지
            threshold: 유사도 임계값

        Returns:
            True if any top-K prediction is hierarchically similar to GT
        """
        top_k_preds = pred_labels[:k] if len(pred_labels) >= k else pred_labels
        for gt in gt_labels:
            for pred in top_k_preds:
                if self.hierarchical_similarity(gt, pred) >= threshold:
                    return True
        return False

    def top_k_hierarchical_f1(
        self,
        gt_labels: List[str],
        pred_labels: List[str],
        k: int
    ) -> Tuple[float, float, float]:
        """
        Top-K Hierarchical F1: 상위 K개 예측을 사용한 계층적 F1

        Args:
            gt_labels: 정답 라벨들
            pred_labels: 예측 라벨들 (순서대로)
            k: 상위 몇 개까지 사용할지

        Returns:
            (precision, recall, f1) using top-K predictions
        """
        top_k_preds = pred_labels[:k] if len(pred_labels) >= k else pred_labels
        return self.hierarchical_precision_recall(gt_labels, top_k_preds)

    def compute_top_k_metrics(
        self,
        gt_labels: List[str],
        pred_labels: List[str],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Dict[int, float]]:
        """
        여러 K값에 대해 Top-K 메트릭 계산

        Args:
            gt_labels: 정답 라벨들
            pred_labels: 예측 라벨들 (순서대로, 예: [primary, diff1, diff2, ...])
            k_values: 계산할 K값들

        Returns:
            {
                'top_k_exact': {1: bool, 3: bool, 5: bool},
                'top_k_h_match': {1: bool, 3: bool, 5: bool},
                'top_k_h_f1': {1: float, 3: float, 5: float}
            }
        """
        gt_valid = self.preprocess_labels(gt_labels)
        pred_valid = self.preprocess_labels(pred_labels)

        result = {
            'top_k_exact': {},
            'top_k_h_match': {},
            'top_k_h_f1': {}
        }

        for k in k_values:
            result['top_k_exact'][k] = self.top_k_exact_match(gt_valid, pred_valid, k) if gt_valid else False
            result['top_k_h_match'][k] = self.top_k_hierarchical_match(gt_valid, pred_valid, k) if gt_valid else False
            _, _, f1 = self.top_k_hierarchical_f1(gt_valid, pred_valid, k) if gt_valid and pred_valid else (0, 0, 0)
            result['top_k_h_f1'][k] = f1

        return result

    # ============ 종합 평가 ============

    def evaluate_single(self, gt_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
        """단일 샘플 평가"""
        # 유효한 라벨만 필터링
        gt_valid = self.preprocess_labels(gt_labels)
        pred_valid = self.preprocess_labels(pred_labels)
        
        if not gt_valid:
            return {'valid': False}
        
        result = {
            'valid': True,
            'exact_match': 1.0 if self.exact_match(gt_valid, pred_valid) else 0.0,
            'partial_match': self.partial_match_ratio(gt_valid, pred_valid),
        }
        
        if pred_valid:
            prec, rec, f1 = self.hierarchical_precision_recall(gt_valid, pred_valid)
            result.update({
                'hierarchical_precision': prec,
                'hierarchical_recall': rec,
                'hierarchical_f1': f1,
                'avg_min_distance': self.avg_min_distance(gt_valid, pred_valid),
                'partial_credit': self.partial_credit_score(gt_valid, pred_valid),
                'ancestor_match': self.ancestor_match_score(gt_valid, pred_valid),
            })
        else:
            result.update({
                'hierarchical_precision': 0.0,
                'hierarchical_recall': 0.0,
                'hierarchical_f1': 0.0,
                'avg_min_distance': float('inf'),
                'partial_credit': 0.0,
                'ancestor_match': 0.0,
            })
        
        return result
    
    def evaluate_batch(
        self, 
        ground_truths: List[List[str]], 
        predictions: List[List[str]]
    ) -> EvaluationResult:
        """
        배치 평가
        
        Args:
            ground_truths: 각 샘플의 GT 라벨 리스트
            predictions: 각 샘플의 예측 라벨 리스트
        
        Returns:
            EvaluationResult
        """
        assert len(ground_truths) == len(predictions), "GT와 Predictions 길이가 다릅니다"
        
        results = []
        sample_metrics = []
        
        for gt, pred in zip(ground_truths, predictions):
            gt_valid = self.preprocess_labels(gt)
            pred_valid = self.preprocess_labels(pred)
            
            results.append(PredictionResult(
                ground_truth=gt_valid,
                prediction=pred_valid,
                raw_ground_truth=gt,
                raw_prediction=pred
            ))
            
            metrics = self.evaluate_single(gt, pred)
            sample_metrics.append(metrics)
        
        # 유효한 샘플만 필터링
        valid_metrics = [m for m in sample_metrics if m.get('valid', False)]
        
        if not valid_metrics:
            return EvaluationResult(
                total_samples=len(ground_truths),
                valid_samples=0,
                skipped_samples=len(ground_truths)
            )
        
        # 평균 계산
        def safe_mean(key):
            values = [m[key] for m in valid_metrics if key in m and m[key] != float('inf')]
            return np.mean(values) if values else 0.0
        
        # 레벨별 정확도
        level_acc = self.compute_level_accuracy(results)
        
        return EvaluationResult(
            exact_match=safe_mean('exact_match'),
            partial_match=safe_mean('partial_match'),
            hierarchical_precision=safe_mean('hierarchical_precision'),
            hierarchical_recall=safe_mean('hierarchical_recall'),
            hierarchical_f1=safe_mean('hierarchical_f1'),
            avg_hierarchical_distance=safe_mean('avg_min_distance'),
            level_accuracy=level_acc,
            avg_partial_credit=safe_mean('partial_credit'),
            total_samples=len(ground_truths),
            valid_samples=len(valid_metrics),
            skipped_samples=len(ground_truths) - len(valid_metrics),
            details={
                'avg_ancestor_match': safe_mean('ancestor_match'),
                'sample_metrics': sample_metrics,
            }
        )

    def evaluate_batch_with_top_k(
        self,
        ground_truths: List[List[str]],
        predictions: List[List[str]],
        k_values: List[int] = [1, 3, 5]
    ) -> EvaluationResult:
        """
        배치 평가 (Top-K 메트릭 포함)

        Args:
            ground_truths: 각 샘플의 GT 라벨 리스트
            predictions: 각 샘플의 예측 라벨 리스트 (순서대로: [primary, diff1, diff2, ...])
            k_values: 계산할 K값들 (기본: [1, 3, 5])

        Returns:
            EvaluationResult with Top-K metrics
        """
        # 기본 평가 수행
        base_result = self.evaluate_batch(ground_truths, predictions)

        # Top-K 메트릭 계산
        top_k_exact_counts = {k: 0 for k in k_values}
        top_k_h_f1_sums = {k: 0.0 for k in k_values}
        valid_count = 0

        for gt, pred in zip(ground_truths, predictions):
            gt_valid = self.preprocess_labels(gt)
            pred_valid = self.preprocess_labels(pred)

            if not gt_valid:
                continue

            valid_count += 1

            for k in k_values:
                # Top-K Exact Match
                if self.top_k_exact_match(gt_valid, pred_valid, k):
                    top_k_exact_counts[k] += 1

                # Top-K Hierarchical F1
                if pred_valid:
                    _, _, f1 = self.top_k_hierarchical_f1(gt_valid, pred_valid, k)
                    top_k_h_f1_sums[k] += f1

        # 평균 계산
        if valid_count > 0:
            base_result.top_k_accuracy = {
                k: top_k_exact_counts[k] / valid_count for k in k_values
            }
            base_result.top_k_hierarchical_f1 = {
                k: top_k_h_f1_sums[k] / valid_count for k in k_values
            }
        else:
            base_result.top_k_accuracy = {k: 0.0 for k in k_values}
            base_result.top_k_hierarchical_f1 = {k: 0.0 for k in k_values}

        return base_result

    def print_evaluation_report(self, result: EvaluationResult):
        """평가 결과 출력"""
        print("=" * 60)
        print("HIERARCHICAL EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\n[Sample Statistics]")
        print(f"  Total samples: {result.total_samples}")
        print(f"  Valid samples: {result.valid_samples}")
        print(f"  Skipped samples: {result.skipped_samples}")
        
        print(f"\n[Basic Metrics]")
        print(f"  Exact Match Accuracy: {result.exact_match:.4f}")
        print(f"  Partial Match Ratio: {result.partial_match:.4f}")
        
        print(f"\n[Hierarchical Metrics]")
        print(f"  Hierarchical Precision: {result.hierarchical_precision:.4f}")
        print(f"  Hierarchical Recall: {result.hierarchical_recall:.4f}")
        print(f"  Hierarchical F1: {result.hierarchical_f1:.4f}")
        print(f"  Avg Min Distance: {result.avg_hierarchical_distance:.4f}")
        
        print(f"\n[Partial Credit]")
        print(f"  Avg Partial Credit Score: {result.avg_partial_credit:.4f}")
        if 'avg_ancestor_match' in result.details:
            print(f"  Avg Ancestor Match Score: {result.details['avg_ancestor_match']:.4f}")
        
        print(f"\n[Level-wise Accuracy]")
        for level, acc in sorted(result.level_accuracy.items()):
            print(f"  Level {level}: {acc:.4f}")

        # Top-K 메트릭 출력 (있는 경우)
        if result.top_k_accuracy:
            print(f"\n[Top-K Accuracy]")
            for k, acc in sorted(result.top_k_accuracy.items()):
                print(f"  Top-{k} Exact Match: {acc:.4f}")

        if result.top_k_hierarchical_f1:
            print(f"\n[Top-K Hierarchical F1]")
            for k, f1 in sorted(result.top_k_hierarchical_f1.items()):
                print(f"  Top-{k} H-F1: {f1:.4f}")

        print("=" * 60)


def demo():
    """데모 함수"""
    try:
        # 자동으로 ontology.json 찾기
        evaluator = HierarchicalEvaluator()
        print(f"Loaded ontology from: {evaluator.tree.ontology_path}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\n사용법:")
        print("  1. 자동 경로: evaluator = HierarchicalEvaluator()")
        print("  2. 수동 경로: evaluator = HierarchicalEvaluator('/path/to/ontology.json')")
        return
    
    print("=== Single Sample Evaluation Demo ===\n")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "Exact Match",
            "gt": ["Tinea corporis"],
            "pred": ["Tinea corporis"]
        },
        {
            "name": "Same Family (Tinea)",
            "gt": ["Tinea corporis"],
            "pred": ["Tinea pedis"]
        },
        {
            "name": "Same Category (fungal)",
            "gt": ["Tinea corporis"],
            "pred": ["Candidiasis"]
        },
        {
            "name": "Different Branch",
            "gt": ["Tinea corporis"],
            "pred": ["Psoriasis"]
        },
        {
            "name": "Ancestor Prediction",
            "gt": ["Tinea corporis"],
            "pred": ["fungal"]
        },
        {
            "name": "Multiple GT, Partial Match",
            "gt": ["Eczema", "Psoriasis"],
            "pred": ["Psoriasis"]
        },
    ]
    
    for tc in test_cases:
        print(f"--- {tc['name']} ---")
        print(f"  GT: {tc['gt']}")
        print(f"  Pred: {tc['pred']}")
        
        result = evaluator.evaluate_single(tc['gt'], tc['pred'])
        print(f"  Results:")
        for key, value in result.items():
            if key != 'valid':
                print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
        print()
    
    print("\n=== Batch Evaluation Demo ===\n")
    
    ground_truths = [tc['gt'] for tc in test_cases]
    predictions = [tc['pred'] for tc in test_cases]
    
    batch_result = evaluator.evaluate_batch(ground_truths, predictions)
    evaluator.print_evaluation_report(batch_result)


if __name__ == "__main__":
    demo()
