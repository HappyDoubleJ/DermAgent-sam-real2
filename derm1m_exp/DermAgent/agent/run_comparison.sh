#!/bin/bash
# ==============================================================================
# DermatologyAgent vs ReActDermatologyAgent 비교 실행 스크립트
# ==============================================================================

set -e

# 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="/home/work/wonjun/DermAgent/dataset/Derm1M"
CSV_FILE="${DATASET_DIR}/Derm1M_v2_pretrain_ontology_sampled_100.csv"

# 결과 디렉터리
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/results/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  DermatologyAgent vs ReActDermatologyAgent Comparison${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${GREEN}Timestamp:${NC} $TIMESTAMP"
echo -e "${GREEN}Output Dir:${NC} $OUTPUT_DIR"
echo -e "${GREEN}CSV File:${NC} $CSV_FILE"
echo ""

# 옵션 파싱
SEED=""
ROW="--row 5"  # 기본값: row 5 (allergic contact dermatitis)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --seed N        Random seed for sample selection"
    echo "  --row N         Use specific row (0-indexed)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --seed 42               # Use seed 42 for random selection"
    echo "  $0 --row 5                 # Use row 5 from CSV"
    echo "  $0                         # Random sample with GPT-4o"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="--seed $2"
            shift 2
            ;;
        --row)
            ROW="--row $2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# 환경 확인
echo -e "${YELLOW}Checking environment...${NC}"

if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}Error: CSV file not found: $CSV_FILE${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} CSV file exists"

# Python 환경 확인
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Python found: $(python --version)"

# 필요한 패키지 확인
python -c "import openai" 2>/dev/null && echo -e "  ${GREEN}✓${NC} openai package installed" || {
    echo -e "  ${YELLOW}!${NC} openai not installed. Installing..."
    pip install openai -q
}

python -c "from dotenv import load_dotenv" 2>/dev/null && echo -e "  ${GREEN}✓${NC} python-dotenv installed" || {
    echo -e "  ${YELLOW}!${NC} python-dotenv not installed. Installing..."
    pip install python-dotenv -q
}

echo ""
echo -e "${YELLOW}Running comparison...${NC}"
echo ""

# Python 스크립트 실행
cd "$SCRIPT_DIR"
python compare_agents.py \
    --csv "$CSV_FILE" \
    --image_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $SEED $ROW

# 결과 파일 확인
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Results${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"
echo ""
echo -e "${GREEN}Generated files:${NC}"
ls -la "$OUTPUT_DIR"
echo ""

# JSON 결과 요약
if ls "$OUTPUT_DIR"/*evaluation*.json 1> /dev/null 2>&1; then
    echo -e "${GREEN}Evaluation Summary:${NC}"
    for f in "$OUTPUT_DIR"/*evaluation*.json; do
        python -c "
import json
with open('$f') as f:
    data = json.load(f)
    meta = data.get('metadata', {})
    print(f\"  Model: {meta.get('model', 'N/A')}\")
    print(f\"  Timestamp: {meta.get('timestamp', 'N/A')}\")
    print(f\"  Ground Truth: {data.get('ground_truth', 'N/A')}\")

    derm = data.get('dermatology_agent', {})
    react = data.get('react_agent', {})

    print(f\"  DermatologyAgent:\")
    print(f\"    Prediction: {derm.get('prediction', [])}\")
    derm_eval = derm.get('evaluation', {})
    print(f\"    Hierarchical F1: {derm_eval.get('hierarchical_f1', 'N/A')}\")
    print(f\"    Exact Match: {derm_eval.get('exact_match', 'N/A')}\")

    print(f\"  ReActAgent:\")
    print(f\"    Prediction: {react.get('prediction', [])}\")
    react_eval = react.get('evaluation', {})
    print(f\"    Hierarchical F1: {react_eval.get('hierarchical_f1', 'N/A')}\")
    print(f\"    Exact Match: {react_eval.get('exact_match', 'N/A')}\")
"
    done
fi

echo ""
echo -e "${GREEN}Comparison complete!${NC}"
echo -e "${BLUE}============================================================${NC}"
