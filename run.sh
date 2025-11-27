#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}Sheptun${NC} - Voice-controlled terminal for Russian language"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./run.sh <command> [options]"
    echo ""
    echo -e "${YELLOW}Application:${NC}"
    echo "  listen              Start voice recognition in CLI mode"
    echo "  install-app         Create macOS menubar application"
    echo "  restart             Restart menubar application"
    echo ""
    echo -e "${YELLOW}Models:${NC}"
    echo "  list-models         Show downloaded Whisper models"
    echo "  cleanup-models      Remove unused models"
    echo ""
    echo -e "${YELLOW}Development:${NC}"
    echo "  check               Run all linting and type checks"
    echo "  test [options]      Run tests (options passed to pytest)"
    echo "  format              Auto-format code with ruff"
    echo "  coverage            Run tests with coverage report"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./run.sh listen"
    echo "  ./run.sh check"
    echo "  ./run.sh test -v"
    echo "  ./run.sh test tests/test_commands.py"
    echo ""
}

activate_venv() {
    if [ ! -d ".venv" ]; then
        echo -e "${RED}Error:${NC} Virtual environment not found."
        echo "Run: python -m venv .venv && pip install -e '.[dev]'"
        exit 1
    fi
    source .venv/bin/activate
}

run_check() {
    echo -e "${BLUE}Running linting and type checks...${NC}"
    echo ""

    echo -e "${YELLOW}[1/3]${NC} Ruff (linting)..."
    if ruff check src tests; then
        echo -e "${GREEN}✓${NC} Ruff passed"
    else
        echo -e "${RED}✗${NC} Ruff failed"
        exit 1
    fi
    echo ""

    echo -e "${YELLOW}[2/3]${NC} Mypy (type checking)..."
    if mypy src; then
        echo -e "${GREEN}✓${NC} Mypy passed"
    else
        echo -e "${RED}✗${NC} Mypy failed"
        exit 1
    fi
    echo ""

    echo -e "${YELLOW}[3/3]${NC} Pyright (IDE type checking)..."
    if pyright src tests; then
        echo -e "${GREEN}✓${NC} Pyright passed"
    else
        echo -e "${RED}✗${NC} Pyright failed"
        exit 1
    fi
    echo ""

    echo -e "${GREEN}All checks passed!${NC}"
}

run_test() {
    echo -e "${BLUE}Running tests...${NC}"
    pytest "$@"
}

run_format() {
    echo -e "${BLUE}Formatting code...${NC}"
    ruff format src tests
    ruff check --fix src tests || true
    echo -e "${GREEN}Done!${NC}"
}

run_coverage() {
    echo -e "${BLUE}Running tests with coverage...${NC}"
    pytest --cov=sheptun --cov-report=term-missing "$@"
}

# Main
activate_venv

case "${1:-}" in
    ""|"-h"|"--help"|"help")
        show_help
        ;;
    "check")
        run_check
        ;;
    "test")
        shift
        run_test "$@"
        ;;
    "format")
        run_format
        ;;
    "coverage")
        shift
        run_coverage "$@"
        ;;
    *)
        sheptun "$@"
        ;;
esac
