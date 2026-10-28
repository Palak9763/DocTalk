"""Unit test for demo_requirements.py execution in fake mode."""
import subprocess
import sys
from pathlib import Path
from rag import config
def test_demo_requirements_fake_mode() -> None:
    """Verify demo_requirements.py runs and passes all checks in default fake mode."""
    python_exe = sys.executable
    script_path = config.ROOT / "demo_requirements.py"
    assert script_path.is_file(), "demo_requirements.py does not exist"

    cmd = [python_exe, str(script_path), "--docs", str(config.NOTES_DIR)]
    result = subprocess.run(
        cmd,capture_output=True,text=True,cwd=str(config.ROOT),)
    stdout = result.stdout
    print(stdout)
    assert result.returncode == 0, f"demo_requirements.py failed with code {result.returncode}:\n{result.stderr}\n{stdout}"
    assert "REQUIREMENT R1: Query Rewriting, Query Expansion, and HyDE" in stdout
    assert ">>> R1: PASS" in stdout
    assert "REQUIREMENT R2: Hybrid Retrieval (Dense + BM25 Sparse), RRF, and Cross-Encoder Rerank" in stdout
    assert ">>> R2: PASS" in stdout
    assert "REQUIREMENT R3: Self-RAG (Adaptive Routing, Chunk Grading, Groundedness Checking)" in stdout
    assert ">>> R3: PASS" in stdout
    assert "ALL CHECKS PASSED (EXIT 0)" in stdout

