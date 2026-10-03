import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_example_is_examples_00_hello() -> None:
    readme = (ROOT / "README.md").read_text()
    block = re.search(r"## Using it\n.*?```python\n(.*?)```", readme, re.S)
    assert block is not None
    assert block.group(1) == (ROOT / "examples" / "00_hello.py").read_text()
