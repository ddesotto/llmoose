import pytest


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    """Never load a developer's real .env while exercising the CLI."""
    monkeypatch.chdir(tmp_path)
