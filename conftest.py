"""Shared pytest setup for the XAI-SOC test suite."""
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The application must be imported only after this line, so the real soc_database.db is never touched.
os.environ["SOC_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="xaisoc_test_"), "test.db")

USERNAME = os.environ.get("SOC_ADMIN_USER", "admin")
PASSWORD = os.environ.get("SOC_ADMIN_PASS", "admin123")


@pytest.fixture(scope="session")
def A():
    """The imported application module (needs threat_model.pkl, model_metrics.json, demo_events.csv)."""
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module


@pytest.fixture(autouse=True)
def clean_db(A):
    """Every test starts with an empty alerts table."""
    with A.get_db() as conn:
        conn.execute("DELETE FROM alerts")


@pytest.fixture()
def anon(A):
    return A.app.test_client()


@pytest.fixture()
def client(A):
    c = A.app.test_client()
    response = c.post("/login", data={"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 302
    return c
