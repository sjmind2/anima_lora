import pytest
import pytest_asyncio
from pathlib import Path
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from workflow.app import create_app


@pytest.fixture
def app(tmp_path):
    wf_root = tmp_path / "workflows"
    wf_root.mkdir()
    return create_app(workflows_root=wf_root)


@pytest_asyncio.fixture
async def client(app):
    server = TestServer(app)
    await server.start_server()
    c = TestClient(server)
    await c.start_server()
    yield c
    await c.close()
    await server.close()


class TestAPI:
    @pytest.mark.asyncio
    async def test_list_workflows_empty(self, client):
        resp = await client.get("/api/workflows")
        assert resp.status == 200
        data = await resp.json()
        assert data == []

    @pytest.mark.asyncio
    async def test_create_workflow(self, client):
        resp = await client.post("/api/workflows", json={"name": "test-wf"})
        assert resp.status == 201
        data = await resp.json()
        assert data["name"] == "test-wf"

    @pytest.mark.asyncio
    async def test_get_schema(self, client):
        resp = await client.get("/api/schemas/preprocess")
        assert resp.status == 200
        data = await resp.json()
        assert data["type"] == "preprocess"

    @pytest.mark.asyncio
    async def test_get_schemas_train_common(self, client):
        resp = await client.get("/api/schemas/train_common")
        assert resp.status == 200
        data = await resp.json()
        assert data["type"] == "train_common"

    @pytest.mark.asyncio
    async def test_get_infrastructure_schema(self, client):
        resp = await client.get("/api/schemas/infrastructure")
        assert resp.status == 200
