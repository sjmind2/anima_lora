import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

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


class TestBucketStatsAPI:
    @pytest.mark.asyncio
    async def test_missing_source_dir_returns_400(self, client):
        resp = await client.post(
            "/api/dataset/bucket-stats",
            json={
                "enabled_families": ["M", "L"],
            },
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_nonexistent_dir_returns_error(self, client):
        resp = await client.post(
            "/api/dataset/bucket-stats",
            json={
                "source_dir": "/nonexistent/path/xyz",
                "enabled_families": ["M", "L"],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_empty_dir_returns_zero_counts(self, client, tmp_path):
        empty = tmp_path / "empty_imgs"
        empty.mkdir()
        resp = await client.post(
            "/api/dataset/bucket-stats",
            json={
                "source_dir": str(empty),
                "enabled_families": ["M", "L"],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["total_images"] == 0
        assert "families" in data
        for fam in data["families"].values():
            assert fam["original"] == 0
            assert fam["resized"] == 0

    @pytest.mark.asyncio
    async def test_with_images_returns_correct_distribution(self, client, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (512, 512)).save(img_dir / "small.png")
        Image.new("RGB", (1024, 1024)).save(img_dir / "large.png")
        resp = await client.post(
            "/api/dataset/bucket-stats",
            json={
                "source_dir": str(img_dir),
                "enabled_families": ["S1"],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["total_images"] == 2
        assert "families" in data
        orig_sum = sum(f["original"] for f in data["families"].values())
        assert orig_sum == 2
        resized_sum = sum(f["resized"] for f in data["families"].values())
        assert resized_sum == 2

    @pytest.mark.asyncio
    async def test_empty_enabled_families(self, client, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (512, 512)).save(img_dir / "test.png")
        resp = await client.post(
            "/api/dataset/bucket-stats",
            json={
                "source_dir": str(img_dir),
                "enabled_families": [],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["total_images"] == 1
        resized_sum = sum(f["resized"] for f in data["families"].values())
        assert resized_sum == 1
