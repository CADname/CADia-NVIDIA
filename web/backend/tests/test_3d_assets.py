from __future__ import annotations

import struct

from fastapi.testclient import TestClient

from standalonecad_web.main import app


def _register(client: TestClient, email: str):
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "test-password-123", "display_name": "3D Tester"},
    )
    assert response.status_code == 201, response.text


def _minimal_glb() -> bytes:
    # Valid GLB 2.0 container with an empty JSON scene.
    payload = b'{"asset":{"version":"2.0"},"scene":0,"scenes":[{}]}'
    while len(payload) % 4:
        payload += b" "
    chunk = struct.pack("<I4s", len(payload), b"JSON") + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


def test_3d_asset_import_list_download_delete():
    with TestClient(app) as client:
        _register(client, "asset-layer@example.com")
        project_id = client.post("/api/projects", json={"name": "Asset layer"}).json()["id"]

        uploaded = client.post(
            f"/api/projects/{project_id}/assets/import",
            files={"file": ("reference.glb", _minimal_glb(), "model/gltf-binary")},
            data={"name": "Reference Mesh"},
        )
        assert uploaded.status_code == 201, uploaded.text
        asset = uploaded.json()
        assert asset["name"] == "Reference Mesh"
        assert asset["format"] == "glb"

        listed = client.get(f"/api/projects/{project_id}/assets")
        assert listed.status_code == 200
        assert [row["id"] for row in listed.json()] == [asset["id"]]

        content = client.get(asset["content_url"])
        assert content.status_code == 200
        assert content.content[:4] == b"glTF"

        deleted = client.delete(f"/api/projects/{project_id}/assets/{asset['id']}")
        assert deleted.status_code == 204
        assert client.get(f"/api/projects/{project_id}/assets").json() == []
