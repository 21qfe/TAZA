import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_chatbot_explain_ai_simple_words():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "Explain artificial intelligence in simple words.",
            "role": "farmer",
            "mode": "chat"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "TAZA" in data["model_used"] or data["model_used"] in ["gpt-5", "TAZA-Qwen2.5-0.5B-LoRA (IBM Z LinuxONE s390x)"]
        assert len(data["output_text"]) > 20
        assert "assistant" in data["output_text"].lower() or "ai" in data["output_text"].lower() or "farming" in data["output_text"].lower()


@pytest.mark.asyncio
async def test_chatbot_farm_query_potato():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "How to manage potato cold storage humidity and temperature in Hooghly?",
            "role": "farmer",
            "mode": "chat",
            "district": "Hooghly"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "potato" in data["output_text"].lower() or "hooghly" in data["output_text"].lower() or "storage" in data["output_text"].lower()


@pytest.mark.asyncio
async def test_chatbot_automate_listing_task():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "List 500kg potato at 20",
            "role": "farmer",
            "mode": "task"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["automated_task"] is not None
        task = data["automated_task"]
        assert task["type"] == "create_listing"
        assert task["executed"] is True
        assert task["deployed"] is True
        assert task["details"]["quantity_kg"] == 500.0
        assert task["details"]["price_per_kg"] == 20.0
        assert "Jyoti Potato" in task["title"]
        assert task["html_card"] is not None


@pytest.mark.asyncio
async def test_chatbot_automate_warehouse_task():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "Reserve cold storage space in Singur Hub",
            "role": "farmer",
            "mode": "task"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["automated_task"] is not None
        assert data["automated_task"]["type"] == "book_warehouse"
        assert data["automated_task"]["executed"] is True
        assert "Singur" in data["automated_task"]["html_card"]


@pytest.mark.asyncio
async def test_chatbot_automate_logistics_task():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "Dispatch logistics route for Kolkata delivery",
            "role": "farmer",
            "mode": "task"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["automated_task"] is not None
        assert data["automated_task"]["type"] == "dispatch_logistics"
        assert "Waybill" in data["automated_task"]["title"] or "Waybill" in data["automated_task"]["html_card"]


@pytest.mark.asyncio
async def test_chatbot_automate_weather_protection_task():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "Scan weather and protect crops from rain",
            "role": "farmer",
            "mode": "task"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["automated_task"] is not None
        assert data["automated_task"]["type"] == "protect_weather"
        assert data["automated_task"]["executed"] is True


@pytest.mark.asyncio
async def test_chatbot_automate_harvest_revenue_pie():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "Show harvest revenue pie chart",
            "role": "farmer",
            "mode": "task"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["automated_task"] is not None
        assert data["automated_task"]["type"] == "harvest_revenue"
        assert "<svg" in data["automated_task"]["html_card"]


@pytest.mark.asyncio
async def test_chatbot_automate_deploy_everything():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "Deploy everything",
            "role": "farmer",
            "mode": "task"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["automated_task"] is not None
        assert data["automated_task"]["type"] == "deploy_everything"
        assert data["automated_task"]["deployed"] is True
        assert "ALL PIPELINES DEPLOYED" in data["automated_task"]["html_card"]


@pytest.mark.asyncio
async def test_chatbot_greeting_hi():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "hi",
            "role": "farmer",
            "mode": "chat"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "hello" in data["output_text"].lower() or "taza" in data["output_text"].lower()
        assert data["automated_task"] is None


@pytest.mark.asyncio
async def test_chatbot_list_500kg_at_20_dash():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "list 500kg at 20/-",
            "role": "farmer",
            "mode": "chat"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["automated_task"] is not None
        task = data["automated_task"]
        assert task["type"] == "create_listing"
        assert task["details"]["quantity_kg"] == 500.0
        assert task["details"]["price_per_kg"] == 20.0
        assert "View In Live Marketplace" in task["html_card"]


@pytest.mark.asyncio
async def test_chatbot_contact_privacy_no_fake_numbers():
    """Verify that asking for contact info returns in-app chat and never hallucinates random phone numbers."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "What is your customer support contact phone number?",
            "role": "farmer",
            "mode": "chat"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        out = data["output_text"]
        # Ensure in-app support or Kisan Mitra is recommended
        assert "taza live chat" in out.lower() or "kisan mitra" in out.lower() or "masked contact relay" in out.lower()
        # Ensure zero hallucinated 1800 or random phone numbers are exposed
        import re
        assert not re.search(r"1800[-\s]?\d{3}", out), "Hallucinated 1800 number detected!"
        assert not re.search(r"\+91[-\s]?[6-9]\d{9}", out), "Hallucinated mobile number detected!"


@pytest.mark.asyncio
async def test_chatbot_mandi_surge_profit_simulator():
    """Verify Mandi Benchmark & 20% Surge Profit Simulator explanation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "Explain the 20% surge profit simulator and Mandi arbitrage.",
            "role": "farmer",
            "mode": "chat"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "surge" in data["output_text"].lower() or "arbitrage" in data["output_text"].lower() or "mandi" in data["output_text"].lower()


@pytest.mark.asyncio
async def test_chatbot_cold_storage_space_fit_radar():
    """Verify Cold Storage Space Fit Calculator & 40km Distance Radar."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/v1/chatbot/chat", json={
            "input": "How does the space fit calculator and distance radar work for cold storage?",
            "role": "farmer",
            "mode": "chat"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "space fit" in data["output_text"].lower() or "radar" in data["output_text"].lower() or "vault" in data["output_text"].lower()
