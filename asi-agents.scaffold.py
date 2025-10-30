"""
asi_agents_scaffold.py

A scaffold demonstrating:
- Agent abstraction (perceive, reason, act)
- Integration points for Fetch.ai uAgents and SingularityNET MeTTa (stubs)
- Agentverse registry (simulated)
- Chat Protocol endpoint (Flask) for human-agent interaction
- Example of agents discovering and messaging each other

Notes:
- Replace stub functions with real SDK/API calls for production.
- This is intentionally minimal and educational — not a full production runtime.
"""
import asyncio
import json
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from aiohttp import ClientSession
from flask import Flask, request, jsonify
import threading
import random
import time

# ---------------------------
# Configuration & Utilities
# ---------------------------

AGENTVERSE_REGISTRY = {}  # simulated in-memory registry (agent_id -> metadata)


def generate_agent_id(prefix="agent"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------
# MeTTa Knowledge Graph Stub
# ---------------------------

class MeTTaClient:
    """
    Stub client for SingularityNET's MeTTa Knowledge Graph.
    Replace these methods with real queries to MeTTa.
    """

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint

    async def query(self, query_text: str) -> Dict[str, Any]:
        # Simulated semantic lookup / structured knowledge retrieval
        # In production this would call MeTTa's API, returning structured results.
        await asyncio.sleep(0.1)
        return {
            "query": query_text,
            "knowledge_snippet": f"Simulated knowledge for: '{query_text}'",
            "confidence": round(random.uniform(0.6, 0.99), 2)
        }

# ---------------------------
# Agent Abstraction
# ---------------------------

@dataclass
class Perception:
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    type: str
    payload: Dict[str, Any]


class BaseAgent:
    """
    Base class for agents. Provides perceived input -> reasoning -> action pipeline.
    Replace or extend perceive() with an NLP model, integrate MeTTa in reason(), and
    use act() to trigger real-world actions (HTTP calls, transactions, messages).
    """

    def __init__(self, name: str, metta_client: MeTTaClient, agent_id: Optional[str] = None):
        self.name = name
        self.id = agent_id or generate_agent_id(prefix=name)
        self.metta = metta_client
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.metadata: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "capabilities": ["nlp", "reason", "act"],
            "last_seen": time.time()
        }
        self.running = False
        print(f"[{self.name}] Created with id={self.id}")

    # perception: turn raw input into structured perception
    async def perceive(self, raw_text: str) -> Perception:
        # Simple rule-based parser; replace with actual NLP (transformers, spaCy, etc.)
        parsed = {"intent": None, "entities": []}
        text = raw_text.lower()
        if "status" in text or "health" in text:
            parsed["intent"] = "status_check"
        elif "deliver" in text or "pickup" in text:
            parsed["intent"] = "logistics_request"
        elif "price" in text or "quote" in text:
            parsed["intent"] = "finance_quote"
        else:
            parsed["intent"] = "general_query"
        return Perception(raw_text=raw_text, parsed=parsed)

    # reasoning: combine perception + knowledge graph -> plan
    async def reason(self, perception: Perception) -> Dict[str, Any]:
        # Example: consult MeTTa for context
        kg = await self.metta.query(perception.raw_text)
        plan = {
            "intent": perception.parsed["intent"],
            "kg": kg,
            "decision": None,
            "confidence": kg.get("confidence", 0.7)
        }

        # Simple decision logic
        if plan["intent"] == "status_check":
            plan["decision"] = {"action": "report_status"}
        elif plan["intent"] == "logistics_request":
            plan["decision"] = {"action": "initiate_delivery", "params": {"eta_min": 15}}
        elif plan["intent"] == "finance_quote":
            plan["decision"] = {"action": "fetch_quote", "params": {"symbol": "SIM"}}
        else:
            plan["decision"] = {"action": "answer_query", "params": {"reply": kg["knowledge_snippet"]}}
        return plan

    # action: execute side-effects (HTTP call, message, etc.)
    async def act(self, plan: Dict[str, Any]) -> Action:
        action = plan["decision"]
        act_type = action["action"]
        print(f"[{self.name}] Acting: {act_type} (confidence={plan['confidence']})")
        # Replace with actual integrations:
        if act_type == "report_status":
            # e.g., update Agentverse heartbeat; respond back
            result = {"status": "OK", "uptime": int(time.time())}
            return Action(type="status", payload=result)
        elif act_type == "initiate_delivery":
            # call external logistics API
            await asyncio.sleep(0.2)
            return Action(type="delivery_initiated", payload={"eta_min": action["params"]["eta_min"]})
        elif act_type == "fetch_quote":
            # call an external price oracle / API
            quote = {"symbol": action["params"]["symbol"], "price": round(random.uniform(10, 100), 2)}
            return Action(type="quote", payload=quote)
        else:
            # generic reply
            return Action(type="reply", payload={"text": action["params"]["reply"]})

    # API to deliver a message to this agent
    async def receive_message(self, message: Dict[str, Any]):
        await self.inbox.put(message)

    # agent main loop
    async def run(self):
        self.running = True
        print(f"[{self.name}] Running event loop...")
        while self.running:
            try:
                message = await asyncio.wait_for(self.inbox.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # periodic heartbeat / registration refresh
                await asyncio.sleep(0.01)
                continue
            raw_text = message.get("text", "")
            perception = await self.perceive(raw_text)
            plan = await self.reason(perception)
            action = await self.act(plan)
            # publish or respond depending on action
            # Here we simply print; in production you'd trigger APIs / send messages
            print(f"[{self.name}] Action result: {action.type} -> {action.payload}")
            # Optionally send reply back to sender
            sender = message.get("from")
            if sender:
                await send_direct_message(sender, {
                    "from": self.id,
                    "text": f"@{self.name} processed '{raw_text}' -> {action.type}: {action.payload}"
                })

    def stop(self):
        self.running = False

# ---------------------------
# Agentverse (Registry/Orchestration) - Simulated
# ---------------------------

async def register_agent(agent_meta: Dict[str, Any]):
    # In production, this would call Agentverse registry APIs.
    AGENTVERSE_REGISTRY[agent_meta["id"]] = agent_meta
    print(f"[Agentverse] Registered agent: {agent_meta['id']} ({agent_meta['name']})")
    return True


async def discover_agents(filter_capability: Optional[str] = None) -> List[Dict[str, Any]]:
    results = []
    for a in AGENTVERSE_REGISTRY.values():
        if not filter_capability or filter_capability in a.get("capabilities", []):
            results.append(a)
    return results


# ---------------------------
# Messaging & Chat Protocol (Simplified)
# ---------------------------

AGENT_INBOXES: Dict[str, BaseAgent] = {}

async def send_direct_message(agent_id: str, message: Dict[str, Any]):
    """
    Send a message to another agent by id (simulated).
    In production this would use Agentverse messaging channel or peer-to-peer transport.
    """
    agent = AGENT_INBOXES.get(agent_id)
    if not agent:
        print(f"[Msg] Agent {agent_id} not found")
        return False
    await agent.receive_message(message)
    return True

# ---------------------------
# Flask Chat Protocol wrapper
# ---------------------------

app = Flask("asi_chat_protocol")

@app.route("/chat/send", methods=["POST"])
def chat_send():
    """
    Minimal Chat Protocol endpoint:
    POST /chat/send
    {
        "to": "<agent_id>",
        "text": "Hello agent",
        "from": "<human_or_agent_id>"
    }
    """
    payload = request.json
    to = payload.get("to")
    text = payload.get("text")
    sender = payload.get("from", "human:cli")
    if not to or not text:
        return jsonify({"error": "invalid payload"}), 400
    # dispatch asynchronously to agent loop
    asyncio.run(send_direct_message(to, {"from": sender, "text": text}))
    return jsonify({"status": "sent", "to": to}), 200

@app.route("/agent/registry", methods=["GET"])
def list_registry():
    return jsonify(list(AGENTVERSE_REGISTRY.values())), 200


def run_flask_in_thread(port=5005):
    server = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False))
    server.daemon = True
    server.start()
    print("[ChatProtocol] Flask endpoint running on port", port)
    return server

# ---------------------------
# Example: Two agents discover & talk
# ---------------------------

async def demo_scenario():
    metta = MeTTaClient()
    # Create two agents
    alice = BaseAgent(name="alice", metta_client=metta)
    bob = BaseAgent(name="bob", metta_client=metta)

    # Register agents with Agentverse (simulated)
    await register_agent(alice.metadata)
    await register_agent(bob.metadata)

    # Make agent inboxes discoverable by id so send_direct_message can find them
    AGENT_INBOXES[alice.id] = alice
    AGENT_INBOXES[bob.id] = bob

    # Start agents concurrently
    alice_task = asyncio.create_task(alice.run())
    bob_task = asyncio.create_task(bob.run())

    # Simulate human sending chat to Alice via Chat Protocol (Flask endpoint)
    # We can also call send_direct_message directly.
    await send_direct_message(alice.id, {"from": "human:cli", "text": "Hello Alice — what's your status?"})
    await asyncio.sleep(0.5)

    # Alice triggers a logistics request to Bob (simulate cross-agent call)
    # We'll send a message as if Alice decided to ask Bob for help:
    await send_direct_message(bob.id, {"from": alice.id, "text": "Please deliver package to station 7, ETA?"})
    await asyncio.sleep(0.5)

    # Human asks Bob for a quote
    await send_direct_message(bob.id, {"from": "human:cli", "text": "Can you give me a quote on SIM token price?"})
    await asyncio.sleep(0.5)

    # Let the agents run for a bit
    await asyncio.sleep(2.0)

    # Stop agents
    alice.stop()
    bob.stop()
    alice_task.cancel()
    bob_task.cancel()
    print("[Demo] Completed demo scenario.")


# ---------------------------
# Entrypoint
# ---------------------------

def main():
    # start chat protocol (Flask) in background so humans can post to /chat/send
    run_flask_in_thread(port=5005)
    # run demo scenario
    try:
        asyncio.run(demo_scenario())
    except Exception as e:
        print("Main error:", e)

if __name__ == "__main__":
    main()
