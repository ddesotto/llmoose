import json
from io import BytesIO

import pytest

from llmoose.agents.providers import ChatCompletionProvider


@pytest.mark.parametrize(
    "base_url,requires_parameters",
    [
        ("https://openrouter.ai/api/v1", True),
        ("http://localhost:8000/v1", False),
        ("https://openrouter.ai.example.com/v1", False),
    ],
)
@pytest.mark.parametrize("constrained", [False, True])
def test_response_format_preserves_exact_request(
    monkeypatch, base_url, requires_parameters, constrained
):
    requests = []

    class FakeOpener:
        def open(self, request, timeout):
            requests.append(request)
            return BytesIO(
                json.dumps(
                    {"choices": [{"message": {"content": '{"action_id":1}'}}]}
                ).encode()
            )

    monkeypatch.setattr("llmoose.agents.providers.build_opener", lambda *_: FakeOpener())
    messages = [{"role": "user", "content": "Choose a legal action."}]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"action_id": {"type": "integer", "enum": [1, 3]}},
                "required": ["action_id"],
                "additionalProperties": False,
            },
        },
    }
    provider = ChatCompletionProvider(
        "test/model", api_key="test-secret", base_url=base_url
    )
    kwargs = {"response_format": response_format} if constrained else {}
    completion = provider(messages, **kwargs)
    expected = {
        "model": "test/model",
        "messages": messages,
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    if constrained:
        expected["response_format"] = response_format
        if requires_parameters:
            expected["provider"] = {"require_parameters": True}
    assert json.loads(requests[0].data) == expected
    assert provider.request_payload(messages, **kwargs) == expected
    assert completion.text == '{"action_id":1}'
