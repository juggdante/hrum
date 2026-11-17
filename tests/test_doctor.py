"""make doctor / demo-check helpers that decide whether a service is really up."""
import http.server
import threading

import pytest

from infra import doctor


@pytest.fixture
def http_server():
    """A local server that answers /_stcore/health like Streamlit and 404 elsewhere."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/_stcore/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def test_http_ok_detects_a_healthy_app(http_server):
    assert doctor.http_ok(f"http://127.0.0.1:{http_server}/_stcore/health") == (True, "ok")


def test_http_ok_false_for_other_program_and_closed_port(http_server):
    assert doctor.http_ok(f"http://127.0.0.1:{http_server}/other")[0] is False
    assert doctor.http_ok("http://127.0.0.1:9/_stcore/health", timeout=0.5)[0] is False


@pytest.mark.parametrize("out,status", [
    ('{"status": "answered", "grounding": {"ok": true}}\ntrace: sent to Langfuse', "answered"),
    ('noise\n{"status": "abstain", "grounding": {"ok": true}}', "abstain"),
])
def test_parse_cli_json_ignores_surrounding_lines(out, status):
    assert doctor.parse_cli_json(out)["status"] == status


def test_parse_cli_json_none_on_garbage():
    assert doctor.parse_cli_json("Traceback (most recent call last): ...") is None


def test_demo_mode_fails_when_app_port_is_taken_by_another_program(monkeypatch):
    # a server that answers 404 on the health path: "port in use, not Streamlit"
    class Handler404(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler404)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(doctor, "OPTS", doctor.argparse.Namespace(cpu=True, demo=True, smoke=False,
                                                                      app_port=srv.server_address[1]))
        app = next(r for r in doctor.check_services(cpu=True) if r.name.startswith("HRUM app"))
        assert app.status == doctor.FAIL and "not a healthy Streamlit" in app.detail
    finally:
        srv.shutdown()


def test_parse_listeners_keeps_only_lines_with_a_visible_pid():
    text = ('LISTEN 0 2048 0.0.0.0:8501 0.0.0.0:* users:(("streamlit",pid=633027,fd=6))\n'
            'LISTEN 0 2048 127.0.0.1:8502 0.0.0.0:* users:(("streamlit",pid=700001,fd=6))\n'
            'LISTEN 0 4096 127.0.0.1:631 0.0.0.0:*\n')
    assert doctor.parse_listeners(text) == {8501: [633027], 8502: [700001]}


@pytest.mark.parametrize("explicit,env_port,detected,expected", [
    (8503, "8502", [8504], (8503, "--app-port")),
    (None, "8502", [8504], (8502, "APP_PORT in .env")),
    (None, "", [8504], (8504, "detected HRUM process")),
    (None, "", [], (8501, "default")),
])
def test_resolve_app_port_priority(explicit, env_port, detected, expected):
    assert doctor.resolve_app_port(explicit, env_port, detected) == expected


def test_this_test_process_is_not_the_hrum_app():
    import os

    assert doctor.is_hrum_app(os.getpid()) is False
    assert doctor.is_hrum_app(999_999_999) is False        # no such process
