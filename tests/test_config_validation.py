"""Tests for config validation improvements."""

import pytest

from ravenrag.config import ServerConfig


class TestServerConfigValidation:
    def test_valid_port(self):
        cfg = ServerConfig(port=8080)
        assert cfg.port == 8080

    def test_port_zero(self):
        cfg = ServerConfig(port=0)
        assert cfg.port == 0

    def test_port_max(self):
        cfg = ServerConfig(port=65535)
        assert cfg.port == 65535

    def test_port_too_high(self):
        with pytest.raises(ValueError, match="port must be 0-65535"):
            ServerConfig(port=70000)

    def test_port_negative(self):
        with pytest.raises(ValueError, match="port must be 0-65535"):
            ServerConfig(port=-1)
