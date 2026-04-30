"""Pytest configuration."""


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests that require real model loading (slow)")
