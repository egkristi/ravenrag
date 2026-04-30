# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in RavenRAG, please report it responsibly:

**Email:** egkristi@gmail.com

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will respond within 72 hours and work on a fix promptly.

## Scope

- RavenRAG library code
- Built-in HTTP server (`raven serve`)
- MCP server (`raven mcp`)
- CLI tool

## Known Limitations

- The built-in HTTP server is intended for local/development use. For production, use a reverse proxy with TLS.
- API key auth uses Bearer tokens over HTTP — use HTTPS in production.
