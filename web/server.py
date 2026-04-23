#!/usr/bin/env python
"""
Tiny HTTP server for local development. WebGPU usually requires
a "secure context", which means http://localhost:8080/ rather than
file:///path/to/index.html.

Usage:
    cd web && python server.py

Then open http://localhost:8080/ in Chrome.
"""
import http.server
import socketserver
import os
import sys

PORT = 9876

# Serve the directory this script lives in
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files with permissive CORS so fetch() can read weights.bin
    cleanly even when the model file lives in a different directory or
    when you proxy this through nginx."""
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        super().end_headers()


def main():
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Serving http://localhost:{PORT}/  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye.")


if __name__ == "__main__":
    main()
