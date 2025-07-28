#!/usr/bin/env python3
"""
Simple HTTP server to serve the debug patch file.
"""

import http.server
import socketserver
import os

PORT = 8080

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

os.chdir('/Users/barrulus/py-fmg')

with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
    print(f"Server running at http://localhost:{PORT}/")
    print(f"Debug patch available at: http://localhost:{PORT}/fmg_debug_patch.js")
    print("Press Ctrl+C to stop")
    httpd.serve_forever()