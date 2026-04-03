import http.server
import socketserver
import webbrowser

# Set the port to 8000
PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

# This allows the server to restart immediately without "Address already in use" errors
socketserver.TCPServer.allow_reuse_address = True

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server started at http://localhost:{PORT}")
    print("Press CTRL+C to stop the server.")

    # Automatically opens your default browser to the local address
    webbrowser.open(f"http://localhost:{PORT}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()
