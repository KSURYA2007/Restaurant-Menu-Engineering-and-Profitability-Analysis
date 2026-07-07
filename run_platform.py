import http.server
import socketserver
import webbrowser
import os
import threading
import time

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving platform at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Open the browser automatically
    print("Opening Master Platform in your default browser...")
    time.sleep(1) # wait for server to bind
    webbrowser.open(f"http://localhost:{PORT}/index.html")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down platform server.")
