import http.server
import socketserver
import os
import webbrowser

PORT = 8080
DIRECTORY = "reports"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run_server():
    # Asegurarse de estar en la raíz del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    if not os.path.exists(DIRECTORY):
        print(f"Directorio '{DIRECTORY}' no encontrado. Generando reporte primero...")
        os.makedirs(DIRECTORY, exist_ok=True)

    # Intentar puerto 8080, si falla probar 8081
    port = PORT
    while True:
        try:
            with socketserver.TCPServer(("", port), Handler) as httpd:
                url = f"http://localhost:{port}/dashboard.html"
                print(f"Iniciando servidor en {url}")
                print("Presiona Ctrl+C para detener el servidor.")
                webbrowser.open(url)
                httpd.serve_forever()
                break
        except OSError:
            print(f"Puerto {port} ocupado, probando siguiente...")
            port += 1
            if port > 8090:
                print("No se encontraron puertos libres.")
                break

if __name__ == "__main__":
    run_server()
