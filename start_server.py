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
                url_main = f"http://localhost:{port}/dashboard.html"
                url_historic = f"http://localhost:{port}/metrics_historic_dashboard.html"
                
                print(f"Iniciando servidor local en el puerto {port}")
                print("Presiona Ctrl+C para detener el servidor.")
                
                # Abrir ambos reportes en pestañas separadas
                webbrowser.open(url_main)
                webbrowser.open(url_historic)
                
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
