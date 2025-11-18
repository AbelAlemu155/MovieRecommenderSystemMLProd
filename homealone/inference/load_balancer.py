from flask import Flask, request, Response
import requests
import logging
import threading
import time
import os
import itertools

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVERS_ENV = os.getenv('INFERENCE_SERVERS', 'http://inference-1:5000,http://inference-2:5000,http://inference-3:5000')
SERVERS = SERVERS_ENV.split(',')

server_health = {server: True for server in SERVERS}
health_lock = threading.Lock()
server_cycle = None
current_healthy_servers = []
cycle_lock = threading.Lock()

def check_health_background():
    """Background thread to check server health"""
    while True:
        for server in SERVERS:
            try:
                resp = requests.get(f"{server}/health", timeout=2)
                is_healthy = resp.status_code == 200 and resp.json().get('status') == 'ready'
                
                with health_lock:
                    server_health[server] = is_healthy
                    
            except:
                with health_lock:
                    server_health[server] = False
        
        time.sleep(5)

health_thread = threading.Thread(target=check_health_background, daemon=True)
health_thread.start()

def get_healthy_servers():
    """Get list of currently healthy servers"""
    with health_lock:
        return [s for s, healthy in server_health.items() if healthy]
    
def get_next_server():
    """Get next server using round-robin"""
    global server_cycle, current_healthy_servers
    
    healthy = get_healthy_servers()
    if not healthy:
        return None
    
    with cycle_lock:
        # Recreate cycle if servers changed
        if server_cycle is None or set(healthy) != set(current_healthy_servers):
            current_healthy_servers = healthy
            server_cycle = itertools.cycle(healthy)
            logger.info(f"Updated server pool: {healthy}")
        
        return next(server_cycle)
    

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    """Forward requests only to healthy servers"""
    server = get_next_server()
    
    if not server:
        logger.error("No healthy servers available!")
        return {"error": "All servers unavailable"}, 503
    
    url = f"{server}/{path}"
    
    try:
        resp = requests.request(
            method=request.method,
            url=url,
            params=request.args,
            headers={k: v for k, v in request.headers if k != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=30
        )
        
        logger.info(f"Forwarded to {server} - Status: {resp.status_code}")
        
        response = Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type', 'text/plain')
        )
        
        excluded = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        for key, value in resp.headers.items():
            if key.lower() not in excluded:
                response.headers[key] = value
        
        return response
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error forwarding to {server}: {e}")
        with health_lock:
            server_health[server] = False
        
        return {"error": "Request failed"}, 503
    
@app.route('/health', methods=['GET'])
def health():
    """Load balancer health status"""
    with health_lock:
        healthy = [s for s, h in server_health.items() if h]
    
    return {
        "status": "healthy" if healthy else "unhealthy",
        "total_servers": len(SERVERS),
        "healthy_servers": len(healthy),
        "servers": server_health
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=False)
