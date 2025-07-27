# python3 -m SimpleHTTPPutServer 8080

from http.server import HTTPServer, SimpleHTTPRequestHandler

class PutHTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_PUT(self):
        print(self.headers)
        length = int(self.headers["Content-Length"])
        path = self.translate_path(self.path)
        with open(path, "wb") as dst:
            dst.write(self.rfile.read(length))
        self.send_response(200)
        self.end_headers()


def run(server_class=HTTPServer, handler_class=PutHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

if __name__ == '__main__':
    run()