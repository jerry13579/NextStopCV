import json
import mimetypes
import os
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

image_dir = "data"
label_dir = "data/seat_labels"
classes = ["empty_seat", "occupied_seat", "seat_back", "seat_back_occupied"]
port = 8765
os.makedirs(label_dir, exist_ok=True)


def images():
    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def label_path(name):
    return os.path.join(label_dir, os.path.splitext(os.path.basename(name))[0] + ".txt")


def read_labels(name):
    path = label_path(name)
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        c, x, y, w, h = line.split()
        rows.append({"class_id": int(c), "x": float(x), "y": float(y), "w": float(w), "h": float(h)})
    return rows


def write_labels(name, labels):
    with open(label_path(name), "w") as file:
        for box in labels:
            file.write(f'{box["class_id"]} {box["x"]:.6f} {box["y"]:.6f} {box["w"]:.6f} {box["h"]:.6f}\n')


html = r"""
<!doctype html><html><head><meta charset="utf-8"><title>Seat Labeler</title>
<style>
body{margin:0;font:15px system-ui;background:#111;color:#eee;display:grid;grid-template-columns:260px 1fr;height:100vh}
aside{padding:14px;background:#1c1c1c;overflow:auto}button,select{width:100%;margin:6px 0;padding:9px;border:0;border-radius:6px;background:#2f6fed;color:white;font:inherit}button.secondary{background:#333}button.danger{background:#9b2c2c}
#list button{background:#262626;text-align:left;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}#list button.active{background:#2f6fed}
main{display:grid;grid-template-rows:auto 1fr}.bar{padding:10px 14px;background:#181818;display:flex;gap:10px;align-items:center}.bar span{margin-left:auto}
#wrap{overflow:auto;display:grid;place-items:start center;padding:16px}canvas{background:#222;max-width:100%;height:auto;cursor:crosshair}
.hint{color:#aaa;line-height:1.4}.boxrow{padding:7px;margin:5px 0;background:#292929;border-radius:6px;cursor:pointer}.selected{outline:2px solid #2f6fed}
</style></head><body><aside>
<h2>Seat Labeler</h2><select id="cls">__CLASS_OPTIONS__</select>
<button onclick="save()">Save</button><button class="secondary" onclick="prev()">Previous</button><button class="secondary" onclick="next()">Next</button><button class="danger" onclick="delBox()">Delete selected</button>
<p class="hint">Draw boxes with the mouse. Press 1-4 for class, S to save, Delete to remove selected.</p><h3>Images</h3><div id="list"></div><h3>Boxes</h3><div id="boxes"></div>
</aside><main><div class="bar"><b id="name"></b><span id="status"></span></div><div id="wrap"><canvas id="c"></canvas></div></main>
<script>
let imgs=[], idx=0, labels=[], selected=-1, img=new Image(), down=null;
const c=document.getElementById('c'), ctx=c.getContext('2d'), cls=document.getElementById('cls');
const classes=__CLASS_JSON__, colors=['#34c759','#ff3b30','#0a84ff','#ffd60a'];
fetch('/api/images').then(r=>r.json()).then(x=>{imgs=x; drawList(); load(0)});
function load(i){idx=Math.max(0,Math.min(imgs.length-1,i)); selected=-1; document.getElementById('name').textContent=imgs[idx]||''; img.onload=()=>{c.width=img.naturalWidth;c.height=img.naturalHeight;render()}; img.src='/image/'+encodeURIComponent(imgs[idx]); fetch('/api/labels?image='+encodeURIComponent(imgs[idx])).then(r=>r.json()).then(x=>{labels=x;render();drawList()})}
function drawList(){document.getElementById('list').innerHTML=imgs.map((x,i)=>`<button class="${i==idx?'active':''}" onclick="load(${i})">${x}</button>`).join('')}
function xy(e){let r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height}}
function norm(a,b){let x1=Math.min(a.x,b.x),y1=Math.min(a.y,b.y),x2=Math.max(a.x,b.x),y2=Math.max(a.y,b.y);return{class_id:+cls.value,x:(x1+x2)/2/c.width,y:(y1+y2)/2/c.height,w:(x2-x1)/c.width,h:(y2-y1)/c.height}}
function px(b){return{x:(b.x-b.w/2)*c.width,y:(b.y-b.h/2)*c.height,w:b.w*c.width,h:b.h*c.height}}
c.onmousedown=e=>down=xy(e); c.onmouseup=e=>{if(!down)return;let b=norm(down,xy(e));down=null;if(b.w>.005&&b.h>.005){labels.push(b);selected=labels.length-1;render()}}; c.onclick=e=>{let p=xy(e);selected=labels.findIndex(b=>{let r=px(b);return p.x>=r.x&&p.y>=r.y&&p.x<=r.x+r.w&&p.y<=r.y+r.h});render()}
function render(){ctx.drawImage(img,0,0);labels.forEach((b,i)=>{let r=px(b);ctx.lineWidth=i==selected?5:3;ctx.strokeStyle=colors[b.class_id]||'#fff';ctx.strokeRect(r.x,r.y,r.w,r.h);ctx.fillStyle=ctx.strokeStyle;ctx.font='22px system-ui';ctx.fillText((classes[b.class_id]||'unknown').replaceAll('_',' '),r.x,Math.max(24,r.y-6))});document.getElementById('boxes').innerHTML=labels.map((b,i)=>`<div class="boxrow ${i==selected?'selected':''}" onclick="selected=${i};render()">${i+1}. ${classes[b.class_id]||'unknown'}</div>`).join('')}
function save(){fetch('/api/labels?image='+encodeURIComponent(imgs[idx]),{method:'POST',body:JSON.stringify(labels)}).then(()=>document.getElementById('status').textContent='Saved '+new Date().toLocaleTimeString())}
function next(){save();load(idx+1)} function prev(){save();load(idx-1)} function delBox(){if(selected>-1){labels.splice(selected,1);selected=-1;render()}}
document.onkeydown=e=>{if('1234'.includes(e.key))cls.value=+e.key-1;if(e.key==='s')save();if(e.key==='Delete')delBox();if(e.key==='ArrowRight')next();if(e.key==='ArrowLeft')prev()}
</script></body></html>
"""
html = html.replace("__CLASS_OPTIONS__", "".join(f'<option value="{i}">{name}</option>' for i, name in enumerate(classes))).replace("__CLASS_JSON__", json.dumps(classes))


class Handler(BaseHTTPRequestHandler):
    def respond(self, data, kind="application/json"):
        data = data.encode() if isinstance(data, str) else data
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self.respond(html, "text/html")
        if path == "/api/images":
            return self.respond(json.dumps(images()))
        if path == "/api/labels":
            return self.respond(json.dumps(read_labels(parse_qs(urlparse(self.path).query)["image"][0])))
        if path.startswith("/image/"):
            name = posixpath.basename(path[7:])
            data = open(os.path.join(image_dir, name), "rb").read()
            return self.respond(data, mimetypes.guess_type(name)[0] or "application/octet-stream")
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/labels":
            return self.send_error(404)
        name = parse_qs(urlparse(self.path).query)["image"][0]
        write_labels(name, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.respond("{}")


url = f"http://127.0.0.1:{port}"
print(f"Seat labeler running at {url}", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
