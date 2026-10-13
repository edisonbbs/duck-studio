"""Local XgoDuck console. Uses the existing venv; never changes robot recipes."""
from __future__ import annotations
import json, os, signal, subprocess, sys, threading, time, secrets, socket, shutil, io, zipfile
import versions
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / 'xgoduck_rl'
LOGS = REPO / 'logs/rsl_rl'
PY = REPO / '.venv/bin/python'
PORT = int(os.environ.get('XGODUCK_STUDIO_PORT', '8092'))
TOKEN = secrets.token_urlsafe(32)
TASKS = {}
JOBS = {}
LOCK = threading.RLock()
METRIC_LOCK = threading.RLock()
CACHE = {}
INSPECTOR = None
HARDWARE = None
(ROOT / 'runtime').mkdir(exist_ok=True)

def discover_tasks():
    code = 'import mjlab_microduck.tasks; from mjlab.tasks.registry import list_tasks,load_rl_cfg; import json; print(json.dumps({t:load_rl_cfg(t).experiment_name for t in list_tasks() if t.endswith("XgoDuck")}))'
    out = subprocess.check_output([str(PY), '-c', code], cwd=REPO, text=True, timeout=60)
    return json.loads(out.strip().splitlines()[-1])

def inside(value, directory):
    p = (directory / value).resolve()
    if not p.is_relative_to(directory.resolve()) or not p.exists():
        raise ValueError('找不到所選檔案，或路徑超出訓練目錄')
    return p

def log_path(value):
    if value.startswith('@'):
        ident,relative=value[1:].split('/',1)
        versions.get_version(ident)
        return inside(relative,versions.STORE/ident/'workspace/logs/rsl_rl')
    return inside(value,LOGS)

def runs():
    result=[]
    sources=[('v001',LOGS,'')]+[(v['id'],versions.STORE/v['id']/'workspace/logs/rsl_rl','@'+v['id']+'/') for v in versions.list_versions() if v['kind']=='recipe']
    for ident,base,prefix in sources:
        for d in base.glob('*/*'):
            if not d.is_dir(): continue
            files=list(d.glob('events.out.tfevents*'))
            cps=sorted(d.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
            onnx=list(d.glob('*.onnx'))
            if not files and not cps:continue
            result.append({'id':prefix+str(d.relative_to(base)),'version':ident,'name':d.name,'experiment':d.parent.name,
                'modified':max([p.stat().st_mtime for p in files+cps] or [d.stat().st_mtime]),
                'checkpoints':[prefix+str(p.relative_to(base)) for p in reversed(cps)],
                'onnx':[prefix+str(p.relative_to(base)) for p in onnx]})
    return sorted(result,key=lambda r:r['modified'],reverse=True)

def available(port):
    from urllib.request import urlopen
    try:
        with urlopen(f'http://127.0.0.1:{port}/',timeout=.5) as response:
            response.read()
            return response.status==200
    except Exception:return False

def external_processes():
    found=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit(): continue
        try:
            argv=(d/'cmdline').read_bytes().decode().split('\0')
            if not any(str(REPO / '.venv/bin' / name) in argv for name in ('train','play','tensorboard')): continue
            found.append({'pid':int(d.name),'command':' '.join(argv).strip(),
                          'kind':next((name for name in ('train','play','tensorboard') if str(REPO/'.venv/bin'/name) in argv),'other')})
        except (OSError,UnicodeError): pass
    return found

def job_state():
    out={}
    for kind,j in list(JOBS.items()):
        code=j['process'].poll()
        try:
            with open(j['log'], 'rb') as f:
                f.seek(0,2); size=f.tell(); f.seek(max(0,size-12000)); tail=f.read().decode(errors='replace')
        except OSError: tail=''
        out[kind]={'running':code is None,'exit_code':code,'pid':j['process'].pid,'started':j['started'],
                   'task':j['task'],'checkpoint':j.get('checkpoint'),'log':tail}
    return out

def metrics(run):
    d=log_path(run)
    if not d.is_dir(): raise ValueError('無效訓練記錄')
    cached=CACHE.get(run)
    if cached and time.time()-cached[0]<8: return cached[1]
    ea=cached[2] if cached else EventAccumulator(str(d),size_guidance={'scalars':600})
    ea.Reload()
    series={}
    for tag in ea.Tags().get('scalars',[]):
        series[tag]=[[e.step,e.value] for e in ea.Scalars(tag) if __import__('math').isfinite(e.value)]
    result={'series':series,'read_at':time.time()}
    CACHE[run]=(time.time(),result,ea)
    return result

def gpu():
    try:
        row=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0]
        name,used,total,util,temp=[x.strip() for x in row.split(',')]
        return {'name':name,'used':int(used),'total':int(total),'util':int(util),'temp':int(temp)}
    except Exception: return None

def start(kind,data):
    if kind not in ('train','play','export'): raise ValueError('無效操作')
    if kind in JOBS and JOBS[kind]['process'].poll() is None: raise ValueError('此操作已在執行，請先停止或等待完成')
    ident=data.get('version','v001')
    v=versions.get_version(ident)
    workspace=versions.STORE/ident/'workspace'
    task=data.get('task')
    if task not in TASKS: raise ValueError('請選擇已註冊任務')
    cp=None
    if kind in ('play','export') or data.get('resume'):
        cp=log_path(data.get('checkpoint',''))
        cp_ident=data.get('checkpoint','').split('/')[0][1:] if data.get('checkpoint','').startswith('@') else 'v001'
        if cp_ident!=ident: raise ValueError('請先選擇與 checkpoint 相同的版本；跨版本遷移需要另行驗證')
        if cp.suffix!='.pt' or cp.parent.parent.name!=TASKS[task]: raise ValueError('checkpoint 與任務實驗目錄不匹配')
    if kind=='train':
        if v['kind']=='imported': raise ValueError('版本 1 是現有訓練的匯入記錄；請先複製為新版本再開訓')
        versions.verify(ident)
        if any(p['kind']=='train' for p in external_processes()): raise ValueError('已有訓練運行中；請先在原本的終端處理該訓練')
        n=int(data.get('num_envs',128)); it=int(data.get('iterations',1000))
        if not 1<=n<=4096 or not 1<=it<=100000: raise ValueError('環境數或輪數超出允許範圍')
        args=[str(PY),str(REPO/'.venv/bin/train'),task,'--env.scene.num-envs',str(n),'--agent.max-iterations',str(it),'--agent.logger','tensorboard']
        if cp: args += ['--agent.resume','--agent.load-run',cp.parent.name,'--agent.load-checkpoint',cp.name]
    elif kind=='play':
        if available(8093): raise ValueError('播放埠 8093 已被使用；現有播放仍可在左側觀看')
        args=[str(PY),str(ROOT/'play_local.py'),task,str(cp)]
    else:
        output=cp.with_name(cp.stem+'_studio.onnx')
        if output.exists(): raise ValueError('此 checkpoint 已有匯出檔，保留原檔，不覆寫')
        args=[str(PY),str(workspace/'scripts/export.py'),task,'--checkpoint-file',str(cp),'--onnx-file',str(output),'--num-envs','1','--device','cpu']
    log=ROOT/'runtime'/f'{kind}-{time.time_ns()}.log'
    with log.open('w') as f:
        p=subprocess.Popen(args,cwd=workspace,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONPATH':str(workspace/'src')+os.pathsep+str(workspace/'vendor/bam'),'PYTHONDONTWRITEBYTECODE':'1'})
    JOBS[kind]={'process':p,'log':str(log),'started':time.time(),'task':task,'checkpoint':str(cp) if cp else None,'version':ident}
    audit=ROOT/'runtime'/'actions.jsonl'
    with audit.open('a') as record: record.write(json.dumps({'kind':kind,'version':ident,'argv':args,'time':time.time(),'pid':p.pid})+'\n')
    return {'message':'已啟動；請查看執行日誌'}

class Handler(BaseHTTPRequestHandler):
    def send(self,data,status=200,ctype='application/json; charset=utf-8'):
        raw=data.encode() if isinstance(data,str) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(raw)
    def public_origin(self):
        p=ROOT/'remote.json'
        return json.loads(p.read_text()).get('origin') if p.exists() else None
    def trusted(self):
        hosts={f'localhost:{PORT}',f'127.0.0.1:{PORT}'}
        origin=self.public_origin()
        if origin:hosts.add(urlparse(origin).netloc)
        return self.headers.get('Host') in hosts
    def do_GET(self):
        if not self.trusted(): return self.send({'error':'Invalid host'},403)
        u=urlparse(self.path)
        try:
            if u.path=='/': return self.send((ROOT/'index.html').read_text().replace('__TOKEN__',TOKEN),ctype='text/html; charset=utf-8')
            if u.path=='/api/state':
                return self.send({'tasks':TASKS,'runs':runs(),'jobs':job_state(),'processes':external_processes(),
                        'viewer':available(8080),'managed_viewer':bool(JOBS.get('play') and JOBS['play']['process'].poll() is None and available(8093)),
                        'tensorboard':available(6006),'gpu':gpu(),'time':time.time(),'artifacts':[json.loads(p.read_text()) | {'id':p.parent.name} for p in sorted((ROOT/'artifacts').glob('*/result.json'))],'versions':[{**v,'manifest':{'digest':v['manifest']['digest']}} for v in versions.list_versions()]})
            if u.path=='/api/artifact':
                ident=parse_qs(u.query).get('id',[''])[0]
                if not ident.isdigit():raise ValueError('無效成果')
                folder=inside(ident,ROOT/'artifacts')
                import tempfile
                with tempfile.TemporaryFile() as buf:
                    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                        for f in folder.rglob('*'):
                            if f.is_file():z.write(f,str(f.relative_to(folder)))
                    size=buf.tell();buf.seek(0)
                    self.send_response(200);self.send_header('Content-Type','application/zip');self.send_header('Content-Disposition','attachment; filename="result-'+ident+'.zip"');self.send_header('Content-Length',str(size));self.end_headers();shutil.copyfileobj(buf,self.wfile)
                return
            if u.path=='/api/download':
                p=log_path(parse_qs(u.query).get('file',[''])[0])
                if p.suffix not in ('.pt','.onnx','.yaml','.json') or not p.is_file():raise ValueError('不支援此下載檔案')
                self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Disposition','attachment; filename="'+p.name+'"');self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
                with p.open('rb') as f:shutil.copyfileobj(f,self.wfile)
                return
            if u.path=='/api/recipe':
                ident=parse_qs(u.query).get('version',[''])[0];v=versions.get_version(ident)
                buf=io.BytesIO()
                with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                    for p in (versions.STORE/ident).rglob('*'):
                        if p.is_file() and 'logs' not in p.parts and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(versions.STORE/ident)))
                raw=buf.getvalue();self.send_response(200);self.send_header('Content-Type','application/zip');self.send_header('Content-Disposition','attachment; filename="'+ident+'-recipe.zip"');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
            if u.path=='/api/metrics':
                with METRIC_LOCK: return self.send(metrics(parse_qs(u.query).get('run',[''])[0]))
            return self.send({'error':'Not found'},404)
        except Exception as e: self.send({'error':str(e)},400)
    def do_POST(self):
        origin=self.headers.get('Origin')
        if not self.trusted() or self.headers.get('X-Studio-Token')!=TOKEN or origin not in (None,f'http://localhost:{PORT}',f'http://127.0.0.1:{PORT}',self.public_origin()):
            return self.send({'error':'請從本機控制台操作'},403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size>8192: raise ValueError('Request too large')
            data=json.loads(self.rfile.read(size))
            with LOCK:
                if self.path=='/api/version': return self.send(versions.create_version(data))
                if self.path=='/api/archive':
                    cp=log_path(data['checkpoint'])
                    if cp.suffix!='.pt': raise ValueError('請選擇 PT checkpoint')
                    v=versions.get_version(data['version'])
                    expected=data['checkpoint'].split('/')[0][1:] if data['checkpoint'].startswith('@') else 'v001'
                    if expected!=v['id']:raise ValueError('成果與版本不匹配')
                    target=ROOT/'artifacts'/str(time.time_ns());target.mkdir(parents=True)
                    shutil.copy2(cp,target/cp.name)
                    if (cp.parent/'params').exists():shutil.copytree(cp.parent/'params',target/'params')
                    for event in cp.parent.glob('events.out.tfevents*'):shutil.copy2(event,target/event.name)
                    for onnx in cp.parent.glob(cp.stem+'*.onnx'):shutil.copy2(onnx,target/onnx.name)
                    versions.write_json(target/'recipe.json',v)
                    versions.write_json(target/'result.json',{'version':v['id'],'checkpoint':str(cp),'sha256':versions.sha(target/cp.name),'saved_at':time.time(),'recipe_digest':v['manifest']['digest']})
                    return self.send({'message':'已封存 checkpoint、曲線及設定：'+target.name})
                if self.path=='/api/start': return self.send(start(data.pop('kind'),data))
                if self.path=='/api/stop':
                    j=JOBS.get(data.get('kind'))
                    if j and j['process'].poll() is None: os.killpg(j['process'].pid,signal.SIGINT)
                    return self.send({'message':'已請求停止本控制台啟動的程序'})
                return self.send({'error':'Not found'},404)
        except Exception as e: self.send({'error':str(e)},400)
    def log_message(self,*args): pass

if __name__=='__main__':
    versions.init_baseline()
    TASKS=discover_tasks()
    print(f'XgoDuck Studio: http://localhost:{PORT}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()