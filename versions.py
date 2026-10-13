"""Immutable experiment recipes; imported run evidence is kept distinct from source snapshots."""
import ast, hashlib, json, math, re, shutil, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent/'xgoduck_rl'
STORE=ROOT/'versions'
CONST=Path('src/mjlab_microduck/robot/xgoduck_constants.py')
ROBOT=Path('src/mjlab_microduck/robot/xgoduck')

def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_params(root):
    s=(root/CONST).read_text()
    def value(pattern): return float(re.search(pattern,s).group(1))
    stand=value(r'XGODUCK_STAND_Z = ([\d.]+)')
    pos=re.search(r'pos=\(0\.0, 0\.0, ([^)]+)\)',s).group(1)
    xml=ET.parse(root/ROBOT/'robot_walk.xml').find('.//body[@name="trunk_base"]/inertial')
    return {'home_deg':value(r'_DEG24 = math.radians\(([\d.]+)\)'),
            'kp':value(r'kp_fw=([\d.]+)'), 'com_x_mm':float(xml.attrib['pos'].split()[0])*1000,
            'root_z_mm':(stand if pos=='XGODUCK_STAND_Z' else float(pos))*1000,
            'stand_z_mm':stand*1000}

def snapshot(dest):
    dest.mkdir(parents=True)
    for name in ('src','scripts'):
        shutil.copytree(REPO/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Include the vendored actuator implementation as well as the dependency lock.
    shutil.copytree(REPO/'vendor',dest/'vendor',ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc'))
    for name in ('pyproject.toml','uv.lock','README.md'):
        shutil.copy2(REPO/name,dest/name)

def manifest(d):
    files={str(p.relative_to(d)):sha(p) for p in sorted(d.rglob('*')) if p.is_file() and 'logs' not in p.relative_to(d).parts and '__pycache__' not in p.relative_to(d).parts and p.suffix!='.pyc'}
    return {'files':files,'digest':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}

def list_versions():
    return [json.loads(p.read_text()) for p in sorted(STORE.glob('v*/recipe.json'))]

def get_version(ident):
    if not re.fullmatch(r'v\d{3}',ident or ''):raise ValueError('無效版本')
    p=STORE/ident/'recipe.json'
    if not p.exists():raise ValueError('版本不存在')
    return json.loads(p.read_text())

def init_baseline():
    STORE.mkdir(exist_ok=True)
    if list_versions():return
    d=STORE/'v001';source=d/'workspace';snapshot(source)
    evidence=d/'imported_run_params';evidence.mkdir()
    refs=[]
    for p in (REPO/'logs/rsl_rl').glob('*/*/params'):
        relative=str(p.parent.relative_to(REPO/'logs/rsl_rl'));refs.append(relative)
        target=evidence/p.parent.parent.name/p.parent.name
        shutil.copytree(p,target)
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True).stdout.strip()
    recipe={'id':'v001','name':'版本 1 · 現有訓練','parent':None,'created':time.time(),'kind':'imported',
        'params':read_params(source),'notes':'正在運行的現有訓練。source 是匯入當下的原始碼快照，未證明與此輪啟動時完全相同；實際訓練設定以 imported_run_params 為準。',
        'git_head':git,'imported_runs':refs,'manifest':manifest(source)}
    write_json(d/'recipe.json',recipe)

def create_version(data):
    base=get_version(data.get('parent','v001'))
    p={k:float(data['params'][k]) for k in ('home_deg','kp','com_x_mm','root_z_mm')}
    bounds={'home_deg':(0,45),'kp':(1,32),'com_x_mm':(-50,50),'root_z_mm':(0,250)}
    for k,(lo,hi) in bounds.items():
        if not math.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError(f'{k} 超出允許範圍')
    if p['kp']!=int(p['kp']):raise ValueError('KP 必須是整數')
    ident=f'v{max(int(v["id"][1:]) for v in list_versions())+1:03d}'
    d=STORE/ident;source=d/'workspace'
    shutil.copytree(STORE/base['id']/'workspace',source,ignore=shutil.ignore_patterns('logs','__pycache__','*.pyc'))
    try:
        const=source/CONST;s=const.read_text()
        s=re.sub(r'_DEG24 = math.radians\([\d.]+\)',f'_DEG24 = math.radians({p["home_deg"]})',s)
        s=re.sub(r'kp_fw=[\d.]+',f'kp_fw={p["kp"]}',s)
        s=re.sub(r'pos=\(0\.0, 0\.0, [^)]+\)',f'pos=(0.0, 0.0, {p["root_z_mm"]/1000})',s,count=1)
        const.write_text(s)
        for name in ('robot_walk.xml','robot_allcollisions.xml'):
            path=source/ROBOT/name;tree=ET.parse(path);inertia=tree.find('.//body[@name="trunk_base"]/inertial')
            xyz=inertia.attrib['pos'].split();xyz[0]=str(p['com_x_mm']/1000);inertia.set('pos',' '.join(xyz));tree.write(path,encoding='unicode')
        converter=source/ROBOT/'convert_from_urdf.py';s=converter.read_text()
        s=re.sub(r'(CAD_TRUNK_COM = \[)[^,]+',lambda m:m[1]+str(p['com_x_mm']/1000),s);converter.write_text(s)
        # Recompute height from actual foot mesh vertices and the new HOME pose.
        import mujoco, numpy as np
        tree=ast.parse(const.read_text())
        node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='XGODUCK_HOME_FRAME' for t in n.targets))
        pose=eval(compile(ast.Expression(next(k.value for k in node.keywords if k.arg=='joint_pos')),'pose','eval'),{'_DEG24':math.radians(p['home_deg'])})
        m=mujoco.MjModel.from_xml_path(str(source/ROBOT/'robot_walk.xml'));q=mujoco.MjData(m);q.qpos[:3]=0;q.qpos[3:7]=[1,0,0,0]
        for j in range(m.njnt):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
            for pattern,value in pose.items():
                if name and re.fullmatch(pattern,name):q.qpos[m.jnt_qposadr[j]]=value
        mujoco.mj_forward(m,q);zs=[]
        for name in ('ankle_left','ankle_right'):
            bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
            for g in np.where(m.geom_bodyid==bid)[0]:
                if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];n=m.mesh_vertnum[mid]
                verts=m.mesh_vert[a:a+n]@q.geom_xmat[g].reshape(3,3).T+q.geom_xpos[g];zs.append(float(verts[:,2].min()))
        stand=-min(zs)+.001
        s=const.read_text();s=re.sub(r'XGODUCK_STAND_Z = [\d.]+',f'XGODUCK_STAND_Z = {stand:.9f}',s);const.write_text(s)
        # Only these two tasks retain additive reset height. Other tasks overwrite it later.
        path=source/'src/mjlab_microduck/tasks/__init__.py';s=path.read_text()
        for fn in ('_xgoduck_velocity_cfg','_xgoduck_ground_pick_cfg'):
            begin=s.index('def '+fn+'(');end=s.index('\n    return cfg',begin)
            insert=f'\n    # Studio recipe: root position + reset offset, without adding height twice.\n    cfg.events["reset_base"].params["pose_range"]["z"] = (XGODUCK_STAND_Z - {p["root_z_mm"]/1000} + 0.004, XGODUCK_STAND_Z - {p["root_z_mm"]/1000} + 0.009)\n'
            s=s[:end]+insert+s[end:]
        path.write_text(s)
        p['stand_z_mm']=stand*1000
        recipe={'id':ident,'name':str(data.get('name',ident))[:100],'parent':base['id'],'created':time.time(),'kind':'recipe','params':p,
            'notes':str(data.get('notes',''))[:2000],'manifest':manifest(source),
            'validation':'已計算水平 HOME 腳底高度；隨機姿態、完整 reset 和動作表現仍需短訓練驗證。'}
        write_json(d/'recipe.json',recipe);return recipe
    except Exception:
        # Preserve failed snapshot for diagnosis; it is not listed or selectable without recipe.json.
        raise

def verify(ident):
    v=get_version(ident)
    if manifest(STORE/ident/'workspace')['digest']!=v['manifest']['digest']:raise ValueError('版本原始碼已被外部修改，請建立新版本後再訓練')
    return v