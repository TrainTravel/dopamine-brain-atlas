const D=window.ATLAS_DATA;
if(!D||!Array.isArray(D.studies)||!Array.isArray(D.nodes)){const m=document.createElement('p');m.className='err';m.id='dataerr';m.textContent='The atlas data did not load. Reload the page; if this persists the site files are incomplete.';document.querySelector('#stage').append(m);throw new Error('atlas data missing')}
const $=s=>document.querySelector(s);
const el=(t,a={},h)=>{const e=document.createElement(t);for(const k in a)e.setAttribute(k,a[k]);if(h!=null)e.innerHTML=h;return e};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const KEYS=D.keys, CVAR={ASD:'--asd',AD:'--ad',PD:'--pd',ADHD:'--adhd',DA:'--da'};
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const RX=[
 ['D1','D1-like, Gs/olf coupled, raises cAMP','<b>D1</b> is the most abundant dopamine receptor. Dense on direct-pathway neurons of the striatum and across prefrontal cortex, where working memory follows an inverted-U with D1 stimulation. Most relevant to Parkinson\'s (direct loop) and ADHD (prefrontal control).'],
 ['D2','D2-like, Gi/o coupled, lowers cAMP','<b>D2</b> sits on indirect-pathway striatal neurons, as an autoreceptor on nigral and VTA cells, and on pituitary lactotrophs. Main target of antipsychotics (used for irritability in autism) and of most Parkinson\'s agonists.'],
 ['D3','D2-like, Gi/o coupled','<b>D3</b> is concentrated in limbic territory: ventral striatum, globus pallidus, substantia nigra and hypothalamus. Tied to reward and motivation; D3-preferring agonists are linked to impulse-control problems in Parkinson\'s.'],
 ['D4','D2-like, Gi/o coupled','<b>D4</b> is mostly extrastriatal: prefrontal cortex, amygdala and hippocampus. The DRD4 exon-3 7-repeat allele is one of the most replicated genetic associations with ADHD.'],
 ['D5','D1-like, Gs coupled','<b>D5</b> is sparse overall but present in hippocampus, thalamus, hypothalamus and prefrontal cortex. Involved in hippocampal plasticity (relevant to memory loss in Alzheimer\'s models); a DRD5 marker is associated with ADHD.']];
const DIS=[
 ['PD',"Parkinson's",'Loss of substantia nigra neurons depletes dopamine in the putamen first, unbalancing the direct (D1) and indirect (D2) loops. Levodopa and D2/D3 agonists are standard treatment.'],
 ['ADHD','ADHD','Genetic studies implicate DRD4, DRD5 and the dopamine transporter. Stimulants raise synaptic dopamine in striatum and prefrontal cortex. fMRI often shows a weaker ventral striatal response when anticipating reward.'],
 ['ASD','Autism','The social motivation hypothesis proposes a weaker mesolimbic response to social reward; repetitive behaviour is linked to striatal circuits. D2-acting drugs are used for irritability. Findings are mixed.'],
 ['AD',"Alzheimer's",'Primarily a cholinergic and medial temporal disease. VTA degeneration and reduced hippocampal dopamine signalling have been reported in animal models and some human imaging; dopamine agonist trials are exploratory.']];

/* ---------- state ---------- */
const S={layers:{ASD:true,AD:true,PD:true,ADHD:true,DA:false},paths:{mesolimbic:true,mesocortical:true,nigrostriatal:true,tubero:true,direct:false,indirect:false},rx:-1,node:null,study:null,hit:null};
const NIDX={};D.nodes.forEach((n,i)=>NIDX[n.id]=i);

/* ---------- static UI ---------- */
$('#sub').textContent=`${D.studies.length.toLocaleString()} fMRI studies of autism, Alzheimer's, Parkinson's, ADHD and dopamine, mapped onto the dopaminergic pathways and D1 to D5 receptor territories`;
$('#tot').textContent=D.total.toLocaleString();
KEYS.forEach(k=>{const g=D.groups[k];const r=el('label',{class:'row'},`<input type="checkbox" id="ly_${k}" ${S.layers[k]?'checked':''}><span class="sw" style="background:var(${CVAR[k]})"></span><span class="nm">${g.name}</span><span class="ct">${g.n} studies</span>`);
  r.querySelector('input').onchange=e=>{S.layers[k]=e.target.checked;sync()};$('#layers').append(r)});
D.paths.forEach(p=>{const r=el('label',{class:'row'},`<input type="checkbox" id="pa_${p.id}" ${S.paths[p.id]?'checked':''}><span class="nm">${p.name}<small>${p.desc}</small></span>`);
  r.querySelector('input').onchange=e=>{S.paths[p.id]=e.target.checked;sync()};$('#paths').append(r)});
[['Off']].concat(RX).forEach((r,i)=>{const b=el('button',{type:'button','aria-pressed':i===0},r[0]);b.onclick=()=>{S.rx=i-1;sync()};$('#rseg').append(b)});
DIS.forEach(d=>$('#dis').append(el('p',{},`<span class="gd" style="background:var(${CVAR[d[0]]})"></span><b>${d[1]}.</b> ${d[2]}`)));
function cellBg(r){if(r>=1){const a=Math.min(1,(r-1)/1.2)*70;return `color-mix(in srgb,var(--da) ${a.toFixed(0)}%,transparent)`}const a=Math.min(1,(1-r)/0.6)*45;return `color-mix(in srgb,var(--cool) ${a.toFixed(0)}%,transparent)`}
function buildMatrix(){const t=$('#mx');t.innerHTML='';const h=el('tr');h.append(el('th',{},'Region'));KEYS.forEach(k=>h.append(el('th',{},`<span class="gd" style="background:var(${CVAR[k]})"></span>${k==='DA'?'DA':k}`)));t.append(h);
  D.nodes.forEach(n=>{if(n.base===0)return;const tr=el('tr',{'data-n':n.id});const td=el('td');const b=el('button',{type:'button',title:n.name},n.id);b.onclick=()=>selectNode(n.id);td.append(b);tr.append(td);
    KEYS.forEach(k=>{const e=n.enr[k];const c=el('td');const bb=el('button',{type:'button',class:e.p<0.05?'sig':'',style:`background:${cellBg(e.ratio)}`,title:`${D.groups[k].name}: ${e.hit} of ${e.n} studies (${e.pct}%) vs ${n.base}% overall, p=${e.p}`},e.ratio.toFixed(1));bb.onclick=()=>selectNode(n.id);c.append(bb);tr.append(c)});t.append(tr)})}
buildMatrix();

/* ---------- detail panel ---------- */
function studyRow(s,i){const li=el('li',S.study===i?{class:'on'}:{});const dots=KEYS.filter((k,j)=>s[5]>>j&1).map(k=>`<span class="gd" style="background:var(${CVAR[k]})" title="${D.groups[k].name}"></span>`).join('');
  const b=el('button',{type:'button'},`<span class="t">${dots}${esc(s[1])}</span><span class="m">${esc(s[4])} ${s[2]} · ${esc(s[3])} · ${s[7].length/3} peaks</span>`);b.onclick=()=>selectStudy(i);
  li.append(b,el('a',{href:`https://pubmed.ncbi.nlm.nih.gov/${s[0]}/`,target:'_blank',rel:'noopener'},'PubMed'));return li}
function renderDetail(){const d=$('#detail');d.innerHTML='';
  document.querySelectorAll('#mx tr').forEach(tr=>tr.classList.toggle('on',tr.dataset.n===S.node));
  if(!S.node&&S.study==null&&!S.hit){d.hidden=true;return}d.hidden=false;
  const close=el('button',{type:'button',class:'btn',style:'align-self:flex-end'},'Clear selection');close.onclick=()=>{S.node=null;S.study=null;S.hit=null;sync()};
  if(S.hit){const h=S.hit,near=[];for(const n of D.nodes){for(let q=0;q<h.peaks.length;q+=3){const dx=Math.abs(h.peaks[q])-n.xyz[0],dy=h.peaks[q+1]-n.xyz[1],dz=h.peaks[q+2]-n.xyz[2];if(dx*dx+dy*dy+dz*dz<=D.radius*D.radius){near.push(n.id);break}}}
    d.append(el('h2',{},'Selected study'),el('h3',{},esc(h.title)),
      el('p',{class:'note'},`${esc(h.author)} (${h.year||'year unknown'}). ${esc(h.journal)}. ${h.peaks.length/3} peaks, ringed on the brain. <a href="https://pubmed.ncbi.nlm.nih.gov/${h.pmid}/" target="_blank" rel="noopener">Open in PubMed</a>`),
      el('p',{class:'note'},`<b>Layers:</b> not in the five study layers (found by search)<br><b>Peaks within 10 mm of:</b> ${near.length?near.join(', '):'none of the mapped regions'}`))}
  if(S.study!=null){const s=D.studies[S.study];d.append(el('h2',{},'Selected study'),el('h3',{},esc(s[1])),
    el('p',{class:'note'},`${esc(s[4])} (${s[2]}). ${esc(s[3])}. ${s[7].length/3} peaks, ringed on the brain. <a href="https://pubmed.ncbi.nlm.nih.gov/${s[0]}/" target="_blank" rel="noopener">Open in PubMed</a>`));
    const near=D.nodes.filter((n,j)=>s[6]>>j&1).map(n=>n.id);
    d.append(el('p',{class:'note'},`<b>Layers:</b> ${KEYS.filter((k,j)=>s[5]>>j&1).map(k=>D.groups[k].name).join(', ')}<br><b>Peaks within 10 mm of:</b> ${near.length?near.join(', '):'none of the mapped regions'}`))}
  if(S.node){const n=D.nodes[NIDX[S.node]];const ni=NIDX[S.node];
    d.append(el('h2',{},'Selected region'),el('h3',{},n.name));
    const pw=D.paths.filter(p=>p.edges.some(e=>e.includes(n.id))).map(p=>p.name).join(', ');
    d.append(el('dl',{class:'kv'},`<dt>MNI centre</dt><dd class="mono">${n.mid?'':'±'}${n.xyz[0]}, ${n.xyz[1]}, ${n.xyz[2]}</dd><dt>Pathways</dt><dd>${pw||'none'}</dd>`));
    const rx=el('div',{class:'rx'});RX.forEach((r,i)=>{rx.innerHTML+=`<b>${r[0]}</b><span class="dots">${[1,2,3].map(j=>`<i class="${n.expr[i]>=j?'f':''}"></i>`).join('')}</span><span>${['not reported','low','moderate','high'][n.expr[i]]}</span>`});
    d.append(el('h2',{},'Receptor density (qualitative)'),rx);
    if(n.base>0){d.append(el('h2',{},'Studies with a peak within 10 mm'));
      KEYS.forEach(k=>{const e=n.enr[k];const max=Math.max(40,...KEYS.map(q=>n.enr[q].pct));
        d.append(el('div',{class:'bar'},`<span>${D.groups[k].name.replace(' studies','')}</span><span class="track"><span class="fill" style="width:${e.pct/max*100}%;background:var(${CVAR[k]})"></span><span class="tick" style="left:${n.base/max*100}%"></span></span><span class="num">${e.pct}% · ${e.ratio.toFixed(1)}×</span>`))});
      d.append(el('p',{class:'note'},`Black tick: ${n.base}% of all Neurosynth studies.`));
      const list=[];D.studies.forEach((s,i)=>{if(s[6]>>ni&1&&KEYS.some((k,j)=>S.layers[k]&&s[5]>>j&1))list.push(i)});
      list.sort((a,b)=>D.studies[b][2]-D.studies[a][2]);
      d.append(el('h2',{},`${list.length} studies in the active layers`));
      const ul=el('ul',{class:'st'});list.slice(0,60).forEach(i=>ul.append(studyRow(D.studies[i],i)));d.append(ul);
      if(list.length>60)d.append(el('p',{class:'note'},`Showing the 60 most recent of ${list.length}.`));
    } else d.append(el('p',{class:'note'},'The pituitary lies outside the standard fMRI brain mask, so no peaks are reported here. It is shown as the endpoint of the tuberoinfundibular pathway.'))}
  d.append(close)}
function selectNode(id){S.node=S.node===id?null:id;S.study=null;S.hit=null;sync();if(S.node&&matchMedia('(max-width:1000px)').matches===false)$('#right').scrollTop=0}
function selectStudy(i){S.study=S.study===i?null:i;S.hit=null;sync()}
// A search result: studies already in the atlas layers use the normal study view, others get a lightweight card.
const PMID_INDEX=new Map(D.studies.map((s,i)=>[s[0],i]));
function selectHit(h){const i=PMID_INDEX.get(h.pmid);S.node=null;if(i!==undefined){S.study=i;S.hit=null}else{S.study=null;S.hit=h}sync();if(!matchMedia('(max-width:1000px)').matches)$('#right').scrollTop=0}

/* ---------- 3D ---------- */
let three=null;
function init3D(){
  const stage=$('#stage');const T=THREE;
  const renderer=new T.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));stage.prepend(renderer.domElement);
  const scene=new T.Scene();const cam=new T.PerspectiveCamera(35,1,1,2000);
  const ctl=new T.OrbitControls(cam,renderer.domElement);ctl.enableDamping=true;ctl.dampingFactor=.12;ctl.target.set(0,8,12);ctl.minDistance=80;ctl.maxDistance=700;
  scene.add(new T.AmbientLight(0xffffff,.75));const dl=new T.DirectionalLight(0xffffff,.55);cam.add(dl);dl.position.set(0,0,1);scene.add(cam);
  const P=(x,y,z)=>new T.Vector3(x,z,-y); // MNI (R,A,S) -> three (x,y,z), right-handed
  const b64=(s,C)=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new C(u.buffer)};
  // cortex
  const hemi={};const cortexMat=new T.MeshLambertMaterial({vertexColors:true,transparent:true,opacity:.16,depthWrite:false});
  for(const h of ['L','R']){const m=D.mesh[h],v=b64(m.v,Int16Array),f=b64(m.f,Uint16Array),s=b64(m.s,Uint8Array);const n=v.length/3;
    const pos=new Float32Array(n*3),col=new Float32Array(n*3);
    for(let i=0;i<n;i++){pos[i*3]=v[i*3]/10;pos[i*3+1]=v[i*3+2]/10;pos[i*3+2]=-v[i*3+1]/10;const c=1-.55*(s[i]/255);col[i*3]=col[i*3+1]=col[i*3+2]=c}
    const g=new T.BufferGeometry();g.setAttribute('position',new T.BufferAttribute(pos,3));g.setAttribute('color',new T.BufferAttribute(col,3));g.setIndex(new T.BufferAttribute(f,1));g.computeVertexNormals();
    const mesh=new T.Mesh(g,cortexMat);mesh.renderOrder=10;scene.add(mesh);hemi[h]=mesh}
  // dot sprite
  const cv=document.createElement('canvas');cv.width=cv.height=64;const cx=cv.getContext('2d');cx.fillStyle='#fff';cx.beginPath();cx.arc(32,32,30,0,7);cx.fill();const disc=new T.CanvasTexture(cv);
  const cv2=document.createElement('canvas');cv2.width=cv2.height=64;const c2=cv2.getContext('2d');c2.strokeStyle='#fff';c2.lineWidth=9;c2.beginPath();c2.arc(32,32,25,0,7);c2.stroke();const ring=new T.CanvasTexture(cv2);
  // study layers
  const layers={};KEYS.forEach((k,j)=>{const nr=[],fr=[];D.studies.forEach((s,i)=>{if(s[5]>>j&1){const c=s[7];for(let q=0;q<c.length;q+=3){const x=c[q],y=c[q+1],z=c[q+2];
      const isNear=D.nodes.some(n=>{const dy=y-n.xyz[1],dz=z-n.xyz[2],dx=Math.abs(x)-n.xyz[0];return dx*dx+dy*dy+dz*dz<=D.radius*D.radius});(isNear?nr:fr).push([x,z,-y,i])}}});
    const all=nr.concat(fr),pos=[],idx=[];all.forEach(p=>{pos.push(p[0],p[1],p[2]);idx.push(p[3])});
    const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(pos,3));
    const pts=new T.Points(g,new T.PointsMaterial({size:2.6,map:disc,alphaTest:.5,sizeAttenuation:true}));pts.userData={idx,key:k,nearN:nr.length};scene.add(pts);layers[k]=pts});
  const hiGeo=new T.BufferGeometry();hiGeo.setAttribute('position',new T.Float32BufferAttribute([],3));
  const hi=new T.Points(hiGeo,new T.PointsMaterial({size:9,map:ring,alphaTest:.3,sizeAttenuation:true,depthTest:false}));hi.renderOrder=20;scene.add(hi);
  const sGeo=new T.BufferGeometry();sGeo.setAttribute('position',new T.Float32BufferAttribute([],3));
  const sPts=new T.Points(sGeo,new T.PointsMaterial({size:3.6,map:disc,alphaTest:.5,sizeAttenuation:true}));sPts.userData.own=[];sPts.visible=false;sPts.renderOrder=15;scene.add(sPts);
  function setSearch(pos,own){sGeo.setAttribute('position',new T.Float32BufferAttribute(pos,3));sPts.userData.own=own;sPts.visible=pos.length>0;dirty=true}
  // nodes
  const nodeMeshes=[],labels=[];const sph=new T.SphereGeometry(1,20,14);
  D.nodes.forEach((n,i)=>{const sides=n.mid?[1]:[1,-1];n._m=[];sides.forEach(sg=>{const m=new T.Mesh(sph,new T.MeshLambertMaterial({transparent:true}));m.position.copy(P(n.xyz[0]*sg,n.xyz[1],n.xyz[2]));m.userData.node=n.id;scene.add(m);nodeMeshes.push(m);n._m.push(m)});
    const l=el('div',{class:'lbl'},n.id);stage.append(l);labels.push([l,n._m[0]])});
  const halo=[0,1].map(()=>{const m=new T.Mesh(new T.SphereGeometry(D.radius,24,16),new T.MeshBasicMaterial({transparent:true,opacity:.14,depthWrite:false}));m.visible=false;m.renderOrder=5;scene.add(m);return m});
  // pathways
  const pathGroups={};const pathMat=new T.MeshLambertMaterial();const pathMat2=new T.MeshLambertMaterial({transparent:true,opacity:.6});
  D.paths.forEach(p=>{const g=new T.Group();p.edges.forEach(([a,b])=>{const A=D.nodes[NIDX[a]],B=D.nodes[NIDX[b]];
      const sides=(A.mid&&B.mid)?[1]:[1,-1];sides.forEach(sg=>{const pa=P(A.xyz[0]*sg,A.xyz[1],A.xyz[2]),pb=P(B.xyz[0]*sg,B.xyz[1],B.xyz[2]);
        const mid=pa.clone().add(pb).multiplyScalar(.5);const len=pa.distanceTo(pb);const out=mid.clone().sub(new T.Vector3(0,-5,5));out.x=0;out.normalize();
        const lift=p.id==='indirect'?-.12:(len>45?.22:.1);mid.addScaledVector(out,len*lift);
        const tube=new T.Mesh(new T.TubeGeometry(new T.QuadraticBezierCurve3(pa,mid,pb),24,p.id==='direct'||p.id==='indirect'?.5:.8,8),p.id==='indirect'?pathMat2:pathMat);g.add(tube);
        const dir=pb.clone().sub(mid).normalize();const cone=new T.Mesh(new T.ConeGeometry(1.7,4.5,10),p.id==='indirect'?pathMat2:pathMat);cone.position.copy(pb).addScaledVector(dir,-5.5);cone.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),dir);g.add(cone)})});
    scene.add(g);pathGroups[p.id]=g});
  // views
  const views={lat:[330,30,-10],top:[0,360,-8],front:[0,40,-340]};
  function setView(v){const p=views[v];cam.position.set(p[0]+ctl.target.x,p[1]+ctl.target.y,p[2]+ctl.target.z);ctl.update();dirty=true}
  document.querySelectorAll('.views button').forEach(b=>b.onclick=()=>setView(b.dataset.v));
  cam.position.set(250,110,-190);
  let dirty=true;
  function theme(){KEYS.forEach(k=>layers[k].material.color.set(cssv(CVAR[k])));hi.material.color.set(cssv('--fg'));sPts.material.color.set(cssv('--fg'));pathMat.color.set(cssv('--da'));pathMat2.color.set(cssv('--da'));halo.forEach(h=>h.material.color.set(cssv('--da')));cortexMat.color.set(cssv('--cortex'));apply()}
  function apply(){
    KEYS.forEach(k=>{layers[k].visible=S.layers[k];layers[k].geometry.setDrawRange(0,$('#near').checked?layers[k].userData.nearN:Infinity)});D.paths.forEach(p=>pathGroups[p.id].visible=S.paths[p.id]);
    cortexMat.opacity=+$('#op').value/100;hemi.L.visible=!$('#hideL').checked&&cortexMat.opacity>0;hemi.R.visible=cortexMat.opacity>0;
    const da=new T.Color(cssv('--da')),nd=new T.Color(cssv('--node'));
    D.nodes.forEach(n=>{const e=S.rx<0?null:n.expr[S.rx];const sel=S.node===n.id;
      const r=e==null?(n.role==='source'?3.2:2.6):1.4+e*1.5;
      n._m.forEach(m=>{m.scale.setScalar(sel?r*1.25:r);m.material.color.copy(e==null?(n.role==='source'||sel?da:nd):(e>0?da:nd));m.material.opacity=e===0?.35:1})});
    const sn=S.node?D.nodes[NIDX[S.node]]:null;halo.forEach((h,i)=>{h.visible=!!sn&&i<(sn?sn._m.length:0);if(h.visible)h.position.copy(sn._m[i].position)});
    const s=S.study!=null?D.studies[S.study]:null;const pk=s?s[7]:S.hit?S.hit.peaks:null;const hp=[];if(pk)for(let q=0;q<pk.length;q+=3)hp.push(pk[q],pk[q+2],-pk[q+1]);
    hiGeo.setAttribute('position',new T.Float32BufferAttribute(hp,3));hi.visible=!!pk;
    labels.forEach(l=>l[0].hidden=!$('#lab').checked);dirty=true}
  function resize(){const w=stage.clientWidth,h=stage.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();dirty=true}
  new ResizeObserver(resize).observe(stage);resize();
  const v3=new T.Vector3();
  function frame(){requestAnimationFrame(frame);if(ctl.update())dirty=true;if(!dirty)return;dirty=false;renderer.render(scene,cam);
    const w=stage.clientWidth,h=stage.clientHeight;
    if($('#lab').checked)labels.forEach(([l,m])=>{v3.copy(m.position).project(cam);const x=(v3.x+1)/2*w+6,y=(1-v3.y)/2*h-14;l.style.transform=`translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;l.style.opacity=v3.z<1?1:0});
    const d=cam.position.clone().sub(ctl.target);const ax=Math.abs(d.x),ay=Math.abs(d.y),az=Math.abs(d.z);
    $('#orient').textContent='Viewing from '+(ay>ax&&ay>az?(d.y>0?'above':'below'):ax>az?(d.x>0?'the right':'the left'):(d.z<0?'the front':'the back'))}
  frame();
  // picking
  const ray=new T.Raycaster();ray.params.Points.threshold=1.6;const m2=new T.Vector2();let down=null;
  const setRay=e=>{const r=renderer.domElement.getBoundingClientRect();m2.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(m2,cam);return r};
  renderer.domElement.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY]);
  renderer.domElement.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;setRay(e);
    const nh=ray.intersectObjects(nodeMeshes)[0];if(nh){selectNode(nh.object.userData.node);return}
    const ph=ray.intersectObjects(KEYS.filter(k=>S.layers[k]).map(k=>layers[k]).concat(sPts.visible?[sPts]:[]));if(ph.length){ph.sort((a,b)=>a.distanceToRay-b.distanceToRay);const best=ph.filter(p=>p.distanceToRay<1.6).sort((a,b)=>a.distance-b.distance)[0]||ph[0];
      if(best.object===sPts){const f=window.__atlas.onSearchPick;if(f)f(sPts.userData.own[best.index]);return}
      S.study=best.object.userData.idx[best.index];S.node=null;S.hit=null;sync();if(!matchMedia('(max-width:1000px)').matches)$('#right').scrollTop=0}});
  renderer.domElement.addEventListener('pointermove',e=>{if(e.buttons)return;const r=setRay(e);const nh=ray.intersectObjects(nodeMeshes)[0];const hv=$('#hover');
    if(nh){const n=D.nodes[NIDX[nh.object.userData.node]];hv.textContent=n.name;hv.hidden=false;hv.style.left=(e.clientX-r.left+12)+'px';hv.style.top=(e.clientY-r.top+12)+'px';renderer.domElement.style.cursor='pointer'}else{hv.hidden=true;renderer.domElement.style.cursor='grab'}});
  renderer.domElement.addEventListener('pointerleave',()=>$('#hover').hidden=true);
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',theme);
  new MutationObserver(theme).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  three={apply,theme,setSearch,searchPts:sPts,layers,hi,halo,nodeMeshes,pathGroups,hemi,renderer,cam,ctl,scene,render(){renderer.render(scene,cam)}};theme();
}
function sync(){
  document.querySelectorAll('#rseg button').forEach((b,i)=>b.setAttribute('aria-pressed',i===S.rx+1));
  $('#rnote').innerHTML=S.rx<0?'Pick a receptor to size each region by how densely it expresses that subtype.':`${RX[S.rx][2]} <br><span class="mono">${RX[S.rx][1]}</span>`;
  renderDetail();if(three)three.apply()}
['op','hideL','lab','near'].forEach(id=>$('#'+id).addEventListener('input',()=>three&&three.apply()));
try{if(!window.THREE||!THREE.OrbitControls)throw new Error('3D library did not load');init3D()}
catch(e){$('#stage').append(el('div',{class:'err'},'The 3D view could not start in this browser ('+esc(e.message)+'). The tables on the right still work.'))}
S.node='PUT';sync();
window.__atlas={S,D,get three(){return three},selectNode,selectStudy,selectHit,sync,onSearchPick:null};
