// FRONTEND FOTO + CAMERA -> backend Python /api
const API = "";
const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('overlay');
const photo = $('photo'), photoCanvas = $('photoCanvas');
const statusEl = $('status'), logEl = $('log'), jsonPreview = $('jsonPreview');
const badge = $('cadBadge'), nameInput = $('nameInput');
const fileInput = $('fileInput'), fileName = $('fileName');
const btnAnalyzePhoto = $('btnAnalyzePhoto'), btnRegisterPhoto = $('btnRegisterPhoto'), btnRecognizePhoto = $('btnRecognizePhoto');
const btnCam = $('btnCam'), btnRegister = $('btnRegister'), btnAuth = $('btnAuth');
const btnJson = $('btnJson'), btnClear = $('btnClear');
const gallery = $('gallery'), galInfo = $('galInfo'), galCount = $('galCount');

let modelsLoaded = false, cameraOn = false;
let authInterval = null, authActive = false;
let cadastro = null;
let lastPhoto = null; // {det, payload}

function log(m){ logEl.textContent = m; console.log(m); }
function setStatus(t,c){ statusEl.textContent = t; statusEl.className = c; }

async function ensureModels(){
  if(modelsLoaded) return true;
  log('Carregando IA...'); setStatus('CARREGANDO IA...','scanning');
  const URLS = [
    'https://justadudewhohacks.github.io/face-api.js/models',
    'https://venus714.github.io/Face-Recognition-Using-Face-API/models'
  ];
  for(const u of URLS){
    try{
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(u),
        faceapi.nets.faceLandmark68Net.loadFromUri(u),
        faceapi.nets.faceRecognitionNet.loadFromUri(u),
        faceapi.nets.faceExpressionNet.loadFromUri(u),
        faceapi.nets.ageGenderNet.loadFromUri(u)
      ]);
      modelsLoaded = true; log('IA pronta. Envie uma foto.'); return true;
    }catch(e){ console.warn(e); }
  }
  setStatus('ERRO AO CARREGAR IA','denied'); return false;
}

async function apiGetStatus(){
  try{
    const r = await fetch(API + "/api/status");
    const j = await r.json();
    cadastro = j.gravado ? await (await fetch(API + "/api/cadastro")).json() : null;
  }catch(e){ cadastro = null; log("Backend offline? Rode: python backend/app.py"); }
  refreshUI();
}

function refreshUI(){
  if(cadastro){
    badge.className = 'yes';
    badge.textContent = '✅ GRAVADO: ' + (cadastro.owner||'').toUpperCase();
    jsonPreview.textContent = JSON.stringify(cadastro,null,2);
    btnJson.disabled = false;
    // NÃO mexe no campo nome: cada foto tem seu próprio nome opcional
    btnRecognizePhoto.disabled = !lastPhoto;
    btnAuth.disabled = !cameraOn;
    btnAuth.textContent = cameraOn ? '🔍 Reconhecer câmera' : '🔒 Reconhecer câmera';
    if(!cameraOn && lastPhoto) setStatus('ROSTO GRAVADO ✔ • ANALISE OUTRA FOTO PARA TESTAR','granted');
  }else{
    badge.className = 'no';
    badge.textContent = '⛔ SEM ROSTO GRAVADO';
    btnJson.disabled = true;
    btnAuth.disabled = true;
    btnAuth.textContent = '🔒 Reconhecer câmera (bloqueado)';
    btnRecognizePhoto.disabled = true; // TRAVA: sem cadastro nao reconhece
    if(!lastPhoto){ setStatus('ENVIE UMA FOTO PARA COMEÇAR','blocked'); jsonPreview.textContent='// Envie uma foto com rosto...'; }
    else { setStatus('FOTO ANALISADA • GRAVE PARA LIBERAR RECONHECIMENTO','blocked'); }
  }
  btnRegisterPhoto.disabled = !lastPhoto;
  btnAnalyzePhoto.disabled = !photo.src;
}
nameInput.addEventListener('input', ()=> nameInput.dataset.touched = '1');

// abas
function showPane(which){
  ['Foto','Cam','Galeria'].forEach(t=>{
    $('tab'+t).classList.toggle('active', t===which);
    $('pane'+t).hidden = (t!==which);
  });
  if(which==='Galeria') carregarGaleria();
}
$('tabFoto').onclick = ()=>showPane('Foto');
$('tabCam').onclick = ()=>showPane('Cam');
$('tabGaleria').onclick = ()=>showPane('Galeria');

// upload foto
fileInput.onchange = async ()=>{
  const f = fileInput.files[0]; if(!f) return;
  fileName.textContent = f.name;
  photo.src = URL.createObjectURL(f);
  lastPhoto = null;
  log('Foto carregada. Clique em Analisar foto.');
  btnAnalyzePhoto.disabled = false;
  refreshUI();
};

photo.onload = ()=>{ // limpa canvas ao trocar foto
  photoCanvas.getContext('2d').clearRect(0,0,photoCanvas.width,photoCanvas.height);
};

function updatePills(det, sim, dist){
  const ex = det.expressions ? Object.entries(det.expressions).sort((a,b)=>b[1]-a[1])[0] : null;
  $('pillFace').textContent='Rosto: ✓ '+Math.round(det.detection.score*100)+'%';
  $('pillExpr').textContent='Expressão: '+(ex?ex[0]+' '+Math.round(ex[1]*100)+'%':'-');
  $('pillAge').textContent='Idade: '+Math.round(det.age)+' anos';
  $('pillGen').textContent='Gênero: '+det.gender+' ('+Math.round(det.genderProbability*100)+'%)';
  $('pillScore').textContent = (sim!==undefined && sim!==null) ? ('Similaridade: '+sim+'% dist '+dist) : 'Similaridade: -';
}

// ---- GALERIA: foto + nome + análise em tempo real ----
function makeThumb(imgEl, maxW){
  maxW = maxW || 240;
  const c = document.createElement('canvas');
  const r = maxW / imgEl.naturalWidth;
  c.width = maxW; c.height = Math.round(imgEl.naturalHeight * r);
  c.getContext('2d').drawImage(imgEl, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.72);
}

async function salvarNaGaleria(tipo, extra){
  if(!lastPhoto) return;
  extra = extra || {};
  // Nome OPCIONAL por foto: usa o que a pessoa digitou, senão "Sem nome"
  const nome = ((nameInput.value || '').trim() || (lastPhoto.payload.owner || '').trim() || 'Sem nome').slice(0,40);
  const ex = lastPhoto.det.expressions ? Object.entries(lastPhoto.det.expressions).sort((a,b)=>b[1]-a[1])[0] : ['-',0];
  try{
    await fetch(API + "/api/historico", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        nome, tipo,
        foto: makeThumb(photo, 240),
        idade: Math.round(lastPhoto.det.age),
        genero: lastPhoto.det.gender,
        expressao: ex[0] + ' ' + Math.round(ex[1]*100) + '%',
        reconhecido: extra.reconhecido !== undefined ? extra.reconhecido : null,
        similaridade: extra.similaridade !== undefined ? extra.similaridade : null,
        distancia: extra.distancia !== undefined ? extra.distancia : null
      })
    });
    await carregarGaleria(); // tempo real: atualiza a aba na hora
  }catch(e){ console.warn('galeria', e); }
}

async function carregarGaleria(){
  try{
    const r = await fetch(API + "/api/historico");
    const lista = await r.json();
    galCount.textContent = lista.length;
    if(!lista.length){ galInfo.textContent = 'Nenhuma análise salva ainda. Analise uma foto!'; gallery.innerHTML=''; return; }
    galInfo.textContent = lista.length + ' análise(s) — mais recente primeiro (tempo real)';
    gallery.innerHTML = '';
    lista.forEach(item=>{
      const div = document.createElement('div');
      div.className = 'gcard';
      let tag = "<span class='gtag info'>ANALISADO</span>";
      if(item.tipo==='gravado') tag = "<span class='gtag ok'>GRAVADO ✔</span>";
      else if(item.tipo==='reconhecimento' && item.reconhecido) tag = "<span class='gtag ok'>BEM-VINDO ✔</span>";
      else if(item.tipo==='reconhecimento') tag = "<span class='gtag no'>NEGADO ⛔</span>";
      div.innerHTML =
        (item.foto ? "<img src='"+item.foto+"' />" : "") +
        "<div class='gbody'>" + tag +
        "<p class='gname'>"+escapeHtml(item.nome||'')+"</p>" +
        "<div class='gmeta'>Idade: "+(item.idade??'-')+" • "+(item.genero??'-')+"<br>Expressão: "+(item.expressao??'-')+
        (item.similaridade!==null&&item.similaridade!==undefined ? "<br>Similaridade: "+item.similaridade+"% (dist "+item.distancia+")" : "")+
        "<br><small>"+new Date(item.createdAt).toLocaleString() +"</small></div>" +
        "<button class='gdel' data-id='"+item.id+"'>excluir</button></div>";
      gallery.appendChild(div);
    });
    gallery.querySelectorAll('.gdel').forEach(b=> b.onclick = async ()=>{
      await fetch(API+"/api/historico/"+b.dataset.id, {method:"DELETE"});
      carregarGaleria();
    });
  }catch(e){ galInfo.textContent = 'Backend offline.'; }
}
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
$('btnClearGal').onclick = async ()=>{ await fetch(API+"/api/historico",{method:"DELETE"}); carregarGaleria(); };

// ANALISAR FOTO — leitura facial completa sem câmera
btnAnalyzePhoto.onclick = async ()=>{
  if(!photo.src){ log('Escolha uma foto primeiro.'); return; }
  if(!await ensureModels()) return;
  setStatus('ANALISANDO FOTO...','scanning'); log('Analisando rosto da foto...');
  try{
    const det = await faceapi.detectSingleFace(photo, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks().withFaceDescriptor().withFaceExpressions().withAgeAndGender();
    // desenha
    const dims = { width: photo.naturalWidth, height: photo.naturalHeight };
    photoCanvas.width = photo.clientWidth; photoCanvas.height = photo.clientHeight;
    const ctx = photoCanvas.getContext('2d'); ctx.clearRect(0,0,photoCanvas.width,photoCanvas.height);
    if(!det){ setStatus('NENHUM ROSTO NA FOTO ⛔','denied'); log('Tente outra foto: de frente, boa luz, 1 pessoa.'); lastPhoto=null; refreshUI(); return; }
    const scaleX = photo.clientWidth / dims.width, scaleY = photo.clientHeight / dims.height;
    const box = det.detection.box;
    ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 3;
    ctx.strokeRect(box.x*scaleX, box.y*scaleY, box.width*scaleX, box.height*scaleY);
    det.landmarks.positions.forEach(p=>{ ctx.fillStyle='#38bdf8'; ctx.beginPath(); ctx.arc(p.x*scaleX,p.y*scaleY,2,0,7); ctx.fill(); });

    const exS = Object.entries(det.expressions).sort((a,b)=>b[1]-a[1]);
    const typedName = (nameInput.value || '').trim() || 'Sem nome';
    const payload = {
      owner: typedName,
      descriptor: Array.from(det.descriptor),
      age: +det.age.toFixed(1), gender: det.gender, genderProbability: det.genderProbability,
      expressions: det.expressions,
      dominantExpression: { name: exS[0][0], probability: exS[0][1] },
      detection: { score: det.detection.score },
      landmarks: det.landmarks.positions.map(p=>({x:+p.x.toFixed(1), y:+p.y.toFixed(1)}))
    };
    lastPhoto = { det, payload };
    updatePills(det, null, null);
    jsonPreview.textContent = JSON.stringify({...payload, descriptor:`[${payload.descriptor.length} numeros - embedding]`}, null, 2);
    setStatus('FOTO ANALISADA ✔ • '+exS[0][0].toUpperCase()+' • '+Math.round(det.age)+' ANOS','idle');
    log('Leitura pronta. Salvando na galeria em tempo real...');
    refreshUI();
    if(cadastro) btnRecognizePhoto.disabled = false;
    await salvarNaGaleria('analise');
  }catch(e){ console.error(e); setStatus('ERRO NA ANÁLISE','denied'); }
};

// GRAVAR foto no backend Python + baixa JSON
btnRegisterPhoto.onclick = async ()=>{
  if(!lastPhoto){ log('Analise a foto primeiro.'); return; }
  const typed = (nameInput.value || '').trim();
  lastPhoto.payload.owner = typed || 'Sem nome';
  const slug = (lastPhoto.payload.owner.toLowerCase().replace(/\s+/g,'-') || 'sem-nome').slice(0,30);
  const r = await fetch(API+"/api/gravar",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(lastPhoto.payload)});
  const j = await r.json();
  if(!r.ok){ log(j.erro||'Erro'); return; }
  downloadJSON({...lastPhoto.payload, createdAt:new Date().toISOString()}, 'embedding-'+slug+'.json');
  log('Rosto GRAVADO no backend Python + JSON baixado!');
  await salvarNaGaleria('gravado');
  await apiGetStatus();
  setStatus('ROSTO GRAVADO ✔ • ENVIE OUTRA FOTO E CLIQUE RECONHECER','granted');
};

// RECONHECER foto — bloqueado se sem cadastro
btnRecognizePhoto.onclick = async ()=>{
  if(!cadastro){ setStatus('⛔ ACESSO NEGADO • NENHUM ROSTO GRAVADO','denied'); alert('Grave um rosto primeiro!'); return; }
  if(!lastPhoto){ log('Analise uma foto primeiro.'); return; }
  setStatus('RECONHECENDO VIA PYTHON...','scanning');
  const resp = await fetch(API+"/api/reconhecer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({descriptor:lastPhoto.payload.descriptor})});
  const j = await resp.json();
  if(resp.status===403 || j.bloqueado){ setStatus(j.mensagem,'denied'); await apiGetStatus(); return; }
  updatePills(lastPhoto.det, j.similaridade, j.distancia);
  setStatus(j.mensagem, j.reconhecido ? 'granted' : 'denied');
  log(j.reconhecido ? ('Match: '+j.owner+' '+j.similaridade+'%') : ('Sem match. dist '+j.distancia));
  await salvarNaGaleria('reconhecimento', { reconhecido: j.reconhecido, similaridade: j.similaridade, distancia: j.distancia });
};

function downloadJSON(o,f){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,2)],{type:'application/json'}));
  a.download=f; document.body.appendChild(a); a.click(); a.remove();
}
btnJson.onclick = ()=>{ if(cadastro) downloadJSON(cadastro,'embedding-'+cadastro.owner.toLowerCase().replace(/\s+/g,'-')+'.json'); };

// câmera (opcional, mantido)
btnCam.onclick = async ()=>{
  if(!await ensureModels()) return;
  try{
    const s = await navigator.mediaDevices.getUserMedia({ video:{width:640,height:480} });
    video.srcObject = s; await video.play(); cameraOn = true;
    btnRegister.disabled = false; showPane('Cam');
    log('Câmera ligada.'); refreshUI();
    setInterval(async ()=>{
      if(video.paused||video.ended||authActive||!cameraOn) return;
      try{
        const d = await faceapi.detectSingleFace(video,new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions().withAgeAndGender();
        const dims={width:video.videoWidth||640,height:video.videoHeight||480};
        faceapi.matchDimensions(canvas,dims); canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
        if(d){ const r=faceapi.resizeResults(d,dims); faceapi.draw.drawDetections(canvas,r); faceapi.draw.drawFaceLandmarks(canvas,r); }
      }catch(e){}
    },600);
  }catch(e){ log('Sem câmera: '+e.message+' — use o modo FOTO.'); }
};
btnRegister.onclick = async ()=>{
  if(!cameraOn) return;
  const d = await faceapi.detectSingleFace(video,new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor().withFaceExpressions().withAgeAndGender();
  if(!d){ setStatus('SEM ROSTO NA CÂMERA','denied'); return; }
  const exS=Object.entries(d.expressions).sort((a,b)=>b[1]-a[1]);
  const camName = (nameInput.value || '').trim() || 'Sem nome';
  const payload={ owner:camName, descriptor:Array.from(d.descriptor), age:+d.age.toFixed(1), gender:d.gender, genderProbability:d.genderProbability, expressions:d.expressions, dominantExpression:{name:exS[0][0],probability:exS[0][1]}, landmarks:d.landmarks.positions.map(p=>({x:+p.x.toFixed(1),y:+p.y.toFixed(1)})) };
  await fetch(API+"/api/gravar",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  downloadJSON(payload,'embedding-camera.json'); await apiGetStatus();
};
btnAuth.onclick = ()=>{
  if(!cadastro){ setStatus('⛔ ACESSO NEGADO • GRAVE PRIMEIRO','denied'); return; }
  if(!cameraOn || authActive){ if(authActive){authActive=false;clearInterval(authInterval);refreshUI();} return; }
  authActive=true; btnAuth.textContent='⏸️ Parar';
  authInterval=setInterval(async()=>{
    const d=await faceapi.detectSingleFace(video,new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
    if(!d) return;
    const r=await fetch(API+"/api/reconhecer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({descriptor:Array.from(d.descriptor)})});
    const j=await r.json(); setStatus(j.mensagem, j.reconhecido?'granted':'denied');
  },900);
};
btnClear.onclick = async ()=>{ if(authActive){authActive=false;clearInterval(authInterval);} await fetch(API+"/api/apagar",{method:"DELETE"}); cadastro=null; lastPhoto=null; refreshUI(); log('Apagado. Bloqueado de novo.'); };

ensureModels().then(async ()=>{ await apiGetStatus(); await carregarGaleria(); });
