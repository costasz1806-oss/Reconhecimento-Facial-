// FRONTEND FOTO + CAMERA -> backend Python /api
const API = (() => {
  const h = location.hostname;
  const local = location.protocol === 'file:' || h === 'localhost' || h === '127.0.0.1' || h === '';
  if (!local) return '';
  if (location.protocol === 'file:' || location.port !== '5000') return 'http://localhost:5000';
  return '';
})();
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
  if(typeof faceapi === 'undefined'){ setStatus('FACE-API NÃO CARREGOU (CDN)','denied'); log('Sem internet ou CDN bloqueado. Recarregue a página.'); return false; }
  log('Carregando IA...'); setStatus('CARREGANDO IA...','scanning');
  const URLS = [
    'https://justadudewhohacks.github.io/face-api.js/models',
    'https://venus714.github.io/Face-Recognition-Using-Face-API/models',
    'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights'
  ];
  for(const u of URLS){
    try{
      log('Tentando modelos: ' + u);
      await faceapi.nets.tinyFaceDetector.loadFromUri(u);
      await faceapi.nets.faceLandmark68Net.loadFromUri(u);
      await faceapi.nets.faceRecognitionNet.loadFromUri(u);
      await faceapi.nets.faceExpressionNet.loadFromUri(u);
      await faceapi.nets.ageGenderNet.loadFromUri(u);
      modelsLoaded = true; log('IA pronta via ' + u); return true;
    }catch(e){ console.warn('modelos falharam em ' + u, e); log('Falhou em ' + u + ', tentando próximo...'); }
  }
  setStatus('ERRO AO CARREGAR IA • SEM INTERNET?','denied');
  log('Não carregou os modelos de IA. Verifique internet/adblock e recarregue.');
  return false;
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
  const hasVideoFile = typeof videoFile !== 'undefined' && !!videoFile;
  if(cadastro){
    badge.className = 'yes';
    badge.textContent = '✅ GRAVADO: ' + (cadastro.owner||'').toUpperCase();
    jsonPreview.textContent = JSON.stringify(cadastro,null,2);
    btnJson.disabled = false;
    // NÃO mexe no campo nome: cada foto tem seu próprio nome opcional
    btnRecognizePhoto.disabled = !lastPhoto;
    btnAuth.disabled = !cameraOn;
    btnAuth.textContent = cameraOn ? '🔍 Reconhecer câmera' : '🔒 Reconhecer câmera';
    if(typeof btnRecognizeVideo !== 'undefined' && btnRecognizeVideo) btnRecognizeVideo.disabled = !hasVideoFile;
    const bf2 = $('btnFrameFoto'); if(bf2) bf2.disabled = !hasVideoFile;
    if(!cameraOn && lastPhoto) setStatus('ROSTO GRAVADO ✔ • ANALISE OUTRA FOTO PARA TESTAR','granted');
  }else{
    badge.className = 'no';
    badge.textContent = '⛔ SEM ROSTO GRAVADO';
    btnJson.disabled = true;
    btnAuth.disabled = true;
    btnAuth.textContent = '🔒 Reconhecer câmera (bloqueado)';
    btnRecognizePhoto.disabled = true; // TRAVA: sem cadastro nao reconhece
    if(typeof btnRecognizeVideo !== 'undefined' && btnRecognizeVideo) btnRecognizeVideo.disabled = true;
    if(!lastPhoto){ setStatus('ENVIE UMA FOTO PARA COMEÇAR','blocked'); jsonPreview.textContent='// Envie uma foto com rosto...'; }
    else { setStatus('FOTO ANALISADA • GRAVE PARA LIBERAR RECONHECIMENTO','blocked'); }
  }
  btnRegisterPhoto.disabled = !lastPhoto;
  const temArquivo = fileInput && fileInput.files && fileInput.files.length > 0;
  btnAnalyzePhoto.disabled = !temArquivo || analyzing;
}
nameInput.addEventListener('input', ()=> nameInput.dataset.touched = '1');

// abas
function showPane(which){
  ['Foto','Cam','Video','Galeria','Desenho'].forEach(t=>{
    $('tab'+t).classList.toggle('active', t===which);
    $('pane'+t).hidden = (t!==which);
  });
  if(which==='Galeria') carregarGaleria();
  if(which==='Video') checkMediapipe();
  if(which==='Desenho' && typeof prepararDesenho === 'function') prepararDesenho();
}
$('tabFoto').onclick = ()=>showPane('Foto');
$('tabCam').onclick = ()=>showPane('Cam');
$('tabVideo').onclick = ()=>showPane('Video');
$('tabGaleria').onclick = ()=>showPane('Galeria');
$('tabDesenho').onclick = ()=>showPane('Desenho');

// ---- VIDEO -> backend MediaPipe ----
const videoInput = $('videoInput'), videoName = $('videoName'), videoPreview = $('videoPreview');
const btnRegisterVideo = $('btnRegisterVideo'), btnRecognizeVideo = $('btnRecognizeVideo');
let videoFile = null;

async function checkMediapipe(){
  try{
    const r = await fetch(API + "/api/mediapipe-status");
    const j = await r.json();
    $('mpStatus').textContent = j.instalado
      ? `MediaPipe OK v${j.versao} • embedding ${j.dim_embedding} nums • threshold ${j.threshold}`
      : `MediaPipe NÃO instalado no backend (${j.erro||''}) — rode: pip install -r backend/requirements.txt`;
  }catch(e){ $('mpStatus').textContent = 'Backend offline.'; }
}

if(videoInput){
  videoInput.onchange = ()=>{
    videoFile = videoInput.files[0] || null;
    if(!videoFile) return;
    videoName.textContent = videoFile.name + ' (' + (videoFile.size/1024/1024).toFixed(1) + 'MB)';
    videoPreview.src = URL.createObjectURL(videoFile);
    btnRegisterVideo.disabled = false;
    btnRecognizeVideo.disabled = !cadastro;
    const bf = $('btnFrameFoto'); if(bf) bf.disabled = false;
    log('Vídeo carregado. Pause num frame bom e use: frame como FOTO ou GRAVAR/RECONHECER.');
  };
  // pause no frame -> libera o botão frame-como-foto
  videoPreview.addEventListener('loadeddata', ()=>{ const bf = $('btnFrameFoto'); if(bf && videoFile) bf.disabled = false; });
  videoPreview.addEventListener('seeked', ()=>{ const bf = $('btnFrameFoto'); if(bf && videoFile) bf.disabled = false; });

  // FRAME DO VÍDEO -> analisa no modelo FOTO (idade, expressão, gênero)
  const btnFrameFoto = $('btnFrameFoto');
  if(btnFrameFoto){
    btnFrameFoto.onclick = async ()=>{
      if(!videoFile){ log('Escolha um vídeo primeiro.'); return; }
      if(videoPreview.readyState < 2 || !videoPreview.videoWidth){
        setStatus('VÍDEO AINDA CARREGANDO... DÊ PLAY/PAUSE','scanning');
        log('Espere o vídeo carregar, dê play e pause num frame com seu rosto.');
        return;
      }
      try{
        videoPreview.pause();
        const c = document.createElement('canvas');
        c.width = videoPreview.videoWidth; c.height = videoPreview.videoHeight;
        c.getContext('2d').drawImage(videoPreview, 0, 0, c.width, c.height);
        const blob = await new Promise(res=>c.toBlob(res, 'image/jpeg', 0.92));
        if(!blob){ setStatus('FALHA AO CAPTURAR FRAME','denied'); return; }
        const f = new File([blob], 'frame-video.jpg', { type: 'image/jpeg' });
        const dt = new DataTransfer(); dt.items.add(f);
        fileInput.files = dt.files;
        fileInput.dispatchEvent(new Event('change'));
        showPane('Foto');
        log('Frame enviado para o modelo FOTO. Analisando idade/expressão...');
      }catch(e){ console.error(e); setStatus('FALHA AO CAPTURAR FRAME','denied'); }
    };
  }

  btnRegisterVideo.onclick = async ()=>{
    if(!videoFile){ log('Escolha um vídeo primeiro.'); return; }
    const nome = ((nameInput.value || '').trim() || 'Sem nome').slice(0,40);
    const fd = new FormData();
    fd.append('video', videoFile);
    fd.append('owner', nome);
    setStatus('PROCESSANDO VÍDEO COM MEDIAPIPE...','scanning'); log('Enviando vídeo ao backend MediaPipe...');
    btnRegisterVideo.disabled = true;
    try{
      const r = await fetch(API + "/api/gravar-video", { method: "POST", body: fd });
      const j = await r.json();
      if(!r.ok || !j.ok){ setStatus('FALHA NO VÍDEO','denied'); log(j.erro||'Erro'); btnRegisterVideo.disabled = false; return; }
      log(`Gravado via MediaPipe! ${j.frames_com_rosto}/${j.frames_total} frames com rosto • qualidade ${j.qualidade}%`);
      if(j.dominantExpression) $('pillExpr').textContent = `Expressão: ${j.dominantExpression.name} ${Math.round(j.dominantExpression.probability * 100)}%`;
      jsonPreview.textContent = JSON.stringify(j, null, 2);
      await apiGetStatus();
      setStatus(`ROSTO GRAVADO VIA VÍDEO ✔ ${j.owner.toUpperCase()} (${j.frames_com_rosto} frames)`,'granted');
      btnRecognizeVideo.disabled = false;
    }catch(e){ console.error(e); setStatus('ERRO NO UPLOAD','denied'); }
    btnRegisterVideo.disabled = false;
  };

  btnRecognizeVideo.onclick = async ()=>{
    if(!videoFile){ log('Escolha um vídeo primeiro.'); return; }
    if(!cadastro){ setStatus('GRAVE UM ROSTO PRIMEIRO','blocked'); return; }
    const fd = new FormData();
    fd.append('video', videoFile);
    setStatus('RECONHECENDO VÍDEO (MEDIAPIPE)...','scanning');
    try{
      const r = await fetch(API + "/api/reconhecer-video", { method: "POST", body: fd });
      const j = await r.json();
      if(r.status===403 && j.bloqueado){ setStatus(j.mensagem,'blocked'); return; }
      if(j.erro || j.ok===false){ setStatus('FALHA: '+(j.erro||''),'denied'); return; }
      $('pillScore').textContent = `Similaridade: ${j.similaridade}% dist ${j.distancia}`;
      setStatus(j.mensagem, j.reconhecido ? 'granted' : 'idle');
      log(j.reconhecido ? `Match vídeo: ${j.owner} ${j.similaridade}%` : `Vídeo sem match. dist ${j.distancia}`);
    }catch(e){ console.error(e); setStatus('ERRO NO UPLOAD','denied'); }
  };
}

// upload foto — espera carregar de verdade e analisa sozinho
let analyzing = false, pendingAuto = false;
function esperarFotoPronta(){
  return new Promise((resolve, reject)=>{
    if(photo.complete && photo.naturalWidth > 0) return resolve();
    const to = setTimeout(()=>reject(new Error('Imagem não carregou (arquivo inválido?)')), 15000);
    photo.onload = ()=>{
      clearTimeout(to);
      photoCanvas.getContext('2d').clearRect(0,0,photoCanvas.width,photoCanvas.height);
      resolve();
    };
    photo.onerror = ()=>{ clearTimeout(to); reject(new Error('Não abri essa imagem. Tente JPG/PNG.')); };
  });
}
fileInput.onchange = async ()=>{
  const f = fileInput.files[0]; if(!f) return;
  if(!f.type.startsWith('image/')){ log('Arquivo não é imagem. Envie JPG/PNG.'); setStatus('ARQUIVO INVÁLIDO','denied'); return; }
  if(photo.src && photo.src.startsWith('blob:')) URL.revokeObjectURL(photo.src);
  fileName.textContent = f.name + ' (' + (f.size/1024).toFixed(0) + 'KB)';
  lastPhoto = null;
  setStatus('CARREGANDO FOTO...','scanning'); log('Foto recebida. Carregando...');
  photo.src = URL.createObjectURL(f);
  btnAnalyzePhoto.disabled = true; refreshUI();
  try{
    await esperarFotoPronta();
    log('Foto carregada (' + photo.naturalWidth + 'x' + photo.naturalHeight + '). Analisando sozinho...');
    pendingAuto = false;
    await analisarFoto();
  }catch(e){ setStatus(e.message || 'FALHA AO CARREGAR FOTO','denied'); log(e.message); }
  refreshUI();
};

photo.onload = ()=>{ // limpa canvas ao trocar foto
  try{ photoCanvas.getContext('2d').clearRect(0,0,photoCanvas.width,photoCanvas.height); }catch(e){}
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
        distancia: extra.distancia !== undefined ? extra.distancia : null,
        descriptor: lastPhoto.payload.descriptor, // p/ identificar "Sem nome" depois
        fonte: 'face-api'
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
      else if(item.tipo==='reconhecimento') tag = "<span class='gtag no'>NÃO IDENTIFICADO</span>";
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
async function analisarFoto(){
  if(analyzing){ log('Análise já em andamento...'); return; }
  if(!fileInput.files.length){ log('Escolha uma foto primeiro.'); setStatus('ESCOLHA UMA FOTO','blocked'); return; }
  try{ await esperarFotoPronta(); }
  catch(e){ setStatus(e.message,'denied'); return; }
  if(!await ensureModels()) return;
  analyzing = true; refreshUI();
  setStatus('ANALISANDO FOTO...','scanning'); log('Analisando rosto da foto...');
  try{
    let det = null;
    // tentativa 1: padrão | tentativa 2: mais sensível p/ foto escura/pequena
    const tentativas = [
      new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 }),
      new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.3 })
    ];
    for(const opt of tentativas){
      try{
        det = await faceapi.detectSingleFace(photo, opt)
          .withFaceLandmarks().withFaceDescriptor().withFaceExpressions().withAgeAndGender();
        if(det) break;
      }catch(e){ console.warn('tentativa detector', e); }
    }
    // desenha
    const dims = { width: photo.naturalWidth, height: photo.naturalHeight };
    photoCanvas.width = photo.clientWidth; photoCanvas.height = photo.clientHeight;
    const ctx = photoCanvas.getContext('2d'); ctx.clearRect(0,0,photoCanvas.width,photoCanvas.height);
    if(!det){ setStatus('NENHUM ROSTO NA FOTO ⛔','denied'); log('Tente outra foto: de frente, boa luz, 1 pessoa, sem óculos escuros.'); lastPhoto=null; analyzing=false; refreshUI(); return; }
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
  }catch(e){ console.error(e); setStatus('ERRO NA ANÁLISE: ' + (e.message || e),'denied'); log('Erro: ' + (e.message || e)); }
  analyzing = false; refreshUI();
}
btnAnalyzePhoto.onclick = analisarFoto;

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
  if(!cadastro){ setStatus('NENHUM ROSTO GRAVADO AINDA • GRAVE UM ROSTO PRIMEIRO','blocked'); alert('Grave um rosto primeiro!'); return; }
  if(!lastPhoto){ log('Analise uma foto primeiro.'); return; }
  setStatus('RECONHECENDO VIA PYTHON...','scanning');
  const resp = await fetch(API+"/api/reconhecer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({descriptor:lastPhoto.payload.descriptor})});
  const j = await resp.json();
  if(resp.status===403 || j.bloqueado){ setStatus(j.mensagem,'blocked'); await apiGetStatus(); return; }
  updatePills(lastPhoto.det, j.similaridade, j.distancia);
  setStatus(j.mensagem, j.reconhecido ? 'granted' : 'idle');
  log(j.reconhecido ? ('Match: '+j.owner+' '+j.similaridade+'%') : ('Sem match. dist '+j.distancia));
  await salvarNaGaleria('reconhecimento', { reconhecido: j.reconhecido, similaridade: j.similaridade, distancia: j.distancia });
};

function downloadJSON(o,f){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,2)],{type:'application/json'}));
  a.download=f; document.body.appendChild(a); a.click(); a.remove();
}
btnJson.onclick = ()=>{ if(cadastro) downloadJSON(cadastro,'embedding-'+cadastro.owner.toLowerCase().replace(/\s+/g,'-')+'.json'); };

// câmera (PRINCIPAL: grava e reconhece sem foto)
let camLoop = null, lastCamDet = null;
async function ligarCamera(){
  if(!await ensureModels()) return;
  if(!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1'){
    setStatus('CÂMERA EXIGE HTTPS OU LOCALHOST','denied');
    log('Abra via http://localhost:5000 (não file://) e permita a câmera.');
    return;
  }
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    setStatus('NAVEGADOR SEM SUPORTE A CÂMERA','denied');
    log('Use Chrome/Edge atualizado via localhost.');
    return;
  }
  // fecha stream anterior
  if(video.srcObject){ video.srcObject.getTracks().forEach(t=>t.stop()); }
  try{
    setStatus('LIGANDO CÂMERA...','scanning'); log('Pedindo permissão da câmera...');
    const s = await navigator.mediaDevices.getUserMedia({
      video:{ width:{ideal:720}, height:{ideal:560}, facingMode:'user' }, audio:false
    });
    video.srcObject = s; video.muted = true;
    await video.play(); cameraOn = true;
    btnRegister.disabled = false; showPane('Cam');
    log('Câmera ligada. Posicione o rosto de frente.');
    setStatus('CÂMERA LIGADA • POSICIONE O ROSTO','idle');
    refreshUI();
    if(camLoop) clearInterval(camLoop);
    camLoop = setInterval(async ()=>{
      if(video.paused||video.ended||authActive||!cameraOn) return;
      try{
        const d = await faceapi.detectSingleFace(video,new faceapi.TinyFaceDetectorOptions({inputSize:320,scoreThreshold:0.4})).withFaceLandmarks().withFaceExpressions().withAgeAndGender();
        const dims={width:video.videoWidth||640,height:video.videoHeight||480};
        faceapi.matchDimensions(canvas,dims); canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
        const tip = document.getElementById('camFace');
        if(d){
          lastCamDet = d;
          const r=faceapi.resizeResults(d,dims); faceapi.draw.drawDetections(canvas,r); faceapi.draw.drawFaceLandmarks(canvas,r);
          if(tip){ tip.textContent = 'rosto OK ✓ pode GRAVAR'; tip.className = 'ok'; }
          const ex = Object.entries(d.expressions).sort((a,b)=>b[1]-a[1])[0];
          updatePills(d, null, null);
        } else if(tip){ tip.textContent = 'procurando rosto… chegue perto, boa luz'; tip.className = 'buscando'; }
      }catch(e){}
    },500);
  }catch(e){
    cameraOn = false; refreshUI();
    let msg = 'Sem câmera: ' + (e.name || '') + ' ' + (e.message || '');
    if(e.name==='NotFoundError' || e.name==='OverconstrainedError'){ msg = 'Nenhuma webcam encontrada. Conecte uma câmera ou use a aba Foto.'; setStatus('NENHUMA CÂMERA ENCONTRADA','denied'); }
    else if(e.name==='NotAllowedError'){ msg = 'Permissão negada. Clique no cadeado do navegador → permitir câmera → tente de novo.'; setStatus('PERMISSÃO DA CÂMERA NEGADA','denied'); }
    else if(e.name==='NotReadableError'){ msg = 'Câmera em uso por outro app (Zoom/Meet). Feche e tente de novo.'; setStatus('CÂMERA EM USO','denied'); }
    else setStatus('FALHA NA CÂMERA','denied');
    log(msg);
  }
}
btnCam.onclick = ligarCamera;
// abrir a aba câmera já tenta ligar sozinha
$('tabCam').addEventListener('click', ()=>{ if(!cameraOn) ligarCamera(); });
// captura o MELHOR frame de N tentativas (mais estável sem foto)
async function capturarMelhorRosto(tentativas){
  tentativas = tentativas || 6;
  let melhor = null, melhorScore = -1;
  for(let i=0;i<tentativas;i++){
    try{
      const d = await faceapi.detectSingleFace(video,new faceapi.TinyFaceDetectorOptions({inputSize:416,scoreThreshold:0.35})).withFaceLandmarks().withFaceDescriptor().withFaceExpressions().withAgeAndGender();
      if(d && d.detection.score > melhorScore){ melhorScore = d.detection.score; melhor = d; }
      if(melhorScore > 0.9) break;
    }catch(e){}
    await new Promise(r=>setTimeout(r,250));
  }
  return melhor;
}
btnRegister.onclick = async ()=>{
  if(!cameraOn){ await ligarCamera(); if(!cameraOn) return; }
  setStatus('CAPTURANDO ROSTO... FIQUE PARADO','scanning'); log('Capturando melhor frame (6 tentativas)...');
  const d = await capturarMelhorRosto(6);
  if(!d){ setStatus('SEM ROSTO NA CÂMERA • CHEGUE PERTO','denied'); log('Não achei rosto. Boa luz, de frente, sem cobrir o rosto.'); return; }
  const exS=Object.entries(d.expressions).sort((a,b)=>b[1]-a[1]);
  const camName = (nameInput.value || '').trim() || 'Sem nome';
  const payload={ owner:camName, descriptor:Array.from(d.descriptor), age:+d.age.toFixed(1), gender:d.gender, genderProbability:d.genderProbability, expressions:d.expressions, dominantExpression:{name:exS[0][0],probability:exS[0][1]}, landmarks:d.landmarks.positions.map(p=>({x:+p.x.toFixed(1),y:+p.y.toFixed(1)})) };
  try{
    const r = await fetch(API+"/api/gravar",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const j = await r.json();
    if(!r.ok){ setStatus(j.erro||'ERRO AO GRAVAR','denied'); return; }
    downloadJSON({...payload, createdAt:new Date().toISOString()},'embedding-camera.json');
    updatePills(d, null, null);
    // salva na galeria/base com foto + características (antes a câmera não salvava nada)
    try{
      const c = document.createElement('canvas');
      const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
      const tw = 240; c.width = tw; c.height = Math.round(vh * tw / vw);
      c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
      await fetch(API + "/api/historico", { method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ nome: camName, tipo: "gravado", foto: c.toDataURL('image/jpeg', 0.72),
          idade: Math.round(d.age), genero: d.gender,
          expressao: exS[0][0] + ' ' + Math.round(exS[0][1] * 100) + '%',
          descriptor: Array.from(d.descriptor), fonte: "camera" }) });
      await carregarGaleria();
    }catch(e){ console.warn('galeria camera', e); }
    await apiGetStatus();
    setStatus('ROSTO GRAVADO PELA CÂMERA ✔ AGORA CLIQUE RECONHECER','granted');
    log('Gravado! Score ' + Math.round(d.detection.score*100) + '%. Pode testar o reconhecimento.');
  }catch(e){ setStatus('BACKEND OFFLINE','denied'); }
};
btnAuth.onclick = async ()=>{
  if(!cadastro){ setStatus('NENHUM ROSTO GRAVADO AINDA • GRAVE PRIMEIRO','blocked'); return; }
  if(!cameraOn){ await ligarCamera(); if(!cameraOn) return; }
  if(authActive){ authActive=false; clearInterval(authInterval); btnAuth.textContent='🔒 3. RECONHECER ao vivo'; refreshUI(); return; }
  // 1 teste imediato (melhor frame) + depois modo contínuo
  setStatus('RECONHECENDO... FIQUE PARADO','scanning');
  const d0 = await capturarMelhorRosto(4);
  if(d0){
    try{
      const r0=await fetch(API+"/api/reconhecer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({descriptor:Array.from(d0.descriptor)})});
      const j0=await r0.json(); setStatus(j0.mensagem, j0.reconhecido?'granted':'idle'); updatePills(d0, j0.similaridade, j0.distancia);
    }catch(e){}
  }
  authActive=true; btnAuth.textContent='⏸️ Parar ao vivo';
  authInterval=setInterval(async()=>{
    try{
      const d=await faceapi.detectSingleFace(video,new faceapi.TinyFaceDetectorOptions({inputSize:320,scoreThreshold:0.4})).withFaceLandmarks().withFaceDescriptor();
      if(!d) return;
      const r=await fetch(API+"/api/reconhecer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({descriptor:Array.from(d.descriptor)})});
      const j=await r.json(); setStatus(j.mensagem, j.reconhecido?'granted':'idle');
    }catch(e){}
  },1200);
};
btnClear.onclick = async ()=>{ if(authActive){authActive=false;clearInterval(authInterval);} await fetch(API+"/api/apagar",{method:"DELETE"}); cadastro=null; lastPhoto=null; refreshUI(); log('Apagado. Bloqueado de novo.'); };

ensureModels().then(async ()=>{ await apiGetStatus(); await carregarGaleria(); });
