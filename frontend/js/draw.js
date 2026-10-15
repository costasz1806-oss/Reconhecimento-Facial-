// DRAW.JS — desenha o "arame" do rosto a partir do JSON (68 ou 468 pontos)
(function () {
  const $ = (id) => document.getElementById(id);
  // conexões do modelo 68 pontos (face-api)
  const L = (a, b) => { const r = []; for (let i = a; i <= b; i++) r.push(i); return r; };
  const CONEXOES = [
    L(0, 16), L(17, 21), L(22, 26), L(27, 30), L(31, 35),
    [...L(36, 41), 36], [...L(42, 47), 42],
    [...L(48, 59), 48], [...L(60, 67), 60]
  ];

  function ajustar(pts, W, H, margem) {
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const s = Math.min((W - margem * 2) / Math.max(1e-6, maxX - minX), (H - margem * 2) / Math.max(1e-6, maxY - minY));
    const ox = (W - (maxX - minX) * s) / 2 - minX * s;
    const oy = (H - (maxY - minY) * s) / 2 - minY * s;
    return pts.map((p) => ({ x: p.x * s + ox, y: p.y * s + oy }));
  }

  function desenhar68(canvas, pts, titulo) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const P = ajustar(pts, canvas.width, canvas.height, 40);
    ctx.lineWidth = 2; ctx.strokeStyle = '#22c55e';
    CONEXOES.forEach((c) => {
      ctx.beginPath();
      c.forEach((idx, k) => { const p = P[idx]; if (k === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
      ctx.stroke();
    });
    ctx.fillStyle = '#38bdf8';
    P.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, 7); ctx.fill(); });
    return titulo || ('68 pontos (foto/câmera)');
  }

  function desenharMP(canvas, embedding, titulo) {
    // embedding 1404 = 468 pontos x,y,z normalizados (origem no nariz) — plota x,y
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const pts = [];
    for (let i = 0; i < 468; i++) pts.push({ x: embedding[i * 3], y: embedding[i * 3 + 1] });
    const P = ajustar(pts, canvas.width, canvas.height, 40);
    ctx.fillStyle = '#a78bfa';
    P.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 1.6, 0, 7); ctx.fill(); });
    return titulo || '468 pontos (vídeo MediaPipe — média normalizada)';
  }

  function info(t) { const el = $('drawInfo'); if (el) el.textContent = t; }

  function desenharRegistro(dados, origem) {
    const canvas = $('drawCanvas');
    if (!canvas) return;
    if (dados.landmarks && Array.isArray(dados.landmarks) && dados.landmarks.length >= 60) {
      const t = desenhar68(canvas, dados.landmarks);
      info(`${origem}: ${t} • dono: ${dados.owner || '?'} — prova que o JSON guarda a FORMA, não a foto.`);
    } else if (dados.mediapipe_descriptor && dados.mediapipe_descriptor.length === 1404) {
      const t = desenharMP(canvas, dados.mediapipe_descriptor);
      info(`${origem}: ${t} • dono: ${dados.owner || '?'}.`);
    } else if (dados.descriptor) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      info(`${origem}: só tem embedding 128-d — NÃO dá para desenhar nem reconstruir a foto (é unidirecional). Cadastre por foto/câmera ou vídeo para ter os pontos.`);
    } else {
      info(`${origem}: JSON sem pontos de rosto.`);
    }
  }

  async function prepararDesenho() {
    // preenche o select da galeria (foto + nome)
    try {
      const r = await fetch('/api/historico');
      const lista = await r.json();
      const sel = $('drawHistSelect');
      if (sel && sel.options.length <= 1) {
        lista.forEach((it) => {
          const o = document.createElement('option');
          o.value = it.id;
          o.textContent = `${it.nome || 'Sem nome'} • ${new Date(it.createdAt).toLocaleString()}`;
          o.dataset.foto = it.foto || '';
          sel.appendChild(o);
        });
      }
    } catch (e) { /* backend offline */ }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const btnC = $('btnDrawCadastro');
    if (btnC) btnC.onclick = async () => {
      info('Buscando cadastro...');
      try {
        const r = await fetch('/api/cadastro');
        if (!r.ok) { info('Nenhum rosto gravado ainda. Grave primeiro.'); return; }
        desenharRegistro(await r.json(), 'Cadastro');
      } catch (e) { info('Backend offline.'); }
    };
    const f = $('drawFile');
    if (f) f.onchange = () => {
      const file = f.files[0]; if (!file) return;
      const rd = new FileReader();
      rd.onload = () => {
        try { desenharRegistro(JSON.parse(rd.result), 'Arquivo ' + file.name); }
        catch (e) { info('JSON inválido.'); }
      };
      rd.readAsText(file);
    };
    const btnH = $('btnDrawHist');
    if (btnH) btnH.onclick = async () => {
      const sel = $('drawHistSelect');
      const foto = sel.options[sel.selectedIndex].dataset.foto;
      if (!sel.value || !foto) { info('Escolha um item da galeria com foto.'); return; }
      info('Recalculando 68 pontos da foto salva...');
      try {
        if (typeof ensureModels === 'function') await ensureModels();
        const img = new Image();
        img.src = foto;
        await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
        const det = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.3 })).withFaceLandmarks();
        if (!det) { info('Rosto ilegível na thumbnail salva.'); return; }
        const t = desenhar68($('drawCanvas'), det.landmarks.positions.map((p) => ({ x: p.x, y: p.y })));
        info(`Galeria: ${t} recalculados da foto — a foto revela o rosto, o embedding sozinho não.`);
      } catch (e) { info('Falha: ' + (e.message || e)); }
    };
  });

  window.prepararDesenho = prepararDesenho;
})();
