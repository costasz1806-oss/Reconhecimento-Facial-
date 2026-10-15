// AUTO.JS — automação + interatividade do frontend (não mexe no app.js)
// 1) auto-analisa foto após upload  2) drag-drop  3) toasts  4) progresso
// 5) steps guiados  6) atalhos teclado  7) polling backend  8) celebração
(function () {
  const $ = (id) => document.getElementById(id);
  const toastBox = () => $('toasts');

  function toast(msg, tipo) {
    const box = toastBox();
    if (!box) return;
    const el = document.createElement('div');
    el.className = 'toast ' + (tipo || 'info');
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => el.classList.add('show'));
    setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 400); }, 3400);
  }

  function progresso(pct) {
    const wrap = $('progressWrap'), bar = $('progressBar');
    if (!wrap || !bar) return;
    if (pct == null) { wrap.hidden = true; bar.style.width = '0%'; return; }
    wrap.hidden = false;
    bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
  }

  // Observa mudanças de status para toast + steps + celebração automática
  function watchStatus() {
    const st = $('status');
    if (!st) return;
    let last = st.textContent;
    new MutationObserver(() => {
      const t = st.textContent;
      if (t === last) return;
      last = t;
      const c = st.className;
      marcarSteps(c);
      if (c.includes('granted')) { toast(t, 'ok'); celebrar(); }
      else if (c.includes('denied')) toast(t, 'erro');
      else toast(t, 'info');
      st.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }).observe(st, { childList: true, characterData: true, subtree: true });
  }

  function marcarSteps(statusClass) {
    const s1 = $('step1'), s2 = $('step2'), s3 = $('step3');
    if (!s1) return;
    [s1, s2, s3].forEach((s) => s && s.classList.remove('active', 'done'));
    const temFoto = !!($('photo') && $('photo').src);
    if (temFoto && s1) s1.classList.add('done');
    else if (s1) s1.classList.add('active');
    if (statusClass.includes('granted')) { s2 && s2.classList.add('done'); s3 && s3.classList.add('active'); }
    else if (temFoto) { s2 && s2.classList.add('active'); }
  }

  function celebrar() {
    const card = $('embed');
    if (!card) return;
    card.classList.remove('party');
    void card.offsetWidth; // reinicia animação
    card.classList.add('party');
    setTimeout(() => card.classList.remove('party'), 1200);
  }

  // Auto-análise: app.js já analisa sozinho no upload; aqui é só backup seguro
  function autoAnalisar() {
    const btn = $('btnAnalyzePhoto'), photo = $('photo');
    if (!btn || !photo) return;
    let ultimaSrc = '';
    setInterval(() => {
      // só clica se: tem arquivo, imagem 100% carregada, botão livre e app.js não está analisando
      const temArquivo = document.getElementById('fileInput').files.length > 0;
      const pronta = photo.complete && photo.naturalWidth > 0;
      const livre = !btn.disabled && typeof analyzing !== 'undefined' ? !analyzing : !btn.disabled;
      if (temArquivo && pronta && photo.src !== ultimaSrc && livre && btn.offsetParent !== null) {
        ultimaSrc = photo.src;
        setTimeout(() => {
          const aindaLivre = !document.getElementById('btnAnalyzePhoto').disabled;
          if (aindaLivre && photo.complete && photo.naturalWidth > 0) { toast('🤖 Análise automática iniciada...', 'info'); btn.click(); }
        }, 800);
      }
      if (!temArquivo) ultimaSrc = '';
    }, 1500);
  }

  // Drag & drop de foto e vídeo em qualquer lugar do card
  function dragDrop() {
    const card = $('embed');
    if (!card) return;
    ['dragenter', 'dragover'].forEach((ev) => card.addEventListener(ev, (e) => {
      e.preventDefault(); card.classList.add('dragover');
    }));
    ['dragleave', 'drop'].forEach((ev) => card.addEventListener(ev, (e) => {
      e.preventDefault(); card.classList.remove('dragover');
    }));
    card.addEventListener('drop', (e) => {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!f) return;
      if (f.type.startsWith('image/')) {
        const dt = new DataTransfer(); dt.items.add(f);
        $('fileInput').files = dt.files;
        $('fileInput').dispatchEvent(new Event('change'));
        document.querySelector('#tabFoto').click();
        toast('📸 Foto recebida por arrastar! Analisando...', 'info');
      } else if (f.type.startsWith('video/')) {
        const dt = new DataTransfer(); dt.items.add(f);
        $('videoInput').files = dt.files;
        $('videoInput').dispatchEvent(new Event('change'));
        document.querySelector('#tabVideo').click();
        toast('🎥 Vídeo recebido por arrastar!', 'info');
      } else {
        toast('Arquivo não suportado. Envie imagem ou vídeo.', 'erro');
      }
    });
  }

  // Upload de vídeo com barra de progresso real (substitui fetch padrão)
  function videoComProgresso() {
    const btnG = $('btnRegisterVideo'), btnR = $('btnRecognizeVideo');
    if (!btnG) return;
    btnG.addEventListener('click', (e) => {
      // deixa o handler original do app.js rodar junto; aqui só mostramos progresso fake suave
      progresso(5);
      let p = 5;
      const iv = setInterval(() => {
        p = Math.min(95, p + Math.random() * 12);
        progresso(p);
        const st = $('status').textContent;
        if (/GRAVADO|FALHA|ERRO/i.test(st)) { progresso(100); setTimeout(() => progresso(null), 800); clearInterval(iv); }
      }, 400);
      setTimeout(() => { clearInterval(iv); progresso(null); }, 60000);
    }, true);
    if (btnR) btnR.addEventListener('click', () => {
      progresso(30);
      setTimeout(() => progresso(null), 8000);
    }, true);
  }

  // Polling do backend a cada 15s: atualiza badge sem recarregar
  function pollingBackend() {
    setInterval(async () => {
      try {
        const r = await fetch('/api/status');
        const j = await r.json();
        const badge = $('cadBadge');
        if (badge && j.gravado && badge.classList.contains('no')) {
          badge.classList.remove('no'); badge.classList.add('yes');
          badge.textContent = '✅ GRAVADO: ' + (j.owner || '').toUpperCase();
          toast('🔄 Backend atualizou: rosto gravado!', 'ok');
        }
      } catch (e) { /* offline, silencioso */ }
    }, 15000);
  }

  // Atalhos de teclado: 1/2/3/4 abas, A analisar, G gravar, R reconhecer
  function atalhos() {
    document.addEventListener('keydown', (e) => {
      if (/INPUT|TEXTAREA/.test(document.activeElement.tagName)) return;
      const k = e.key.toLowerCase();
      if (k === '1') $('tabFoto').click();
      else if (k === '2') $('tabCam').click();
      else if (k === '3' && $('tabVideo')) $('tabVideo').click();
      else if (k === '4') $('tabGaleria').click();
      else if (k === 'a' && !$('btnAnalyzePhoto').disabled) $('btnAnalyzePhoto').click();
      else if (k === 'g') {
        if (!$('btnRegisterPhoto').disabled) $('btnRegisterPhoto').click();
        else if ($('btnRegisterVideo') && !$('btnRegisterVideo').disabled) $('btnRegisterVideo').click();
      }
      else if (k === 'r') {
        if (!$('btnRecognizePhoto').disabled) $('btnRecognizePhoto').click();
        else if ($('btnRecognizeVideo') && !$('btnRecognizeVideo').disabled) $('btnRecognizeVideo').click();
      }
    });
  }

  // Dica contextual automática conforme estado
  function dicaViva() {
    const hint = $('hint');
    if (!hint) return;
    setInterval(() => {
      const badge = $('cadBadge');
      const gravado = badge && badge.classList.contains('yes');
      const temFoto = !!($('photo') && $('photo').src);
      if (!temFoto && !gravado) hint.innerHTML = '💡 <b>Passo 1:</b> arraste uma foto para cá ou clique em Escolher foto — a análise é <b>automática</b>.';
      else if (temFoto && !gravado) hint.innerHTML = '💡 Foto detectada! O sistema analisou sozinho — clique <b>GRAVAR (G)</b> para liberar o reconhecimento.';
      else if (gravado) hint.innerHTML = '💡 Rosto gravado! Envie <b>outra foto/vídeo</b> e pressione <b>R</b> para testar o reconhecimento.';
    }, 3000);
  }

  // Base de dados: exportar JSON/CSV/backup + contador
  function baixar(url, nome) {
    const a = document.createElement('a');
    a.href = url; a.download = nome; document.body.appendChild(a); a.click(); a.remove();
  }
  function baseDados() {
    const bJ = $('btnExpJson'), bC = $('btnExpCsv'), bB = $('btnBackup');
    if (bJ) bJ.onclick = () => { baixar('/api/export/json', 'base-facial.json'); toast('⬇️ JSON exportado (pasta de downloads).', 'ok'); };
    if (bC) bC.onclick = () => { baixar('/api/export/csv', 'base-facial.csv'); toast('📊 CSV exportado — abre no Excel.', 'ok'); };
    if (bB) bB.onclick = () => { baixar('/api/export/backup', 'banco-facial.db'); toast('💾 Backup .db exportado.', 'ok'); };
    setInterval(async () => {
      try {
        const r = await fetch('/api/db-info');
        const j = await r.json();
        if (j.ok && $('dbInfo')) $('dbInfo').textContent = `SQLite • ${j.rostos} rostos • ${j.historico} análises`;
      } catch (e) {}
    }, 10000);
  }

  // Quem é quem: identifica "Sem nome" pelo embedding guardado ou recomputando da foto
  async function identificarSemNome() {
    const box = $('idResult'), btn = $('btnIdentificar');
    if (!box) return;
    box.innerHTML = '🔎 Buscando desconhecidos...';
    btn.disabled = true;
    try {
      const r = await fetch('/api/desconhecidos');
      const j = await r.json();
      if (!j.total) { box.innerHTML = '✅ Ninguém sem nome. Todos identificados!'; btn.disabled = false; return; }
      box.innerHTML = '';
      for (const item of j.items) {
        const card = document.createElement('div');
        card.className = 'idcard';
        card.innerHTML = (item.foto ? `<img src="${item.foto}" />` : '') +
          `<div class="idbody"><div class="idname">❓ Sem nome • ${new Date(item.createdAt).toLocaleString()}</div><div class="idsug">analisando...</div></div>`;
        box.appendChild(card);
        const sug = card.querySelector('.idsug');
        try {
          // 1) tenta pelo embedding guardado no backend
          let resp = await fetch('/api/identificar-id', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: item.id }) });
          let jj = await resp.json();
          if (jj.match) {
            sug.innerHTML = `parece ser <b>${jj.match.nome}</b> (${jj.match.similaridade}%) <small>${jj.match.origem}</small>`;
            const ok = document.createElement('button');
            ok.textContent = '✔ Confirmar ' + jj.match.nome; ok.className = 'btn-reg';
            ok.onclick = async () => {
              await fetch('/api/historico/' + item.id, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ nome: jj.match.nome }) });
              toast('Nome atualizado: ' + jj.match.nome, 'ok'); card.remove();
            };
            card.appendChild(ok);
            continue;
          }
          // 2) sem embedding: recomputa da thumbnail com face-api
          if (jj.sem_embedding && item.foto) {
            sug.textContent = 'recalculando rosto da foto...';
            const img = new Image();
            img.src = item.foto;
            await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
            const det = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.3 })).withFaceLandmarks().withFaceDescriptor();
            if (!det) { sug.textContent = 'rosto ilegível — digite o nome manual na galeria.'; continue; }
            const r2 = await fetch('/api/identificar', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ descriptor: Array.from(det.descriptor) }) });
            const j2 = await r2.json();
            if (j2.match) {
              sug.innerHTML = `parece ser <b>${j2.match.nome}</b> (${j2.match.similaridade}%)`;
              const ok2 = document.createElement('button');
              ok2.textContent = '✔ Confirmar ' + j2.match.nome; ok2.className = 'btn-reg';
              ok2.onclick = async () => {
                await fetch('/api/historico/' + item.id, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ nome: j2.match.nome }) });
                toast('Nome atualizado: ' + j2.match.nome, 'ok'); card.remove();
              };
              card.appendChild(ok2);
            } else sug.textContent = j2.mensagem || 'ninguém parecido na base.';
          } else sug.textContent = jj.mensagem || 'ninguém parecido na base.';
        } catch (e) { sug.textContent = 'erro: ' + e.message; }
      }
      toast('Identificação concluída. Confirme os nomes.', 'info');
    } catch (e) { box.innerHTML = 'Backend offline.'; }
    btn.disabled = false;
  }

  document.addEventListener('DOMContentLoaded', () => {
    watchStatus(); autoAnalisar(); dragDrop();
    videoComProgresso(); pollingBackend(); atalhos(); dicaViva(); baseDados();
    if ($('btnIdentificar')) $('btnIdentificar').onclick = identificarSemNome;
    marcarSteps('blocked');
    console.log('🤖 auto.js ativo: automação + interatividade ligadas.');
  });
})();
