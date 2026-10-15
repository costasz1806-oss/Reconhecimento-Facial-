"""
Backend Python - Sistema Embedding Facial Sr. Valentim
Roda com: python app.py  (dentro da pasta backend/)
API:
  GET  /api/status     -> { gravado: bool, owner: str|None, tipo, tem_mediapipe }
  POST /api/gravar     -> { owner, descriptor[128], ... } grava rosto (face-api.js)
  POST /api/reconhecer -> { descriptor[128] } compara (face-api.js)
  POST /api/gravar-video     -> form-data(video, owner) grava rosto A PARTIR DE VIDEO via MediaPipe
  POST /api/reconhecer-video -> form-data(video) reconhece A PARTIR DE VIDEO via MediaPipe
  GET  /api/mediapipe-status -> { instalado, versao, threshold }
  GET  /api/cadastro   -> retorna o embedding gravado (para baixar JSON)
  DELETE /api/apagar   -> apaga gravação (bloqueia reconhecimento de novo)
  GET/POST/DELETE /api/historico -> galeria
Frontend fica em ../frontend e é servido automaticamente em http://localhost:5000

MediaPipe: detecção + FaceMesh (468 landmarks). O embedding MediaPipe é a média
dos landmarks normalizados ao longo dos frames do vídeo (1404 números).
Comparação por distância euclidiana com THRESHOLD_MEDIAPIPE.
"""
import json
import math
import os
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
BANCO_PATH = os.path.join(BASE_DIR, "banco.json")       # legado (migrado p/ SQLite)
HISTORICO_PATH = os.path.join(BASE_DIR, "historico.json")  # legado (migrado p/ SQLite)
DB_PATH = os.path.join(BASE_DIR, "banco.db")            # base oficial SQLite
THRESHOLD = 0.6              # threshold face-api.js (128-d)
THRESHOLD_MEDIAPIPE = 0.18   # threshold MediaPipe FaceMesh (1404-d normalizado)
MAX_HISTORICO = 100
MAX_FRAMES_ANALISE = 90      # nº máx de frames com rosto usados na média
FRAME_SKIP = 3               # analisa 1 a cada N frames (performance)
MAX_VIDEO_MB = 50

app = Flask(__name__)
CORS(app)

# ---------- MediaPipe (import lazy com fallback) ----------
# Suporta mediapipe 0.10.x (solutions.FaceMesh) E 1.x (tasks.python.vision.FaceLandmarker)
MEDIAPIPE_OK = False
MEDIAPIPE_VERSAO = None
MEDIAPIPE_ERRO = None
MEDIAPIPE_MODO = None  # "solutions" | "tasks"
try:
    import mediapipe as mp
    import cv2
    import numpy as np
    MEDIAPIPE_VERSAO = getattr(mp, "__version__", "desconhecida")
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        MEDIAPIPE_MODO = "solutions"
        MEDIAPIPE_OK = True
    else:
        # testa nova API Tasks
        from mediapipe.tasks.python import vision as _mp_vision  # noqa
        MEDIAPIPE_MODO = "tasks"
        MEDIAPIPE_OK = True
except Exception as e:  # backend continua funcionando sem mediapipe (modo face-api.js)
    MEDIAPIPE_OK = False
    MEDIAPIPE_ERRO = str(e)

MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_LANDMARKER_TASK = os.path.join(MODELS_DIR, "face_landmarker.task")
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _garantir_modelo_landmarker():
    """Baixa o bundle face_landmarker.task na primeira utilização."""
    if os.path.exists(FACE_LANDMARKER_TASK):
        return FACE_LANDMARKER_TASK
    os.makedirs(MODELS_DIR, exist_ok=True)
    import urllib.request
    try:
        print(f"Baixando modelo MediaPipe (~15MB): {FACE_LANDMARKER_URL}")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_TASK)
        return FACE_LANDMARKER_TASK
    except Exception as e:
        raise RuntimeError(f"Não foi possível baixar o modelo face_landmarker.task: {e}")


def carregar_banco():
    b = db_ativo()
    if b:
        return b
    # fallback legado
    if not os.path.exists(BANCO_PATH):
        return None
    try:
        with open(BANCO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def salvar_banco(dados):
    db_salvar_rosto(dados)
    # espelha no JSON legado p/ compatibilidade
    try:
        with open(BANCO_PATH, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def carregar_historico():
    try:
        items = db_listar_historico(MAX_HISTORICO)
        if items:
            return items
    except Exception:
        pass
    if not os.path.exists(HISTORICO_PATH):
        return []
    try:
        with open(HISTORICO_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
            return dados if isinstance(dados, list) else []
    except Exception:
        return []


def salvar_historico(lista):
    try:
        with open(HISTORICO_PATH, "w", encoding="utf-8") as f:
            json.dump(lista, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------- SQLite: base de dados oficial (rostos + histórico) ----------
def _db():
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def db_init():
    con = _db()
    con.execute("""CREATE TABLE IF NOT EXISTS rostos(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      owner TEXT, tipo TEXT, dados TEXT,
      tem_faceapi INTEGER DEFAULT 0, tem_mediapipe INTEGER DEFAULT 0,
      ativo INTEGER DEFAULT 1, created_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS historico(
      id TEXT PRIMARY KEY, nome TEXT, foto TEXT,
      idade REAL, genero TEXT, expressao TEXT,
      similaridade REAL, distancia REAL, reconhecido INTEGER,
      tipo TEXT, created_at TEXT)""")
    # colunas novas (quem já tem db antigo ganha via ALTER)
    for col in ("ALTER TABLE historico ADD COLUMN descriptor TEXT",
                "ALTER TABLE historico ADD COLUMN fonte TEXT"):
        try:
            con.execute(col)
        except Exception:
            pass
    con.commit()
    # migração única do JSON legado
    try:
        n = con.execute("SELECT COUNT(*) c FROM rostos").fetchone()["c"]
        if n == 0 and os.path.exists(BANCO_PATH):
            with open(BANCO_PATH, encoding="utf-8") as f:
                b = json.load(f)
            db_salvar_rosto(b, _con=con)
    except Exception:
        pass
    try:
        n = con.execute("SELECT COUNT(*) c FROM historico").fetchone()["c"]
        if n == 0 and os.path.exists(HISTORICO_PATH):
            with open(HISTORICO_PATH, encoding="utf-8") as f:
                lst = json.load(f)
            if isinstance(lst, list):
                for it in lst[:MAX_HISTORICO]:
                    try:
                        con.execute(
                            "INSERT OR IGNORE INTO historico(id,nome,foto,idade,genero,expressao,similaridade,distancia,reconhecido,tipo,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (it.get("id"), it.get("nome"), it.get("foto"), it.get("idade"), it.get("genero"),
                             it.get("expressao"), it.get("similaridade"), it.get("distancia"),
                             1 if it.get("reconhecido") is True else (0 if it.get("reconhecido") is False else None),
                             it.get("tipo"), it.get("createdAt")))
                    except Exception:
                        pass
                con.commit()
    except Exception:
        pass
    con.close()


def db_salvar_rosto(dados, _con=None):
    con = _con or _db()
    con.execute("UPDATE rostos SET ativo=0")
    con.execute(
        "INSERT INTO rostos(owner,tipo,dados,tem_faceapi,tem_mediapipe,ativo,created_at) VALUES(?,?,?,?,?,?,?)",
        (dados.get("owner"), dados.get("tipo"), json.dumps(dados, ensure_ascii=False),
         1 if dados.get("descriptor") else 0, 1 if dados.get("mediapipe_descriptor") else 0,
         1, dados.get("createdAt") or datetime.now().isoformat()))
    con.commit()
    if not _con:
        con.close()
        # poda: mantém últimos 20 rostos (multi-pessoa futuro)
        try:
            c2 = _db()
            c2.execute("DELETE FROM rostos WHERE id NOT IN (SELECT id FROM rostos ORDER BY id DESC LIMIT 20)")
            c2.commit(); c2.close()
        except Exception:
            pass


def db_ativo():
    try:
        con = _db()
        r = con.execute("SELECT dados FROM rostos WHERE ativo=1 ORDER BY id DESC LIMIT 1").fetchone()
        con.close()
        return json.loads(r["dados"]) if r else None
    except Exception:
        return None


def db_listar_historico(limite=100):
    con = _db()
    rows = con.execute("SELECT * FROM historico ORDER BY created_at DESC LIMIT ?", (limite,)).fetchall()
    con.close()
    out = []
    for r in rows:
        rec = r["reconhecido"]
        d = {"id": r["id"], "nome": r["nome"], "foto": r["foto"], "idade": r["idade"],
             "genero": r["genero"], "expressao": r["expressao"], "similaridade": r["similaridade"],
             "distancia": r["distancia"],
             "reconhecido": True if rec == 1 else (False if rec == 0 else None),
             "tipo": r["tipo"], "createdAt": r["created_at"],
             "tem_embedding": False, "fonte": None}
        try:
            desc = r["descriptor"]
            if desc:
                d["tem_embedding"] = True
        except Exception:
            pass
        try:
            d["fonte"] = r["fonte"]
        except Exception:
            pass
        out.append(d)
    return out


def db_add_historico(item):
    con = _db()
    desc = item.get("descriptor")
    desc_txt = json.dumps(desc) if isinstance(desc, list) else None
    con.execute(
        "INSERT OR REPLACE INTO historico(id,nome,foto,idade,genero,expressao,similaridade,distancia,reconhecido,tipo,created_at,descriptor,fonte) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (item.get("id"), item.get("nome"), item.get("foto"), item.get("idade"), item.get("genero"),
         item.get("expressao"), item.get("similaridade"), item.get("distancia"),
         1 if item.get("reconhecido") is True else (0 if item.get("reconhecido") is False else None),
         item.get("tipo"), item.get("createdAt"), desc_txt, item.get("fonte") or "face-api"))
    con.execute("DELETE FROM historico WHERE id NOT IN (SELECT id FROM historico ORDER BY created_at DESC LIMIT ?)", (MAX_HISTORICO,))
    con.commit(); con.close()


db_init()


def distancia_euclidiana(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------- MediaPipe: extração de embedding a partir de VÍDEO ----------
def _normalizar_landmarks(pts):
    """pts: np.array (N,3). Origem no nariz (idx 1), escala pela distância interocular (33-263)."""
    import numpy as _np
    pts = _np.asarray(pts, dtype=_np.float64)
    nariz = pts[1]
    iod = float(_np.linalg.norm(pts[33][:2] - pts[263][:2]))
    if iod < 1e-6:
        return None
    norm = (pts - nariz) / iod
    return norm.flatten().tolist()


def _extrair_via_solutions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo. Envie mp4/avi/mov/webm.")
    mp_face_mesh = mp.solutions.face_mesh
    vetores, bboxes, frames_total, idx = [], [], 0, 0
    with mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_total += 1
            idx += 1
            if idx % FRAME_SKIP != 0 or len(vetores) >= MAX_FRAMES_ANALISE:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
            vec = _normalizar_landmarks(pts)
            if vec is None:
                continue
            vetores.append(vec)
            xs, ys = pts[:, 0], pts[:, 1]
            bboxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    cap.release()
    return vetores, bboxes, frames_total


def _extrair_via_tasks(video_path):
    """MediaPipe 1.x: FaceLandmarker (RunningMode.VIDEO). Baixa modelo .task se preciso."""
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

    model_path = _garantir_modelo_landmarker()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo. Envie mp4/avi/mov/webm.")
    vetores, bboxes, frames_total, idx = [], [], 0, 0
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_total += 1
            idx += 1
            if idx % FRAME_SKIP != 0 or len(vetores) >= MAX_FRAMES_ANALISE:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or idx * 33
            res = landmarker.detect_for_video(mp_image, ts_ms)
            if not res.face_landmarks:
                continue
            lm = res.face_landmarks[0]  # 478 pontos (inclui íris); usa 468 base
            pts = np.array([[p.x, p.y, p.z] for p in lm[:468]], dtype=np.float64)
            vec = _normalizar_landmarks(pts)
            if vec is None:
                continue
            vetores.append(vec)
            xs, ys = pts[:, 0], pts[:, 1]
            bboxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    cap.release()
    return vetores, bboxes, frames_total


def extrair_embedding_mediapipe(video_path):
    """Lê vídeo, detecta rosto com MediaPipe e retorna embedding médio.

    Retorna dict: { embedding[1404], frames_total, frames_com_rosto,
                    frames_usados, bbox, qualidade }
    Levanta ValueError se nenhum rosto for encontrado.
    """
    if not MEDIAPIPE_OK:
        raise RuntimeError(
            f"MediaPipe não instalado. Rode: pip install -r requirements.txt "
            f"(erro original: {MEDIAPIPE_ERRO})"
        )
    if MEDIAPIPE_MODO == "solutions":
        vetores, bboxes, frames_total = _extrair_via_solutions(video_path)
    else:
        vetores, bboxes, frames_total = _extrair_via_tasks(video_path)

    if not vetores:
        raise ValueError(
            f"Nenhum rosto detectado pelo MediaPipe em {frames_total} frames. "
            "Grave de frente, boa luz, sem óculos escuros."
        )

    media = np.mean(np.array(vetores, dtype=np.float64), axis=0)
    amostrados = max(1, frames_total // FRAME_SKIP)
    qualidade = round(len(vetores) / amostrados * 100, 1)
    bbox_medio = None
    if bboxes:
        bbox_medio = [round(float(x), 4) for x in np.mean(np.array(bboxes), axis=0).tolist()]

    return {
        "embedding": [float(x) for x in media.tolist()],  # 1404 números
        "dim": len(media.tolist()),
        "frames_total": frames_total,
        "frames_com_rosto": len(vetores),
        "frames_usados": len(vetores),
        "bbox_medio": bbox_medio,
        "qualidade": qualidade,
    }


def comparar_mediapipe(emb_a, emb_b):
    dist = distancia_euclidiana(emb_a, emb_b)
    similaridade = max(0.0, (1 - dist) * 100)
    return dist, similaridade


# ---------- ROTAS FRONTEND ----------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    # serve css/js do frontend
    full = os.path.join(FRONTEND_DIR, path)
    if os.path.isfile(full):
        return send_from_directory(FRONTEND_DIR, path)
    return jsonify({"erro": "nao encontrado"}), 404


# ---------- API ----------
@app.route("/api/status", methods=["GET"])
def status():
    banco = carregar_banco()
    if banco:
        return jsonify({
            "gravado": True,
            "owner": banco.get("owner"),
            "tipo": banco.get("tipo", "face-api"),
            "tem_faceapi": bool(banco.get("descriptor")),
            "tem_mediapipe": bool(banco.get("mediapipe_descriptor")),
            "mediapipe_instalado": MEDIAPIPE_OK,
        })
    return jsonify({"gravado": False, "owner": None, "mediapipe_instalado": MEDIAPIPE_OK})


@app.route("/api/mediapipe-status", methods=["GET"])
def mediapipe_status():
    return jsonify({
        "instalado": MEDIAPIPE_OK,
        "versao": MEDIAPIPE_VERSAO,
        "modo": MEDIAPIPE_MODO,
        "erro": MEDIAPIPE_ERRO,
        "threshold": THRESHOLD_MEDIAPIPE,
        "dim_embedding": 468 * 3,
        "modelo_task": os.path.exists(FACE_LANDMARKER_TASK),
        "formatos": ["mp4", "avi", "mov", "webm", "mkv"],
    })


@app.route("/api/gravar", methods=["POST"])
def gravar():
    dados = request.get_json(force=True)
    owner = (dados.get("owner") or "Sem nome").strip()[:40] or "Sem nome"
    descriptor = dados.get("descriptor")

    if not descriptor or len(descriptor) != 128:
        return jsonify({"ok": False, "erro": "embedding (descriptor) com 128 numeros e obrigatorio"}), 400

    anterior = carregar_banco() or {}
    registro = {
        "owner": owner,
        "createdAt": datetime.now().isoformat(),
        "app": "Embedding Facial Valentim - backend Python",
        "tipo": "face-api" if not anterior.get("mediapipe_descriptor") else "hibrido",
        "age": dados.get("age"),
        "gender": dados.get("gender"),
        "genderProbability": dados.get("genderProbability"),
        "expressions": dados.get("expressions"),
        "dominantExpression": dados.get("dominantExpression"),
        "landmarks": dados.get("landmarks"),
        "descriptor": descriptor,
        "descriptorLength": 128,
        "threshold": THRESHOLD,
        # preserva gravação MediaPipe anterior (modo híbrido)
        "mediapipe_descriptor": anterior.get("mediapipe_descriptor"),
        "mediapipe_dim": anterior.get("mediapipe_dim"),
        "mediapipe_frames": anterior.get("mediapipe_frames"),
        "mediapipe_qualidade": anterior.get("mediapipe_qualidade"),
        "threshold_mediapipe": THRESHOLD_MEDIAPIPE,
    }
    salvar_banco(registro)
    return jsonify({"ok": True, "mensagem": f"Rosto de {owner} gravado!", "owner": owner})


def _salvar_temp_video(file_storage):
    suffix = os.path.splitext(file_storage.filename or "")[1].lower() or ".mp4"
    if suffix not in (".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"):
        suffix = ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file_storage.save(tmp.name)
    finally:
        tmp.close()
    if os.path.getsize(tmp.name) > MAX_VIDEO_MB * 1024 * 1024:
        os.remove(tmp.name)
        raise ValueError(f"Vídeo muito grande (máx {MAX_VIDEO_MB}MB). Envie um trecho mais curto.")
    return tmp.name


@app.route("/api/gravar-video", methods=["POST"])
def gravar_video():
    """Grava rosto A PARTIR DE VÍDEO usando MediaPipe (form-data: video, owner)."""
    if not MEDIAPIPE_OK:
        return jsonify({
            "ok": False,
            "erro": f"MediaPipe não instalado no backend ({MEDIAPIPE_ERRO}). Rode: pip install mediapipe opencv-python numpy"
        }), 500
    if "video" not in request.files:
        return jsonify({"ok": False, "erro": "Envie o vídeo em form-data com a chave 'video'"}), 400

    owner = (request.form.get("owner") or "Sem nome").strip()[:40] or "Sem nome"
    tmp_path = None
    try:
        tmp_path = _salvar_temp_video(request.files["video"])
        info = extrair_embedding_mediapipe(tmp_path)
    except ValueError as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"ok": False, "erro": str(e)}), 422
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"ok": False, "erro": f"Falha ao processar vídeo: {e}"}), 500
    finally:
        pass
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

    anterior = carregar_banco() or {}
    registro = {
        "owner": owner,
        "createdAt": datetime.now().isoformat(),
        "app": "Embedding Facial Valentim - backend Python + MediaPipe",
        "tipo": "mediapipe-video" if not anterior.get("descriptor") else "hibrido",
        "threshold": THRESHOLD,
        "threshold_mediapipe": THRESHOLD_MEDIAPIPE,
        # preserva face-api.js anterior (modo híbrido)
        "descriptor": anterior.get("descriptor"),
        "descriptorLength": anterior.get("descriptorLength"),
        "age": anterior.get("age"),
        "gender": anterior.get("gender"),
        # novo embedding MediaPipe
        "mediapipe_descriptor": info["embedding"],
        "mediapipe_dim": info["dim"],
        "mediapipe_frames": {
            "total": info["frames_total"],
            "com_rosto": info["frames_com_rosto"],
            "usados": info["frames_usados"],
        },
        "mediapipe_qualidade": info["qualidade"],
        "mediapipe_bbox_medio": info["bbox_medio"],
    }
    salvar_banco(registro)
    return jsonify({
        "ok": True,
        "mensagem": f"Rosto de {owner} gravado via MediaPipe!",
        "owner": owner,
        "tipo": registro["tipo"],
        "frames_total": info["frames_total"],
        "frames_com_rosto": info["frames_com_rosto"],
        "qualidade": info["qualidade"],
        "dim": info["dim"],
    })


@app.route("/api/reconhecer-video", methods=["POST"])
def reconhecer_video():
    """Reconhece A PARTIR DE VÍDEO usando MediaPipe (form-data: video)."""
    banco = carregar_banco()
    if not banco or not banco.get("mediapipe_descriptor"):
        return jsonify({
            "reconhecido": False,
            "bloqueado": True,
            "mensagem": "⛔ ACESSO NEGADO • NENHUM ROSTO (VÍDEO/MediaPipe) GRAVADO. Grave via /api/gravar-video primeiro."
        }), 403
    if not MEDIAPIPE_OK:
        return jsonify({"reconhecido": False, "erro": "MediaPipe não instalado"}), 500
    if "video" not in request.files:
        return jsonify({"ok": False, "erro": "Envie o vídeo em form-data com a chave 'video'"}), 400

    tmp_path = None
    try:
        tmp_path = _salvar_temp_video(request.files["video"])
        info = extrair_embedding_mediapipe(tmp_path)
    except ValueError as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"ok": False, "erro": str(e)}), 422
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    dist, sim = comparar_mediapipe(banco["mediapipe_descriptor"], info["embedding"])
    if dist < banco.get("threshold_mediapipe", THRESHOLD_MEDIAPIPE):
        return jsonify({
            "reconhecido": True,
            "bloqueado": False,
            "owner": banco["owner"],
            "distancia": round(dist, 4),
            "similaridade": round(sim, 1),
            "frames_com_rosto": info["frames_com_rosto"],
            "qualidade": info["qualidade"],
            "mensagem": f"SEJA BEM-VINDO, {banco['owner'].upper()} ✔ (MediaPipe)"
        })
    return jsonify({
        "reconhecido": False,
        "bloqueado": False,
        "owner": None,
        "distancia": round(dist, 4),
        "similaridade": round(sim, 1),
        "frames_com_rosto": info["frames_com_rosto"],
        "qualidade": info["qualidade"],
        "mensagem": "⛔ ACESSO NEGADO (MediaPipe)"
    })


@app.route("/api/reconhecer", methods=["POST"])
def reconhecer():
    banco = carregar_banco()
    # TRAVA que voce pediu: sem rosto gravado, NAO permite reconhecer
    if not banco:
        return jsonify({
            "reconhecido": False,
            "bloqueado": True,
            "mensagem": "⛔ ACESSO NEGADO • NENHUM ROSTO GRAVADO. Grave o rosto primeiro."
        }), 403

    dados = request.get_json(force=True)
    descriptor = dados.get("descriptor")
    mp_descriptor = dados.get("mediapipe_descriptor")

    # Caminho 1: embedding MediaPipe (1404-d) via JSON
    if mp_descriptor and len(mp_descriptor) == 468 * 3:
        if not banco.get("mediapipe_descriptor"):
            return jsonify({
                "reconhecido": False,
                "bloqueado": True,
                "mensagem": "⛔ Nenhum embedding MediaPipe gravado. Grave via vídeo primeiro."
            }), 403
        dist, sim = comparar_mediapipe(banco["mediapipe_descriptor"], mp_descriptor)
        if dist < banco.get("threshold_mediapipe", THRESHOLD_MEDIAPIPE):
            return jsonify({
                "reconhecido": True,
                "bloqueado": False,
                "owner": banco["owner"],
                "distancia": round(dist, 4),
                "similaridade": round(sim, 1),
                "mensagem": f"SEJA BEM-VINDO, {banco['owner'].upper()} ✔"
            })
        return jsonify({
            "reconhecido": False,
            "bloqueado": False,
            "owner": None,
            "distancia": round(dist, 4),
            "similaridade": round(sim, 1),
            "mensagem": "⛔ ACESSO NEGADO"
        })

    # Caminho 2: embedding face-api.js (128-d) — original
    if not descriptor or len(descriptor) != 128:
        return jsonify({"ok": False, "erro": "descriptor invalido"}), 400
    if not banco.get("descriptor"):
        return jsonify({
            "reconhecido": False,
            "bloqueado": True,
            "mensagem": "⛔ ACESSO NEGADO • NENHUM ROSTO (FOTO) GRAVADO. Grave uma foto primeiro."
        }), 403

    dist = distancia_euclidiana(banco["descriptor"], descriptor)
    similaridade = max(0.0, (1 - dist) * 100)

    if dist < banco.get("threshold", THRESHOLD):
        return jsonify({
            "reconhecido": True,
            "bloqueado": False,
            "owner": banco["owner"],
            "distancia": round(dist, 4),
            "similaridade": round(similaridade, 1),
            "mensagem": f"SEJA BEM-VINDO, {banco['owner'].upper()} ✔"
        })
    return jsonify({
        "reconhecido": False,
        "bloqueado": False,
        "owner": None,
        "distancia": round(dist, 4),
        "similaridade": round(similaridade, 1),
        "mensagem": "⛔ ACESSO NEGADO"
    })


@app.route("/api/cadastro", methods=["GET"])
def cadastro():
    banco = carregar_banco()
    if not banco:
        return jsonify({"gravado": False}), 404
    return jsonify(banco)


@app.route("/api/apagar", methods=["DELETE"])
def apagar():
    try:
        con = _db()
        con.execute("UPDATE rostos SET ativo=0")
        con.commit(); con.close()
    except Exception:
        pass
    if os.path.exists(BANCO_PATH):
        os.remove(BANCO_PATH)
    return jsonify({"ok": True, "mensagem": "Gravacao apagada. Reconhecimento bloqueado."})


# ---------- HISTÓRICO / GALERIA ----------
@app.route("/api/historico", methods=["GET"])
def listar_historico():
    return jsonify(carregar_historico())


@app.route("/api/historico", methods=["POST"])
def adicionar_historico():
    dados = request.get_json(force=True)
    lista = carregar_historico()
    item = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "nome": (dados.get("nome") or "Sem nome").strip()[:40],
        "foto": dados.get("foto") or "",  # thumbnail dataURL jpeg
        "idade": dados.get("idade"),
        "genero": dados.get("genero"),
        "expressao": dados.get("expressao"),
        "similaridade": dados.get("similaridade"),
        "distancia": dados.get("distancia"),
        "reconhecido": dados.get("reconhecido"),  # True/False/None (só análise)
        "tipo": dados.get("tipo") or "analise",  # analise | gravado | reconhecimento
        "createdAt": datetime.now().isoformat(),
        "descriptor": dados.get("descriptor"),  # embedding 128-d p/ identificar depois
        "fonte": dados.get("fonte") or "face-api",
    }
    try:
        db_add_historico(item)
    except Exception:
        pass
    lista.insert(0, item)  # mais recente primeiro (tempo real)
    salvar_historico(lista[:MAX_HISTORICO])
    return jsonify({"ok": True, "item": item})


@app.route("/api/historico", methods=["DELETE"])
def limpar_historico():
    try:
        con = _db()
        con.execute("DELETE FROM historico")
        con.commit(); con.close()
    except Exception:
        pass
    salvar_historico([])
    return jsonify({"ok": True})


@app.route("/api/historico/<item_id>", methods=["PUT"])
def renomear_item(item_id):
    """Dá nome a um 'Sem nome' depois de identificar."""
    dados = request.get_json(force=True, silent=True) or {}
    nome = (dados.get("nome") or "").strip()[:40]
    if not nome:
        return jsonify({"ok": False, "erro": "Nome obrigatório"}), 400
    try:
        con = _db()
        con.execute("UPDATE historico SET nome=? WHERE id=?", (nome, item_id))
        con.commit(); con.close()
    except Exception as e:
        return jsonify({"ok": False, "erro": str(e)}), 500
    salvar_historico([x if x.get("id") != item_id else {**x, "nome": nome} for x in carregar_historico()])
    return jsonify({"ok": True, "id": item_id, "nome": nome})


def _candidatos_nomeados():
    """Rostos com nome: tabela rostos + histórico nomeado com embedding."""
    cands = []
    try:
        con = _db()
        for r in con.execute("SELECT owner, dados FROM rostos ORDER BY id DESC LIMIT 20").fetchall():
            try:
                d = json.loads(r["dados"])
                if d.get("descriptor") and (d.get("owner") or "").strip().lower() not in ("", "sem nome"):
                    cands.append({"nome": d["owner"], "descriptor": d["descriptor"], "origem": "cadastro"})
            except Exception:
                pass
        for r in con.execute("SELECT id, nome, descriptor FROM historico LIMIT 200").fetchall():
            try:
                if not r["descriptor"] or not r["nome"]:
                    continue
                if r["nome"].strip().lower() in ("", "sem nome"):
                    continue
                desc = json.loads(r["descriptor"])
                if isinstance(desc, list) and len(desc) == 128:
                    cands.append({"nome": r["nome"], "descriptor": desc, "origem": "galeria:" + r["id"]})
            except Exception:
                pass
        con.close()
    except Exception:
        pass
    # dedup por nome (último vale)
    vistos, unicos = set(), []
    for c in cands:
        k = c["nome"].strip().lower()
        if k not in vistos:
            vistos.add(k); unicos.append(c)
    return unicos


@app.route("/api/identificar", methods=["POST"])
def identificar():
    """Recebe {descriptor[128]} desconhecido e devolve ranking de nomes parecidos."""
    dados = request.get_json(force=True, silent=True) or {}
    desc = dados.get("descriptor")
    if not desc or len(desc) != 128:
        return jsonify({"ok": False, "erro": "descriptor 128-d obrigatório"}), 400
    cands = _candidatos_nomeados()
    if not cands:
        return jsonify({"ok": True, "match": None, "ranking": [],
                        "mensagem": "Nenhum rosto nomeado na base para comparar. Nomeie alguém primeiro."})
    ranking = []
    for c in cands:
        try:
            dist = distancia_euclidiana(desc, c["descriptor"])
            ranking.append({"nome": c["nome"], "origem": c["origem"],
                            "distancia": round(dist, 4),
                            "similaridade": round(max(0.0, (1 - dist) * 100), 1)})
        except Exception:
            pass
    ranking.sort(key=lambda x: x["distancia"])
    top = ranking[:3]
    melhor = top[0] if top and top[0]["distancia"] < THRESHOLD else None
    return jsonify({"ok": True,
                    "match": {**melhor, "confiavel": melhor["distancia"] < 0.5} if melhor else None,
                    "ranking": top,
                    "mensagem": f"Parece ser {melhor['nome']} ({melhor['similaridade']}%)" if melhor else "Ninguém parecido na base."})


@app.route("/api/desconhecidos", methods=["GET"])
def desconhecidos():
    """Lista itens 'Sem nome' — com foto e se já tem embedding guardado."""
    todos = carregar_historico()
    desc = [x for x in todos if not (x.get("nome") or "").strip() or x.get("nome", "").strip().lower() == "sem nome"]
    return jsonify({"total": len(desc), "items": desc})


@app.route("/api/identificar-id", methods=["POST"])
def identificar_id():
    """Identifica um item da galeria pelo embedding GUARDADO (sem reenviar foto)."""
    dados = request.get_json(force=True, silent=True) or {}
    item_id = dados.get("id")
    if not item_id:
        return jsonify({"ok": False, "erro": "id obrigatório"}), 400
    try:
        con = _db()
        r = con.execute("SELECT descriptor FROM historico WHERE id=?", (item_id,)).fetchone()
        con.close()
    except Exception as e:
        return jsonify({"ok": False, "erro": str(e)}), 500
    if not r or not r["descriptor"]:
        return jsonify({"ok": True, "match": None, "sem_embedding": True,
                        "mensagem": "Sem embedding guardado. Recompute pela foto."})
    try:
        desc = json.loads(r["descriptor"])
    except Exception:
        return jsonify({"ok": True, "match": None, "sem_embedding": True,
                        "mensagem": "Embedding corrompido."})
    cands = _candidatos_nomeados()
    ranking = []
    for c in cands:
        try:
            dist = distancia_euclidiana(desc, c["descriptor"])
            ranking.append({"nome": c["nome"], "origem": c["origem"],
                            "distancia": round(dist, 4),
                            "similaridade": round(max(0.0, (1 - dist) * 100), 1)})
        except Exception:
            pass
    ranking.sort(key=lambda x: x["distancia"])
    top = ranking[:3]
    melhor = top[0] if top and top[0]["distancia"] < THRESHOLD else None
    return jsonify({"ok": True, "match": melhor, "ranking": top})


@app.route("/api/historico/<item_id>", methods=["DELETE"])
def remover_item(item_id):
    try:
        con = _db()
        con.execute("DELETE FROM historico WHERE id=?", (item_id,))
        con.commit(); con.close()
    except Exception:
        pass
    lista = [x for x in carregar_historico() if x.get("id") != item_id]
    salvar_historico(lista)
    return jsonify({"ok": True})


# ---------- EXPORTAÇÃO: base de dados em documento ----------
@app.route("/api/export/json", methods=["GET"])
def export_json():
    """Baixa tudo (rosto ativo + histórico) em JSON — pasta/documento de base de dados."""
    from flask import Response
    dump = {
        "exportado_em": datetime.now().isoformat(),
        "cadastro": carregar_banco(),
        "historico": carregar_historico(),
    }
    return Response(json.dumps(dump, ensure_ascii=False, indent=2),
                    mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=base-facial.json"})


@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """Histórico em CSV (abre no Excel). ?com_foto=1 inclui thumbnail base64."""
    import csv, io
    from flask import Response
    com_foto = request.args.get("com_foto") == "1"
    hist = carregar_historico()
    buf = io.StringIO()
    cols = ["id", "nome", "idade", "genero", "expressao", "similaridade", "distancia", "reconhecido", "tipo", "createdAt"] + (["foto"] if com_foto else ["tem_foto"])
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for it in hist:
        row = {k: it.get(k) for k in cols if k != "tem_foto"}
        if not com_foto:
            row["tem_foto"] = "sim" if it.get("foto") else "nao"
        w.writerow(row)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=base-facial.csv"})


@app.route("/api/export/backup", methods=["GET"])
def export_backup():
    """Baixa o banco SQLite oficial (banco.db) — pasta de base de dados."""
    from flask import send_file
    if not os.path.exists(DB_PATH):
        return jsonify({"ok": False, "erro": "Banco ainda vazio. Grave um rosto primeiro."}), 404
    return send_file(DB_PATH, as_attachment=True, download_name="banco-facial.db")


@app.route("/api/db-info", methods=["GET"])
def db_info():
    try:
        con = _db()
        n_rostos = con.execute("SELECT COUNT(*) c FROM rostos").fetchone()["c"]
        n_hist = con.execute("SELECT COUNT(*) c FROM historico").fetchone()["c"]
        con.close()
    except Exception as e:
        return jsonify({"ok": False, "erro": str(e)}), 500
    return jsonify({"ok": True, "sqlite": os.path.basename(DB_PATH), "rostos": n_rostos,
                    "historico": n_hist, "json_legado": os.path.exists(BANCO_PATH)})


if __name__ == "__main__":
    print("=" * 50)
    print("  Embedding Facial - Backend Python + MediaPipe")
    print("  Abra: http://localhost:5000")
    print(f"  MediaPipe: {'OK v' + str(MEDIAPIPE_VERSAO) if MEDIAPIPE_OK else 'NAO INSTALADO -> pip install -r requirements.txt'}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
