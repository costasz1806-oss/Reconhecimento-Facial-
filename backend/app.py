"""
Backend Python - Sistema Embedding Facial Sr. Valentim
Roda com: python app.py  (dentro da pasta backend/)
API:
  GET  /api/status     -> { gravado: bool, owner: str|None }
  POST /api/gravar     -> { owner, descriptor[128], age, gender, expressions... } grava rosto
  POST /api/reconhecer -> { descriptor[128] } retorna { reconhecido, owner, distancia, mensagem }
  GET  /api/cadastro   -> retorna o embedding gravado (para baixar JSON)
  DELETE /api/apagar   -> apaga gravação (bloqueia reconhecimento de novo)
  GET/POST/DELETE /api/historico -> galeria: foto + nome + analise em tempo real
Frontend fica em ../frontend e é servido automaticamente em http://localhost:5000
"""
import json
import math
import os
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
BANCO_PATH = os.path.join(BASE_DIR, "banco.json")
HISTORICO_PATH = os.path.join(BASE_DIR, "historico.json")
THRESHOLD = 0.6
MAX_HISTORICO = 100

app = Flask(__name__)
CORS(app)


def carregar_banco():
    if not os.path.exists(BANCO_PATH):
        return None
    try:
        with open(BANCO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def salvar_banco(dados):
    with open(BANCO_PATH, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def carregar_historico():
    if not os.path.exists(HISTORICO_PATH):
        return []
    try:
        with open(HISTORICO_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
            return dados if isinstance(dados, list) else []
    except Exception:
        return []


def salvar_historico(lista):
    with open(HISTORICO_PATH, "w", encoding="utf-8") as f:
        json.dump(lista, f, ensure_ascii=False, indent=2)


def distancia_euclidiana(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


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
        return jsonify({"gravado": True, "owner": banco.get("owner")})
    return jsonify({"gravado": False, "owner": None})


@app.route("/api/gravar", methods=["POST"])
def gravar():
    dados = request.get_json(force=True)
    owner = (dados.get("owner") or "Sem nome").strip()[:40] or "Sem nome"
    descriptor = dados.get("descriptor")

    if not descriptor or len(descriptor) != 128:
        return jsonify({"ok": False, "erro": "embedding (descriptor) com 128 numeros e obrigatorio"}), 400

    registro = {
        "owner": owner,
        "createdAt": datetime.now().isoformat(),
        "app": "Embedding Facial Valentim - backend Python",
        "age": dados.get("age"),
        "gender": dados.get("gender"),
        "genderProbability": dados.get("genderProbability"),
        "expressions": dados.get("expressions"),
        "dominantExpression": dados.get("dominantExpression"),
        "landmarks": dados.get("landmarks"),
        "descriptor": descriptor,
        "descriptorLength": 128,
        "threshold": THRESHOLD,
    }
    salvar_banco(registro)
    return jsonify({"ok": True, "mensagem": f"Rosto de {owner} gravado!", "owner": owner})


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
    if not descriptor or len(descriptor) != 128:
        return jsonify({"ok": False, "erro": "descriptor invalido"}), 400

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
    }
    lista.insert(0, item)  # mais recente primeiro (tempo real)
    salvar_historico(lista[:MAX_HISTORICO])
    return jsonify({"ok": True, "item": item})


@app.route("/api/historico", methods=["DELETE"])
def limpar_historico():
    salvar_historico([])
    return jsonify({"ok": True})


@app.route("/api/historico/<item_id>", methods=["DELETE"])
def remover_item(item_id):
    lista = [x for x in carregar_historico() if x.get("id") != item_id]
    salvar_historico(lista)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("=" * 50)
    print("  Embedding Facial - Backend Python")
    print("  Abra: http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
