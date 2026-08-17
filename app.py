"""
API FastAPI para servir o modelo final de clusterização de municípios brasileiros
(KPrototypes, k=2), treinado no notebook `notebooks/04_Avaliacao_Conclusao.ipynb`.

Como rodar:
    pip install -r requirements.txt
    uvicorn app:app --reload

Depois, acesse http://127.0.0.1:8000/docs para a documentação interativa.

Pré-requisito: os artefatos em ./models/ (final_scaler.joblib, final_kprototypes.joblib)
precisam existir. Eles são gerados rodando os notebooks 01 a 04, nessa ordem.
"""

import os
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PASTA_MODELS = "models"
PASTA_REPORTS = "reports"

CAMINHO_SCALER = os.path.join(PASTA_MODELS, "final_scaler.joblib")
CAMINHO_MODELO = os.path.join(PASTA_MODELS, "final_kprototypes.joblib")
CAMINHO_DATASET_CLUSTERIZADO = os.path.join(PASTA_REPORTS, "municipios_clusterizados.csv")

REGIOES_VALIDAS = Literal["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]


class MunicipioInput(BaseModel):
    """Indicadores de um município (mesmas variáveis usadas no treino do modelo)."""

    populacao: float = Field(..., ge=0, description="População estimada")
    pib_total_mil_reais: float = Field(..., ge=0, description="PIB total, em mil R$")
    vab_agropecuaria_mil_reais: float = Field(..., ge=0, description="VAB da agropecuária, em mil R$")
    vab_industria_mil_reais: float = Field(..., ge=0, description="VAB da indústria, em mil R$")
    vab_servicos_mil_reais: float = Field(..., ge=0, description="VAB de serviços, em mil R$")
    vab_adm_publica_mil_reais: float = Field(..., ge=0, description="VAB da administração pública, em mil R$")
    pib_per_capita_reais: float = Field(..., ge=0, description="PIB per capita, em R$")
    regiao: REGIOES_VALIDAS = Field(..., description="Região geográfica do município")

    model_config = {
        "json_schema_extra": {
            "example": {
                "populacao": 111148,
                "pib_total_mil_reais": 3211294,
                "vab_agropecuaria_mil_reais": 293001,
                "vab_industria_mil_reais": 407675,
                "vab_servicos_mil_reais": 1307977,
                "vab_adm_publica_mil_reais": 782306,
                "pib_per_capita_reais": 28892.05,
                "regiao": "Norte",
            }
        }
    }


class PredicaoOutput(BaseModel):
    cluster: int
    rotulo: str
    descricao: str


class ClusterInfo(BaseModel):
    cluster: int
    rotulo: str
    n_municipios: int
    populacao_mediana: float
    pib_total_mediano: float
    pib_per_capita_mediano: float


COLUNAS_NUMERICAS = [
    "populacao",
    "pib_total_mil_reais",
    "vab_agropecuaria_mil_reais",
    "vab_industria_mil_reais",
    "vab_servicos_mil_reais",
    "vab_adm_publica_mil_reais",
    "pib_per_capita_reais",
]


def _carregar_artefatos():
    """Carrega scaler e modelo; falha rápido e com mensagem clara se faltarem."""
    if not os.path.exists(CAMINHO_SCALER) or not os.path.exists(CAMINHO_MODELO):
        raise RuntimeError(
            "Artefatos do modelo não encontrados em ./models/. "
            "Rode os notebooks 01 a 04 (em ordem) para gerá-los antes de subir a API."
        )
    scaler = joblib.load(CAMINHO_SCALER)
    modelo = joblib.load(CAMINHO_MODELO)
    return scaler, modelo


def _montar_rotulos_dos_clusters(modelo):
    """
    A numeração dos clusters (0, 1, ...) é arbitrária (depende da inicialização do
    treino) — por isso os nomes/descrições de cada cluster são calculados aqui, a
    partir do perfil real dos dados, em vez de fixados no código (ver notebook `04`).
    Se o dataset clusterizado não estiver disponível, cai para rótulos genéricos.
    """
    n_clusters = modelo.n_clusters
    rotulos = {i: f"Cluster {i}" for i in range(n_clusters)}
    perfil = {}

    if os.path.exists(CAMINHO_DATASET_CLUSTERIZADO):
        df = pd.read_csv(CAMINHO_DATASET_CLUSTERIZADO)
        medianas = df.groupby("cluster")["populacao"].median()
        maior_cluster = medianas.idxmax()

        for c in range(n_clusters):
            tamanho_relativo = "maior porte (grandes centros urbanos e econômicos)" if c == maior_cluster else "menor porte"
            rotulos[c] = f"Municípios de {tamanho_relativo}"
            grupo = df[df["cluster"] == c]
            perfil[c] = ClusterInfo(
                cluster=int(c),
                rotulo=rotulos[c],
                n_municipios=int(len(grupo)),
                populacao_mediana=float(grupo["populacao"].median()),
                pib_total_mediano=float(grupo["pib_total_mil_reais"].median()),
                pib_per_capita_mediano=float(grupo["pib_per_capita_reais"].median()),
            )

    return rotulos, perfil


app = FastAPI(
    title="Clusterização de Municípios Brasileiros",
    description=(
        "Classifica um município brasileiro em um dos clusters socioeconômicos "
        "encontrados pelo modelo KPrototypes treinado em dados abertos do IBGE."
    ),
    version="1.0.0",
)

scaler, modelo = _carregar_artefatos()
ROTULOS_CLUSTER, PERFIL_CLUSTER = _montar_rotulos_dos_clusters(modelo)
IDX_COLUNA_CATEGORICA = len(COLUNAS_NUMERICAS)  # regiao é sempre a última coluna montada


@app.get("/")
def raiz():
    return {
        "projeto": "Clusterização de Municípios Brasileiros",
        "modelo": "KPrototypes",
        "n_clusters": modelo.n_clusters,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/clusters", response_model=list[ClusterInfo])
def listar_clusters():
    """Perfil de cada cluster (tamanho, mediana de população/PIB)."""
    if not PERFIL_CLUSTER:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{CAMINHO_DATASET_CLUSTERIZADO} não encontrado — rode o notebook "
                "04_Avaliacao_Conclusao.ipynb para gerá-lo."
            ),
        )
    return list(PERFIL_CLUSTER.values())


@app.post("/predict", response_model=PredicaoOutput)
def prever(municipio: MunicipioInput):
    """Classifica um município num dos clusters, a partir de seus indicadores."""
    dados = pd.DataFrame([municipio.model_dump()])

    X_num = scaler.transform(np.log1p(dados[COLUNAS_NUMERICAS]))
    X_mix = np.hstack([X_num, dados[["regiao"]].astype(str).values]).astype(object)

    cluster_previsto = int(modelo.predict(X_mix, categorical=[IDX_COLUNA_CATEGORICA])[0])
    rotulo = ROTULOS_CLUSTER[cluster_previsto]

    if cluster_previsto in PERFIL_CLUSTER:
        p = PERFIL_CLUSTER[cluster_previsto]
        descricao = (
            f"Grupo com {p.n_municipios} municípios (base de treino), população mediana de "
            f"{p.populacao_mediana:,.0f} hab. e PIB per capita mediano de R$ {p.pib_per_capita_mediano:,.2f}."
        )
    else:
        descricao = "Perfil detalhado indisponível (dataset clusterizado não encontrado)."

    return PredicaoOutput(cluster=cluster_previsto, rotulo=rotulo, descricao=descricao)
