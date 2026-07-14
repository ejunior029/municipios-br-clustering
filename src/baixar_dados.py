"""
Baixa dados públicos do IBGE (Instituto Brasileiro de Geografia e Estatística) sobre os
municípios brasileiros e monta um dataset único para o projeto de clusterização.

Fontes (dados abertos do governo federal, também catalogados no Portal Brasileiro de
Dados Abertos - https://dados.gov.br):
- Localidades (IBGE): metadados de cada município (UF, região, mesorregião, microrregião)
  https://servicodados.ibge.gov.br/api/v1/localidades/municipios
- SIDRA - Estimativas de população (tabela 6579): população residente estimada
- SIDRA - Produto Interno Bruto dos Municípios (tabela 5938): PIB total e composição
  do valor adicionado bruto por setor (agropecuária, indústria, serviços, adm. pública)

Usa-se o ano de 2021 (ANO_REFERENCIA) porque é o último ano em que o IBGE publicou,
simultaneamente, o PIB total e a composição setorial (VAB) por município.

O resultado é salvo em data/municipios_brasil.csv
"""

import time

import pandas as pd
import requests

ANO_REFERENCIA = "2021"

URL_MUNICIPIOS = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
URL_SIDRA_POPULACAO = f"https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/all/p/{ANO_REFERENCIA}"
URL_SIDRA_PIB = (
    f"https://apisidra.ibge.gov.br/values/t/5938/n6/all/v/37,513,517,6575,525/p/{ANO_REFERENCIA}"
)

CAMINHO_SAIDA = "../data/municipios_brasil.csv"

# Nomes das variáveis do PIB (tabela 5938) para renomear as colunas
VARIAVEIS_PIB = {
    "37": "pib_total_mil_reais",
    "513": "vab_agropecuaria_mil_reais",
    "517": "vab_industria_mil_reais",
    "6575": "vab_servicos_mil_reais",
    "525": "vab_adm_publica_mil_reais",
}


def corrigir_encoding(texto):
    """A API do SIDRA retorna alguns textos com encoding latin-1 mal interpretado."""
    if not isinstance(texto, str):
        return texto
    try:
        return texto.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return texto


def baixar_municipios():
    print("Baixando metadados de municípios (IBGE)...")
    resp = requests.get(URL_MUNICIPIOS, timeout=60)
    resp.raise_for_status()
    dados = resp.json()

    registros = []
    for m in dados:
        # Alguns municípios não têm microrregião cadastrada; nesse caso obtém-se a
        # UF/região pela hierarquia de região imediata/intermediária.
        if m["microrregiao"] is not None:
            uf = m["microrregiao"]["mesorregiao"]["UF"]
            mesorregiao = m["microrregiao"]["mesorregiao"]["nome"]
            microrregiao = m["microrregiao"]["nome"]
        else:
            uf = m["regiao-imediata"]["regiao-intermediaria"]["UF"]
            mesorregiao = None
            microrregiao = None
        regiao = uf["regiao"]
        registros.append(
            {
                "codigo_municipio": m["id"],
                "municipio": m["nome"],
                "uf_sigla": uf["sigla"],
                "uf_nome": uf["nome"],
                "regiao": regiao["nome"],
                "mesorregiao": mesorregiao,
                "microrregiao": microrregiao,
            }
        )
    df = pd.DataFrame(registros)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(corrigir_encoding)
    print(f"  {len(df)} municípios encontrados.")
    return df


def baixar_populacao():
    print("Baixando estimativas de população (SIDRA/IBGE)...")
    resp = requests.get(URL_SIDRA_POPULACAO, timeout=120)
    resp.raise_for_status()
    dados = resp.json()[1:]  # primeira linha é o cabeçalho

    df = pd.DataFrame(dados)[["D1C", "V"]].rename(
        columns={"D1C": "codigo_municipio", "V": "populacao"}
    )
    df["codigo_municipio"] = df["codigo_municipio"].astype(int)
    df["populacao"] = pd.to_numeric(df["populacao"], errors="coerce")
    print(f"  {len(df)} registros de população.")
    return df


def baixar_pib():
    print("Baixando PIB e composição setorial dos municípios (SIDRA/IBGE)...")
    resp = requests.get(URL_SIDRA_PIB, timeout=120)
    resp.raise_for_status()
    dados = resp.json()[1:]  # primeira linha é o cabeçalho

    df = pd.DataFrame(dados)[["D1C", "D2C", "V"]].rename(
        columns={"D1C": "codigo_municipio", "D2C": "variavel"}
    )
    df["codigo_municipio"] = df["codigo_municipio"].astype(int)
    df["V"] = pd.to_numeric(df["V"], errors="coerce")
    df["variavel"] = df["variavel"].map(VARIAVEIS_PIB)

    df_pivot = df.pivot_table(
        index="codigo_municipio", columns="variavel", values="V", aggfunc="first"
    ).reset_index()
    print(f"  {len(df_pivot)} municípios com dados de PIB.")
    return df_pivot


def montar_dataset():
    df_municipios = baixar_municipios()
    time.sleep(1)
    df_populacao = baixar_populacao()
    time.sleep(1)
    df_pib = baixar_pib()

    df = df_municipios.merge(df_populacao, on="codigo_municipio", how="left")
    df = df.merge(df_pib, on="codigo_municipio", how="left")

    # Feature derivada: PIB per capita (PIB está em milhares de reais)
    df["pib_per_capita_reais"] = (df["pib_total_mil_reais"] * 1000) / df["populacao"]

    df = df.sort_values("codigo_municipio").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df_final = montar_dataset()
    df_final.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8")
    print(f"\nDataset salvo em {CAMINHO_SAIDA}")
    print(f"Linhas: {len(df_final)} | Colunas: {df_final.shape[1]}")
    print(df_final.head())
