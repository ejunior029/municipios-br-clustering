"""
Baixa dados públicos do IBGE (Instituto Brasileiro de Geografia e Estatística) sobre os
municípios brasileiros e monta um dataset único para o projeto de clusterização.

Duas interfaces diferentes do IBGE aparecem aqui, e não são a mesma coisa:

  - SIDRA (https://sidra.ibge.gov.br) — o site de NAVEGAÇÃO do Sistema IBGE de
    Recuperação Automática, feito para um humano explorar as tabelas pelo navegador,
    ver quais variáveis existem e descobrir os códigos numéricos delas. É onde foram
    "garimpados" manualmente os IDs usados abaixo (tabela 5938, variável 37, etc.).
    Este script NUNCA acessa sidra.ibge.gov.br diretamente — o site só foi consultado
    uma vez, manualmente, para montar este arquivo. Links de referência (um por
    tabela usada aqui, só para conferir a fonte original):
      https://sidra.ibge.gov.br/tabela/5938  (Produto Interno Bruto dos Municípios)
      https://sidra.ibge.gov.br/tabela/6579  (População residente estimada)

  - API do SIDRA (apisidra.ibge.gov.br) — o mesmo acervo do SIDRA, só que pensado
    para máquina: em vez de uma página para navegar, devolve os números direto em
    JSON quando já se sabe o código da tabela/variável que se quer. É o que este
    script REALMENTE chama, via requests.get(), em baixar_populacao() e baixar_pib().

  - API de Localidades do IBGE (servicodados.ibge.gov.br/api/v1/localidades) — um
    serviço à parte, que NÃO faz parte do SIDRA: é o cadastro de municípios/UFs/
    regiões do IBGE, usado aqui só para os metadados geográficos (nome do município,
    UF, região, mesorregião, microrregião) em baixar_municipios(). Docs:
      https://servicodados.ibge.gov.br/api/docs/localidades

Resumindo o fluxo: o site do SIDRA responde "quais dados existem e com que código"
(pesquisa manual, feita uma vez, fora deste script); a API do SIDRA responde "me dá
os valores desses códigos" (o que as funções abaixo automatizam).

Fontes (dados abertos do governo federal, também catalogados no Portal Brasileiro de
Dados Abertos - https://dados.gov.br):
- Localidades (IBGE): metadados de cada município (UF, região, mesorregião, microrregião)
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

# Localidades: não é SIDRA, é a API de cadastro geográfico do IBGE (ver docstring acima).
URL_MUNICIPIOS = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"

# SIDRA, tabela 6579 (https://sidra.ibge.gov.br/tabela/6579), variável "9324 - População
# residente estimada". n6 = nível territorial "Município" (todos, "all"); p = período/ano.
URL_SIDRA_POPULACAO = f"https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/all/p/{ANO_REFERENCIA}"

# SIDRA, tabela 5938 (https://sidra.ibge.gov.br/tabela/5938) — "Produto Interno Bruto
# dos Municípios". Os números em v/ são os códigos das variáveis dessa tabela (vistos
# navegando o SIDRA), não índices de coluna nem nada inventado por este script.
URL_SIDRA_PIB = (
    f"https://apisidra.ibge.gov.br/values/t/5938/n6/all/v/37,513,517,6575,525/p/{ANO_REFERENCIA}"
)

CAMINHO_SAIDA = "../data/municipios_brasil.csv"

# Nome de cada variável (código SIDRA -> nome de coluna do dataset final), conforme o
# nome oficial da variável na tabela 5938 do SIDRA:
#   37   = "Produto Interno Bruto a preços correntes"
#   513  = "Valor adicionado bruto a preços correntes da agropecuária"
#   517  = "Valor adicionado bruto a preços correntes da indústria"
#   6575 = "Valor adicionado bruto a preços correntes dos serviços, exclusive
#           administração, defesa, educação e saúde públicas e seguridade social"
#   525  = "Valor adicionado bruto a preços correntes da administração, defesa,
#           educação e saúde públicas e seguridade social"
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
    # v/all funciona aqui porque a tabela 6579 só tem uma variável (população
    # residente estimada) — não há o que filtrar, ao contrário da tabela de PIB
    # abaixo, que tem dezenas de variáveis e por isso pede códigos específicos.
    print("Baixando estimativas de população (SIDRA/IBGE)...")
    resp = requests.get(URL_SIDRA_POPULACAO, timeout=120)
    resp.raise_for_status()
    dados = resp.json()[1:]  # primeira linha é o cabeçalho (nomes das colunas, não dado)

    # A API do SIDRA devolve colunas com nomes crípticos e fixos: D1C é o código da
    # 1ª dimensão da tabela (aqui, o código do município) e V é o valor numérico.
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
    dados = resp.json()[1:]  # primeira linha é o cabeçalho (nomes das colunas, não dado)

    # D1C = código do município; D2C = código da variável (37, 513, ...); V = valor.
    # Como pedimos 5 variáveis de uma vez (ver URL_SIDRA_PIB), o retorno vem "empilhado"
    # (uma linha por combinação município x variável) — por isso o pivot_table abaixo.
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
