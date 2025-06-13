import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bcb import sgs
from datetime import datetime

# Configuração da página
st.set_page_config(layout="wide")

# Dicionário de códigos SGS (Banco Central)
SERIES_BCB = {
    "PIB ( R$ milhões)": 4380,
    "PIB per capita (R$)": 4381,
    "Inflação (IPCA)": 433,
    "Dívida Pública Bruta (% PIB)": 4505,
    "Taxa Selic": 432,
    "Câmbio (USD/BRL)": 1
}

# Função com cache para carregar dados
@st.cache_data(ttl=3600)
def carregar_dados_bcb(codigo_serie, data_inicio, data_fim):
    return sgs.get({list(SERIES_BCB.keys())[list(SERIES_BCB.values()).index(codigo_serie)]: codigo_serie}, 
                   start=data_inicio, end=data_fim)
@st.cache_data
def calcular_variacao_periodica(dados, periodo='A'):
    """
    Calcula a variação periódica (anual, trimestral ou mensal)
    periodo: 'A' (anual), 'Q' (trimestral), 'M' (mensal)
    """
    if periodo == 'A':
        dados_periodicos = dados.resample('Y').last()
    elif periodo == 'Q':
        dados_periodicos = dados.resample('Q').last()
    else:
        dados_periodicos = dados.resample('M').last()
    
    variacao = dados_periodicos.pct_change() * 100
    return variacao.dropna()

# Interface do usuário
st.title("📈 Indicadores Econômicos do Brasil")
st.write("""
Visualize a evolução temporal dos principais indicadores econômicos do Brasil, 
dados disponibilizados pelo Banco Central através do Sistema Gerenciador de Séries Temporais (SGS).
""")

# Sidebar com controles
with st.sidebar:
    st.header("Filtros")
    
    # Seleção do indicador
    indicador_selecionado = st.selectbox(
        "Selecione o indicador econômico",
        options=list(SERIES_BCB.keys())
    )
    
    # Período de análise
    data_inicio = st.date_input(
        "Data inicial",
        value=datetime(2000, 1, 1),
        min_value=datetime(1980, 1, 1),
        max_value=datetime.today()
    )
    
    data_fim = st.date_input(
        "Data final",
        value=datetime.today(),
        min_value=datetime(1980, 1, 1),
        max_value=datetime.today()
    )
    periodicidade = st.sidebar.radio(
    "Periodicidade da Variação",
    options=['Anual', 'Trimestral', 'Mensal'],
    index=0
    )
    
    # Opções adicionais
    mostrar_media_movel = st.checkbox("Mostrar média móvel (12 meses)", value=True)
    escala_log = st.checkbox("Escala logarítmica", value=False)

# Carrega os dados
codigo_serie = SERIES_BCB[indicador_selecionado]
dados = carregar_dados_bcb(codigo_serie, data_inicio, data_fim)
variacao = calcular_variacao_periodica(dados[indicador_selecionado], 
                                     periodo=periodicidade[0])

# Processamento dos dados
if not dados.empty:
    dados = dados.rename(columns={codigo_serie: indicador_selecionado})
    
    if mostrar_media_movel:
        dados['Média Móvel'] = dados[indicador_selecionado].rolling(window=12).mean()
    
    # Formatação específica para cada série
    if "PIB" in indicador_selecionado:
        dados[indicador_selecionado] = dados[indicador_selecionado]  # Converte para milhões
        unidade = "BRL" if "PIB" in indicador_selecionado else "R$"
    elif "Dívida" in indicador_selecionado:
        unidade = "% do PIB"
    elif "IPCA" in indicador_selecionado:
        unidade = "Variação %"
    elif "Selic" in indicador_selecionado:
        unidade = "% ao ano"
    else:
        unidade = ""

# Estatísticas descritivas
st.subheader("Estatísticas Descritivas")
col1, col2, col3 = st.columns(3)
    
with col1:
        st.metric("Valor Inicial", 
                 f"{dados[indicador_selecionado].iloc[0]:,.2f} {unidade}")
    
with col2:
        st.metric("Valor Atual", 
                 f"{dados[indicador_selecionado].iloc[-1]:,.2f} {unidade}")
    
with col3:
        variacao = ((dados[indicador_selecionado].iloc[-1] / dados[indicador_selecionado].iloc[0] - 1) * 100)
        st.metric("Variação no Período", 
                 f"{variacao:.2f}%",
                 delta_color="inverse" if "Dívida" in indicador_selecionado else "normal")

# Exibição dos dados
if dados.empty:
    st.warning("Não foram encontrados dados para o período selecionado.")
else:
    # Cria duas colunas para os gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de linha original
        if indicador_selecionado == "PIB ( R$ milhões)":
            st.subheader(f"Evolução do PIB ( trilhões R$) ({unidade})\n")
        else:
            st.subheader(f"Evolução do {indicador_selecionado} ({unidade})\n")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        dados[indicador_selecionado].plot(ax=ax1, label=indicador_selecionado)
        
        if mostrar_media_movel:
            dados['Média Móvel'].plot(ax=ax1, linestyle='--', label='Média Móvel (12 meses)')
        
        if escala_log:
            ax1.set_yscale('log')
        
        ax1.set_ylabel(unidade)
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)
    
    with col2:
        # Novo gráfico de barras com variação periódica
        st.subheader(f"Variação Anual do {indicador_selecionado}")
        
        # Calcula a variação anual
        variacao_anual = calcular_variacao_periodica(dados[indicador_selecionado], 'A')
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bars = ax2.bar(
            variacao_anual.index.year,
            variacao_anual,
            color=['#6495ED' if x > 0 else 'red' for x in variacao_anual]
        )
        
        # Adiciona os valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height + (0.3 if height > 0 else -0.8),
                f'{height:.1f}%',
                ha='center',
                va='bottom' if height > 0 else 'top',
                color='black',
                fontsize=8
            )
        
        ax2.set_xlabel("Ano")
        ax2.set_ylabel("Variação %")
        # ax2.grid(axis='y', linestyle='--', alpha=0.7)
        # ax2.axhline(0, color='black', linewidth=0.8)
        st.pyplot(fig2)

# Dados brutos
st.subheader("Dados Brutos")
st.dataframe(dados.style.format({indicador_selecionado: "{:,.2f}"}), 
                height=300)

# Rodapé
st.markdown("---")
st.markdown("""
### Dados obtidos através da API do Banco Central do Brasil (SGS).  
Atualizado em: {}  
Desenvolvido com Streamlit
""".format(datetime.now().strftime("%d/%m/%Y")))