import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bcb import sgs
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(layout="wide")

# Dicionário de códigos SGS (Banco Central)
SERIES_BCB = {
    "PIB (R$ milhões)": 4380,
    "PIB per capita (R$)": 4381,
    "Inflação (IPCA)": 433,
    "Dívida Pública Bruta (% PIB)": 4505,
    "Taxa Selic": 432,
    "Câmbio (USD/BRL)": 1
}

# Limites de anos por indicador (para evitar crashes)
LIMITES_ANOS = {
    "Taxa Selic": 10,
    "Câmbio (USD/BRL)": 10,
    "Inflação (IPCA)": 15,
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

def preparar_dados_previsao(dados, indicador):
    """Prepara os dados para previsão"""
    # Agrupa por ano e pega o último valor
    dados_anuais = dados.resample('Y').last()
    
    # Cria features temporais
    dados_anuais['ano'] = dados_anuais.index.year
    dados_anuais['tempo'] = range(len(dados_anuais))
    
    return dados_anuais

def prever_pib(dados, indicador, modelo_tipo='linear', anos_previsao=10):
    """
    Faz previsão do PIB usando diferentes modelos
    modelo_tipo: 'linear', 'polinomial', 'random_forest'
    """
    # Prepara dados
    dados_prep = preparar_dados_previsao(dados, indicador)
    
    # Features e target
    X = dados_prep[['tempo']].values
    y = dados_prep[indicador].values
    
    # Escolhe o modelo
    if modelo_tipo == 'linear':
        modelo = LinearRegression()
        modelo.fit(X, y)
    elif modelo_tipo == 'polinomial':
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        modelo = LinearRegression()
        modelo.fit(X_poly, y)
    elif modelo_tipo == 'random_forest':
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X, y)
    
    # Faz previsões
    ultimo_tempo = X[-1][0]
    ultimo_ano = dados_prep.index[-1].year
    
    tempos_futuros = np.array([[ultimo_tempo + i + 1] for i in range(anos_previsao)])
    anos_futuros = [ultimo_ano + i + 1 for i in range(anos_previsao)]
    
    if modelo_tipo == 'polinomial':
        tempos_futuros_poly = poly.transform(tempos_futuros)
        previsoes = modelo.predict(tempos_futuros_poly)
    else:
        previsoes = modelo.predict(tempos_futuros)
    
    # Calcula métricas de avaliação no conjunto de treino
    if modelo_tipo == 'polinomial':
        y_pred_treino = modelo.predict(X_poly)
    else:
        y_pred_treino = modelo.predict(X)
    
    mae = mean_absolute_error(y, y_pred_treino)
    rmse = np.sqrt(mean_squared_error(y, y_pred_treino))
    r2 = r2_score(y, y_pred_treino)
    
    # Cria DataFrame com previsões
    df_previsoes = pd.DataFrame({
        'Ano': anos_futuros,
        'Previsão': previsoes
    })
    
    metricas = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    }
    
    return df_previsoes, dados_prep, metricas

# Interface do usuário
with st.container(border=True, height=200):
    st.title("📈 Indicadores Econômicos do Brasil com Previsão de PIB")
    st.write("""
    ##### Visualize a evolução temporal dos principais indicadores econômicos do Brasil e faça previsões do PIB para os próximos 10 anos.
    """)

# Sidebar com controles
with st.sidebar:
    st.header("Filtros")
    
    # Seleção do indicador
    indicador_selecionado = st.selectbox(
        "Selecione o indicador econômico",
        options=list(SERIES_BCB.keys())
    )
    
    # Verifica limite de anos para o indicador selecionado
    limite_anos = LIMITES_ANOS.get(indicador_selecionado, None)
    
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
    
    # Validação do intervalo de datas
    if limite_anos:
        anos_selecionados = (data_fim - data_inicio).days / 365.25
        if anos_selecionados > limite_anos:
            st.error(f"⚠️ Para o indicador '{indicador_selecionado}', o intervalo máximo permitido é de {limite_anos} anos. Por favor, ajuste as datas.")
            st.stop()
    
    periodicidade = st.sidebar.radio(
        "Periodicidade da Variação",
        options=['Anual', 'Trimestral', 'Mensal'],
        index=0
    )
    
    # Opções adicionais
    mostrar_media_movel = st.checkbox("Mostrar média móvel (12 meses)", value=True)
    escala_log = st.checkbox("Escala logarítmica", value=False)
    
    # Seção de previsão (apenas para PIB)
    if "PIB" in indicador_selecionado:
        st.markdown("---")
        st.header("🔮 Previsão de PIB")
        
        modelo_previsao = st.selectbox(
            "Modelo de Previsão",
            options=['linear', 'polinomial', 'random_forest'],
            format_func=lambda x: {
                'linear': 'Regressão Linear',
                'polinomial': 'Regressão Polinomial',
                'random_forest': 'Random Forest'
            }[x]
        )
        
        anos_previsao = st.slider(
            "Anos para prever",
            min_value=1,
            max_value=20,
            value=10
        )
        
        fazer_previsao = st.button("🚀 Gerar Previsão", type="primary")

# Carrega os dados
codigo_serie = SERIES_BCB[indicador_selecionado]
dados = carregar_dados_bcb(codigo_serie, data_inicio, data_fim)

# Processamento dos dados
if not dados.empty:
    dados = dados.rename(columns={codigo_serie: indicador_selecionado})
    
    if mostrar_media_movel:
        dados['Média Móvel'] = dados[indicador_selecionado].rolling(window=12).mean()
    
    # Formatação específica para cada série
    if "PIB (R$ milhões)" == indicador_selecionado:
        dados[indicador_selecionado] = dados[indicador_selecionado] / 1000  # Converte milhões para bilhões
        unidade = "R$ bilhões"
    elif "PIB per capita" in indicador_selecionado:
        unidade = "R$"
    elif "Dívida" in indicador_selecionado:
        unidade = "% do PIB"
    elif "IPCA" in indicador_selecionado:
        unidade = "Variação %"
    elif "Selic" in indicador_selecionado:
        unidade = "% ao ano"
    elif "Câmbio" in indicador_selecionado:
        unidade = "R$"
    else:
        unidade = ""

# Estatísticas descritivas
with st.container(border=True, height=80):
    st.subheader("Estatísticas Descritivas")

col1, col2, col3 = st.columns(3)
    
with col1:
    with st.container(border=True, height=120):    
        st.metric("Valor Inicial", 
                  f"{dados[indicador_selecionado].iloc[0]:,.2f} {unidade}")
    
with col2:
    with st.container(border=True, height=120):    
        st.metric("Valor Atual", 
                  f"{dados[indicador_selecionado].iloc[-1]:,.2f} {unidade}")
    
with col3:
    with st.container(border=True, height=120):
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
        with st.container(border=True, height=450):
            if indicador_selecionado == "PIB (R$ milhões)":
                st.subheader(f"Evolução do PIB (R$ bilhões)")
            else:
                st.subheader(f"Evolução do {indicador_selecionado} ({unidade})")

            fig1, ax1 = plt.subplots(figsize=(8, 4.5))

            dados[indicador_selecionado].plot(ax=ax1, label=indicador_selecionado)
        
            if mostrar_media_movel:
                dados['Média Móvel'].plot(ax=ax1, linestyle='--', label='Média Móvel (12 meses)')
        
            if escala_log:
                ax1.set_yscale('log')
        
            ax1.set_ylabel(unidade)
            ax1.grid(True)
            ax1.legend()
            plt.tight_layout()
            st.pyplot(fig1)

    with col2:
        variacao_periodica = calcular_variacao_periodica(dados[indicador_selecionado], 'A')
        
        with st.container(border=True, height=450):
            st.subheader(f"Variação Anual do {indicador_selecionado}")
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))

            bars = ax2.bar(
                variacao_periodica.index.year,
                variacao_periodica,
                color=['#6495ED' if x > 0 else 'red' for x in variacao_periodica]
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
            ax2.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig2)

# Seção de Previsão (apenas para PIB)
if "PIB" in indicador_selecionado and 'fazer_previsao' in locals() and fazer_previsao:
    st.markdown("---")
    with st.container(border=True):
        st.header("🔮 Previsão do PIB para os Próximos Anos")
        
        with st.spinner("Gerando previsões..."):
            df_previsoes, dados_historicos, metricas = prever_pib(
                dados, 
                indicador_selecionado, 
                modelo_previsao, 
                anos_previsao
            )
        
        # Métricas do modelo
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE (Erro Médio Absoluto)", f"{metricas['MAE']:,.2f}")
        with col2:
            st.metric("RMSE (Raiz do Erro Quadrático)", f"{metricas['RMSE']:,.2f}")
        with col3:
            st.metric("R² (Coeficiente de Determinação)", f"{metricas['R²']:.4f}")
        
        # Gráfico de previsão
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True, height=450):
                st.subheader("Dados Históricos vs Previsão")
                
                fig3, ax3 = plt.subplots(figsize=(8, 4.5))
                
                # Dados históricos
                ax3.plot(dados_historicos.index.year, 
                        dados_historicos[indicador_selecionado], 
                        marker='o', 
                        label='Dados Históricos',
                        linewidth=2,
                        color='#2E86AB')
                
                # Previsões
                ax3.plot(df_previsoes['Ano'], 
                        df_previsoes['Previsão'], 
                        marker='s', 
                        label='Previsão',
                        linewidth=2,
                        linestyle='--',
                        color='#A23B72')
                
                ax3.set_xlabel("Ano")
                ax3.set_ylabel(unidade)
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                plt.tight_layout()
                st.pyplot(fig3)
        
        with col2:
            with st.container(border=True, height=450):
                st.subheader("Taxa de Crescimento Projetada")
                
                # Calcula taxa de crescimento anual
                df_previsoes['Crescimento %'] = df_previsoes['Previsão'].pct_change() * 100
                
                fig4, ax4 = plt.subplots(figsize=(8, 4.5))
                
                bars = ax4.bar(
                    df_previsoes['Ano'][1:],
                    df_previsoes['Crescimento %'][1:],
                    color=['#28a745' if x > 0 else '#dc3545' for x in df_previsoes['Crescimento %'][1:]]
                )
                
                # Adiciona valores nas barras
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + (0.1 if height > 0 else -0.3),
                        f'{height:.1f}%',
                        ha='center',
                        va='bottom' if height > 0 else 'top',
                        fontsize=8
                    )
                
                ax4.set_xlabel("Ano")
                ax4.set_ylabel("Crescimento Anual (%)")
                ax4.axhline(0, color='black', linewidth=0.8)
                ax4.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig4)
        
        # Tabela de previsões
        with st.container(border=True):
            st.subheader("📊 Tabela de Previsões Detalhada")
            
            # Adiciona crescimento à tabela
            df_previsoes_exibicao = df_previsoes.copy()
            df_previsoes_exibicao['Crescimento %'] = df_previsoes_exibicao['Previsão'].pct_change() * 100
            df_previsoes_exibicao = df_previsoes_exibicao.fillna('-')
            
            st.dataframe(
                df_previsoes_exibicao.style.format({
                    'Previsão': '{:,.2f}',
                    'Crescimento %': '{:.2f}%'
                }),
                height=300,
                use_container_width=True
            )
        
        # Insights automáticos
        with st.container(border=True):
            st.subheader("💡 Insights")
            
            previsao_final = df_previsoes['Previsão'].iloc[-1]
            previsao_inicial = df_previsoes['Previsão'].iloc[0]
            crescimento_total = ((previsao_final / previsao_inicial - 1) * 100)
            crescimento_medio = df_previsoes['Crescimento %'][1:].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Projeção de Crescimento:**
                - Crescimento total projetado: **{crescimento_total:.2f}%**
                - Crescimento médio anual: **{crescimento_medio:.2f}%**
                - Valor projetado em {df_previsoes['Ano'].iloc[-1]}: **{previsao_final:,.2f} {unidade}**
                """)
            
            with col2:
                st.warning(f"""
                **⚠️ Observações Importantes:**
                - Estas previsões são baseadas em dados históricos
                - Fatores econômicos não previstos podem alterar as projeções
                - Quanto mais distante a previsão, menor a confiabilidade
                - Use como referência, não como certeza
                """)

# Dados brutos
with st.container(border=True, height=350):
    st.subheader("Dados Brutos")
    st.dataframe(dados.style.format({indicador_selecionado: "{:,.2f}"}), 
                 height=300)

# Rodapé
st.markdown("---")
st.markdown("""
### Dados obtidos através da API do Banco Central do Brasil (SGS).  
**Modelos de Previsão:** Regressão Linear, Regressão Polinomial, Random Forest  
Atualizado em: {}  
Desenvolvido com Streamlit
""".format(datetime.now().strftime("%d/%m/%Y")))
