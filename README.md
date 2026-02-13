# GNN-PCST-R

Utilizando Redes Neurais Gráficas para otimizar os pesos no algoritmo Prize Collecting Steiner Tree usando o protocolo RADNET.


## Dataset 

O dataset utilizado como base para a topologia foi o [CAIDA](https://publicdata.caida.org/) para treinar Redes Neurais de Grafos (GNNs), os mesmos podem ser acessados clicando nesse [link](https://publicdata.caida.org/datasets/as-relationships/serial-2/).

### Processo de Geração

A construção do dataset seguiu três etapas:

1. **Extração de Topologia Estática:** A malha física foi extraída de snapshots reais. Os nós foram classificados em **Servidor Central (Root)**, **Roteadores de Trânsito** e **Antenas de Borda**.

2. **Modelagem de Demanda Dinâmica:** Simulamos a mobilidade de usuários entre as antenas. Cada instância utiliza um conceito de **Look-back (Janela Temporal)**, onde a GNN recebe o histórico de snapshots passados para prever a demanda futura.

3. **Otimização Combinatória (PCST):** Para cada cenário de demanda, a label foi gerada através do problema da **Árvore de Steiner com Prêmios e Custos (PCST)**. Utilizamos um *ensemble* de heurísticas para encontrar a sub-rede mais eficiente, ativando links de menor custo e compartilhando rotas.

### Estrutura dos Arquivos

* **`nodes.csv`**: Todos os nós e seus tipos (servidor, antena ou roteador).
* **`as_graph_[DATA].pt`**: Grafos físicos em formato PyTorch Geometric, contendo as arestas reais e pesos para um treinamento posterior.
* **`instances.jsonl`**: As instâncias geradas, contendo a demanda atual, o histórico temporal e a meta de entrega..
* **`labels.jsonl`**: A "label" para a verificação do treinamento posterior, contendo a lista de nós e arestas que devem ser ativados para atingir a eficiência máxima.


## Ambiente

Todo o desenvolvimento foi realizado utilizando Python 3.12.4.

### Pacotes

Podemos instalar as dependências do projeto usando o comando abaixo:

```bash
$ pip install -r requirements.txt
```

Ou manualmente através dos seguintes comandos:

```bash
pip install ipykernel
pip install nbformat
pip install tqdm
pip install numpy
pip install pcst_fast
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Execução

Para testar o experimento, basta executar o notebook [run.ipynb](https://github.com/charlesluizmendes/GNN-PCST-R/blob/main/run.ipynb).