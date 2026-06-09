# Competição VRPTW — Metaheurísticas

Adaptação dos 5 algoritmos bioinspirados da disciplina (AG, PSO, ACO,
CLONALG e DE) para o **Vehicle Routing Problem with Time Windows (VRPTW)**,
conforme a Tarefa Prática 2.

## Estrutura

```
.
├── src/
│   ├── vrptw_comum.py          # leitura, decodificação, avaliação, Deb,
│   │                           # busca local, verificação e saída (comum aos 5)
│   ├── geneticos/
│   │   ├── ag.py               # AG binário original (função contínua)
│   │   ├── ag_real.py          # AG real original (função contínua)
│   │   ├── ag_tsp.py           # AG de permutação original (base do AG VRPTW)
│   │   └── ag_vrptw.py         # AG adaptado ao VRPTW
│   ├── evolucao_diferencial/
│   │   ├── de_booth.py         # DE original (função contínua)
│   │   ├── de_sphere.py        # DE original (função contínua)
│   │   └── de_vrptw.py         # DE adaptada ao VRPTW (random keys)
│   ├── enxame/
│   │   ├── pso.py              # PSO original (função contínua)
│   │   ├── pso_vrptw.py        # PSO adaptado ao VRPTW (random keys)
│   │   ├── col_for.py          # ACO original (TSP)
│   │   └── col_for_vrptw.py    # ACO adaptado ao VRPTW
│   └── imunologico/
│       ├── clonalg.py          # CLONALG original (TSP)
│       └── clonalg_vrptw.py    # CLONALG adaptado ao VRPTW
├── dados/
│   ├── tsp/                    # instância TSP dos códigos originais
│   ├── vrptw/                  # 5 instâncias de treino do VRPTW
│   └── resultados/             # melhores valores conhecidos + saídas geradas
├── requirements.txt
└── README.md
```

## Pré-requisitos

```bash
pip install -r requirements.txt
```

## Como executar

Cada algoritmo aceita o nome da instância (ou `todas`) e o tempo máximo em
segundos por instância (a competição permite até 480 s):

```bash
python src/geneticos/ag_vrptw.py c101 120
python src/enxame/pso_vrptw.py todas 120
python src/enxame/col_for_vrptw.py rc208 300
python src/evolucao_diferencial/de_vrptw.py r209 120
python src/imunologico/clonalg_vrptw.py c1_2_1 240
```

O resumo completo sai no console e o arquivo de resultado é gravado em
`dados/resultados/<instancia>_resultado_<algoritmo>.txt` no formato exigido
pelo manual da competição (4 casas decimais, distância euclidiana).

## Decisões de projeto (resumo)

- **Representação (igual nos 5 algoritmos):** permutação dos clientes
  ("tour gigante") decodificada em rotas pelo **split ótimo** (programação
  dinâmica de Prins adaptada às janelas de tempo): a melhor partição do
  tour em rotas viáveis — mínimo de veículos e, em empate, mínima
  distância. Em instâncias com mais de 250 clientes a evolução usa um
  decodificador guloso O(N) (mais barato) e o split fica para a solução
  final.
- **Função objetivo:** `veículos * 100000 + distância` — comparação
  lexicográfica seguindo as prioridades da competição (1º frota, 2º
  distância).
- **Tratamento de restrições:** regras de Deb (viável > inviável;
  entre inviáveis, menor violação) em todas as seleções/comparações.
- **PSO e DE:** representação por *random keys* (vetor contínuo, permutação
  via argsort) para manter intactas as equações originais dos algoritmos.
- **Busca local (memética):** subida de encosta estocástica (swap,
  realocação de cliente e de segmentos, inversão 2-opt) em rajadas curtas
  sobre o melhor global durante a evolução e de forma intensiva ao final.
- **Resultados:** ver `dados/resultados/RESULTADOS_TESTES.md` — a DE
  igualou o melhor valor conhecido da instância c101 (10 veículos /
  828.9369). Os arquivos de resultado nunca são sobrescritos por uma
  execução pior: repetir execuções só pode melhorar o que será entregue.
