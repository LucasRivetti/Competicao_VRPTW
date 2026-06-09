# Resultados dos testes — algoritmos adaptados ao VRPTW

Execuções reais em 09/06/2026 (notebook local, Python 3.14, 5 processos em
paralelo). Todas as soluções foram validadas pela verificação independente
(`verificar_solucao`): **todas viáveis** — capacidade, janelas de tempo,
atendimento único e retorno ao depósito respeitados. Logs em `dados/logs/`.

## Evolução das estratégias (3 rodadas de teste)

1. **Rodada 1 (60 s/inst.)** — decodificador guloso (fecha a rota na
   primeira violação): viável sempre, mas fragmenta a frota.
2. **Rodada 2 (120 s/inst.)** — decodificador por **split ótimo** (DP de
   Prins adaptada às janelas: melhor partição do tour em rotas, custo
   lexicográfico veículos→distância) + **polimento memético** (rajadas de
   busca local no melhor global durante a evolução) + busca local final
   com movimentos de segmento (or-opt 2–3).
3. **Rodada 3 (rc2_4_9, 120 s)** — decodificação **híbrida**: o split é
   ~20× mais caro na instância de 400 clientes e derrubava o número de
   gerações; acima de 250 clientes a evolução avalia com o guloso e o
   split fica para a decodificação final (`LIMIAR_SPLIT` em
   `vrptw_comum.py`).

## Resultados finais (veículos / distância)

Rodada 2 para as instâncias até 200 clientes; rodada 3 para rc2_4_9.
Negrito = melhor dos 5 pelo critério da competição (frota, depois distância).

| Instância | Melhor conhecido* | AG | PSO | DE | ACO | CLONALG |
|-----------|------------------|-----|-----|-----|-----|---------|
| c101    | 10 / 828.94  | 11 / 897.09 | 11 / 924.82 | **10 / 828.9369** | 11 / 985.70 | 12 / 965.91 |
| c1_2_1  | 20 / 2704.57 | 28 / 4227.30 | 26 / 3681.61 | **26 / 3505.06** | 27 / 3887.90 | 27 / 3751.34 |
| r209    | 3 / 909.16   | **5 / 1024.37** | 5 / 1101.25 | 7 / 1015.18 | 7 / 998.75 | 6 / 1013.55 |
| rc208   | 3 / 828.14   | 5 / 883.90 | 6 / 907.58 | 6 / 832.39 | **5 / 837.07** | 6 / 900.26 |
| rc2_4_9 | 8 / 4551.11  | 20 / 8227.26 | 21 / 6988.55 | 20 / 7249.46 | 23 / 7835.20 | **20 / 6617.46** |

\* referências fornecidas em `dados/resultados/<inst>_resultado.txt`.

**Destaque: a DE igualou o melhor valor conhecido da c101 — 10 veículos /
828.9369 — com solução viável verificada de forma independente.**

### Comparação rodada 1 (guloso) → final, melhor entre os 5

| Instância | Rodada 1 | Final | Referência |
|-----------|----------|-------|------------|
| c101    | 12 / 886.23  | **10 / 828.94** | 10 / 828.94 |
| c1_2_1  | 28 / 3932.17 | 26 / 3505.06 | 20 / 2704.57 |
| r209    | 6 / 1162.50  | 5 / 1024.37  | 3 / 909.16 |
| rc208   | 6 / 886.47   | 5 / 837.07   | 3 / 828.14 |
| rc2_4_9 | 20 / 9301.12 | 20 / 6617.46 | 8 / 4551.11 |

## Parâmetros usados

- AG: pop=100, elite=6, torneio=3, mutação=0.10, OX, polimento a cada 50 ger.;
- PSO: 60 partículas, c1=c2=2.05, w 0.9→0.4, vel_max=0.25, sementes=50%,
  polimento a cada 25 it.;
- DE: pop=60, F=0.6, BLX α=0.5, sementes=50%, polimento a cada 25 ger.;
- ACO: 20 formigas, α=1.0, β=2.0, ρ=0.10, visibilidade 1/(d·due_date),
  elitista=5, polimento a cada 20 it. (β=3 foi testado: ganho não
  consistente, mantido β=2);
- CLONALG: pop=30, selecionados=10, β=1.0, d=3, polimento a cada 50 ger.;
- Todos: regras de Deb, objetivo lexicográfico, 25% finais do tempo em
  busca local intensiva, decodificação final por split ótimo.

## Observações honestas

- c101 atingiu a referência; rc208/r209 ficaram a 2 veículos dela; as
  instâncias R2/RC2 com pouquíssimos veículos (3) exigem rotas muito longas
  que a representação por tour gigante dificilmente produz sem um
  procedimento dedicado de minimização de frota (ejection chains etc.).
- rc2_4_9 (400 clientes) segue distante da referência (20 vs 8 veículos):
  é a instância mais difícil e o orçamento de teste foi 120 s — na
  competição use os 480 s.
- A variância entre execuções é alta. `escrever_resultado` nunca
  sobrescreve um resultado melhor com um pior, então **repita execuções**:
  o arquivo final guarda sempre a melhor solução já encontrada.

## Sugestões para a competição

- Usar o orçamento integral (480 s) e repetir execuções curtas (2–3×120 s
  costuma render mais que 1×480 s, pela variância).
- DE e AG foram os mais fortes nas instâncias pequenas; CLONALG e PSO nas
  grandes — rodar pelo menos dois algoritmos por instância.
- Ajustar `AUTORES` nos 5 arquivos antes da entrega.
