Configuração: Código executado em ambiente Windows 64 versão 22H2 através da powershell com interpretador Python versão 3.9.13. O programa recorre às bibliotecas numpy, sys, random e math

- O ficheiro "connect4.py" é um script de python que simula o "Connect 4" implementando os algoritmos de pesquisa adversariais Minimax, Minimax com cortes Alpha-Beta e Monte Carlo Tree Search (excecionalmente, incluímos também o algoritmo "Dummy", que faz jogadas aleatórias.)

- A interação com o jogo é feita através do terminal. A cada jogada, são impressos o estado atual do tabuleiro e as instruções para cada jogador. O jogo pode ser executado com 2 jogadores humanos, um jogador humano e o outro um dos algoritmos implementados (neste caso o jogador humano é sempre o jogador 1), ou um algoritmo contra outro.

- Para correr o script, certificar-se que está no diretório correto e que tem instalado na máquina o interpretador de python e quaisquer bibliotecas necessárias e executar na linha de comandos o seguinte comando: python connect4.py <tipo_de_jogo> <profundidade/numero de iteracoes>
Este comando executa o jogo com os parâmetros escolhidos através dos argumentos da linha de comandos

Para selecionar o tipo de jogo, escolher um dos algoritmos (têm que ser escritas exatamente desta forma, este será o argumento 1):
    - Human - 2 jogadores humanos
    - Minimax - jogador 1 é humano e jogador 2 é o algoritmo Minimax sem cortes Alpha Beta
    - MinimaxAlphaBeta - jogador 2 é humano e o jogador 2 é o algoritmo Minimax com cortes Alpha Beta
    - MCTS - jogador 1 é humano e jogador 2 é o algoritmo Monte Carlo Tree Search
    - Dummy - jogador 1 é humano e jogador 2 é o algoritmo Random
De seguida, escolher a profundidade (recomenda-se 3 para o Minimax, 5 para o Minimax com cortes Alpha-Beta e 20000 para o MCTS)

Por exemplo, "python connect4.py Minimax 3" inicializa o jogo em que os movimentos do jogador 2 são decididos pelo algoritmo Minimax com profundidade 3

   