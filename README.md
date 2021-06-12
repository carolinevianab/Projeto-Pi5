# Projeto 4 - Pi 5
### Projeto feito por Caroline Viana, Danilo Duarte e Richard Santino para o 5° Semestre de Ciência da Computação

---

## Descrição

Esse projeto é a criação de uma Inteligência Artificial que consiga jogar o jogo de NES Contra. Para isso, fizemos o uso da biblioteca Tensorflow para a IA, que foi treinada usando um [ambiente Gym Retro](https://pypi.org/project/gym-contra/).

## Conteúdo e Execução

O projeto constitui de uma pasta `src`, contendo 4 arquivos `.py` e uma pasta de modelos, e um arquivo `txt` contendo as dependências.

Primeiramente, é necessário fazer a instalação das dependências usadas pelo programa. É possível fazer isso através desse comando:

```shell
pip install -r requirements.txt
```

Após a instalação, é preciso acessar a pasta `src` do projeto (com o comando `cd src`). Por fim, executar o arquivo python usando o seguinte comando:

```shell
python observe.py
```
O programa pode demorar um pouco para iniciar. Isso é devido as alocações de memórias necessárias para que a IA funcione. O tempo levado para isso varia de computador para computador.
