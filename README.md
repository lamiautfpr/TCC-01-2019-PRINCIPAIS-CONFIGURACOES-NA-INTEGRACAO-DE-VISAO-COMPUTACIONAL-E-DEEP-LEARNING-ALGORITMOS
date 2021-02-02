<h1 align="center">
  <br>
  <a href="https://www.lamia.sh.utfpr.edu.br/">
    <img src="https://user-images.githubusercontent.com/26206052/86039037-3dfa0b80-ba18-11ea-9ab3-7e0696b505af.png" alt="LAMIA - Laboratório de                  Aprendizagem de Máquina e Imagens Aplicados à Indústria" width="400"></a>
<br> <br>
Principais Configurações na Integração de Visão Computacional e Aprendizagem Profunda: Algoritmos e Técnicas
</h1>

<p align="center">  
<b>Grupo</b>: <a href="https://www.lamia.sh.utfpr.edu.br/" target="_blank">LAMIA - Laboratório de Aprendizado de Máquina e Imagens Aplicados à Indústria </a> <br>
<b>Email</b>: <a href="mailto:lamia-sh@utfpr.edu.br" target="_blank">lamia-sh@utfpr.edu.br</a> <br>
<b>Organização</b>: <a href="http://portal.utfpr.edu.br" target="_blank">Universidade Tecnológica Federal do Paraná</a> <a href="http://www.utfpr.edu.br/campus/santahelena" target="_blank"> - Campus Santa Helena</a> <br>
</p>

<p align="center">
<br>
Status do Projeto: Em desenvolvimento :warning:
</p>
___

## Resumo
Os scritps implementam vários modelos do estado da arte da aprendizagem profunda. Dentre eles estão AlexNet,ResNet-34 e MobileNet. Os modelos não se limitam as redes CNNs, assim este repositório contem  implementações de Máquinas de Boltzmann e Autocodificadoras. Além dos modelos implementados, os scripts fornem um protocolo de experimentos que possibilitam a visualiação de métricas, por meio de gráficos e tabelas, e a criação e restauração de estados dos experimentos. Todo o protocolo de experimentos é configurado no arquivo .json. Os modelos foram contruidos em Tensorflow e Keras. Os testes realizados são feitos nas bases CIFAR-10 e MNIST.

Os códigos desenvolvidos fazem parte da monografia: Principais Configurações na Integração de Visão Computacional e Aprendizagem Profunda: Algoritmos e Técnicas. Monografia defendida na UTFPR

## Objetivos
O objetivo geral deste projeto é fornecer modelos de apredizagem profunda construídos em Tensorflow para tarefas de visão computacional, que facilitem o desenvolvimento de aplicações de propósito geral, em especial a aplicaçoes de visão computacional.
Dentre alguns dos objetivos específicos do projeto estão.
  - Implementar um protocolo de experimento para realização de testes com modelos construídos
  - Visualizar principais métricas de avalição por meio de gráficos e tabelas
  - Criar um módulo para salvar os estados do experimentos

## Como Utilizar
Para clonar e rodar está aplicação será necessário o [Git](https://git-scm.com) e o [Python3](https://www.python.org/downloads/) instalados em sua máquina. A partir da linha de comando descrita abaixo será possível clonar este repositório.

```bash
# Clone this repository
$ git clone https://github.com/lamia-utfpr/IC02-2019-Painel-de-Dados-Covid19.git

# Go into the repository
$ cd IC02-2019-Painel-de-Dados-Covid19
```
Note: If you're using Linux Bash for Windows, [see this guide](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/) or use the command prompt from your IDE.

Agora que você já está com o repositório clonado será necessário criar um virtual environment para armazenamento das bibliotecas presentes no requeriments. No diretório do projeto utilize as linhas de comando abaixo:

```bash
# Create virtualenv
$ virtualenv venv

# Execute virtual env
$ source venv/bin/activate
```

Com o virtual enviroment criado e sendo executa será necessário baixar as bibliotecas presentes no requeriments.txt. Para isso basta utilizar o pip3 para fazer a instalação recursiva de todas as bibliotecas presentes no arquivo de texto. Certifique-se que o shell está no diretório do requeriments. Recomenda-se a utilização da execução em super usuário utilizando sudo.

```bash
# Install all requeriments
$ sudo pip3 install -r requeriments.txt
```
Com a criação do ambiente finalizada, configure o arquivo credentials.json com as credenciais de seu banco de dados e voalá! É só rodar o arquivo main.py para inserir todas as bases em seu banco de dados. É importante lembrar que não se utilize nenhum lint do Python na compilação dos algoritmos, pois o mesmo demonstra alguns bugs para importação de packages.

As bibliotecas utilizadas no projeto estão presentes no arquivo requeriments.txt.

```bash
astroid==2.4.2
autopep8==1.5.3
certifi==2020.6.20
chardet==3.0.4
colorama==0.4.3
flake8==3.8.3
idna==2.10
isort==4.3.21
lazy-object-proxy==1.4.3
mccabe==0.6.1
mypy==0.782
mypy-extensions==0.4.3
numpy==1.19.0
pandas==1.0.5
psycopg2==2.8.5
pycodestyle==2.6.0
pyflakes==2.2.0
pylint==2.5.3
python-dateutil==2.8.1
pytz==2020.1
requests==2.24.0
schedule==0.6.0
six==1.15.0
SQLAlchemy==1.3.18
toml==0.10.1
typed-ast==1.4.1
typing-extensions==3.7.4.2
urllib3==1.25.9
wrapt==1.12.1
```
## Tecnologias

Painel Paraná Inteligente Covid-19 usa os seguintes algoritmos e tecnologias:

* [Regressão Logística](https://medium.com/turing-talks/turing-talks-14-modelo-de-predi%C3%A7%C3%A3o-regress%C3%A3o-log%C3%ADstica-7b70a9098e43) - executar predições
* [Regressão Linear](https://medium.com/@lucasoliveiras/regress%C3%A3o-linear-do-zero-com-python-ef74a81c4b84) - executar predições
* [Mínimos Quadrados](https://www.scielo.br/scielo.php?pid=S0100-40422007000200020&script=sci_arttext) - comparar predições geradas e efetuar ajustes
* [Árvores de Decisão](https://www.vooo.pro/insights/um-tutorial-completo-sobre-a-modelagem-baseada-em-tree-arvore-do-zero-em-r-python/) - gerar tomadas de decisão com base em condições pré-estabelecidas
* [Tabula-py](https://tabula-py.readthedocs.io/en/latest/tabula.html/) - tabula-py é um wrapper Python simples de tabula-java, que pode ler a tabelad de arquivos PDF.
* [SQLAlchemy](https://docs.sqlalchemy.org/en/13/) - O SQLAlchemy SQL Toolkit e o Object Relational Mapper são um conjunto abrangente de ferramentas para trabalhar com bancos de dados e Python.
* [Requests](https://requests.readthedocs.io/en/master/) - Utilizada para fazer requisições HTTP pelo Python.
* [Schedule](https://pypi.org/project/schedule/) - O Schedule permite executar funções Python (ou qualquer outra chamada) periodicamente em intervalos predeterminados.
* [Numpy](https://numpy.org/) - plotagens de dados e gráficos
* [Pandas](https://pandas.pydata.org/) - execução de algoritmos de predição
* [Beautiful Soap](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - extração automática de dados

## Citação

Se você utliza e quer citar o projeto em sua pesquisa, por favor utilize o formato de citação abaixo:
    
    @inproceedings{LAMIA_ic02,
      title={Painel Inteligente de Dados Covid-19},
      author={Naves, T. F.; BEUREN, A. T.; BRILHADOR, A.},
      journal={IEEE Conference on Big Data},
      year={2020}
    }
