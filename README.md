# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão de um PDF em um banco vetorial (PostgreSQL + pgVector) com busca semântica via linha de comando.
As respostas são geradas a partir do conteúdo do PDF.

Funciona com OpenAI ou Gemini, escolhido pela variável `LLM_PROVIDER`.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Uma chave de API da OpenAI ou do Google

## Configuração

Crie o ambiente virtual e instale as dependências:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copie o `.env.example` e preencha a chave do provider escolhido:

```bash
cp .env.example .env
```

Para OpenAI, basta definir `OPENAI_API_KEY` e manter `LLM_PROVIDER=openai`.
Para Gemini, defina `GOOGLE_API_KEY` e troque para `LLM_PROVIDER=gemini`.

O `DATABASE_URL` já vem apontando para o banco do `docker-compose.yml`, que expõe a porta
5432 no host. Se você tiver um Postgres local ocupando essa porta, troque o mapeamento no
`docker-compose.yml` e o `DATABASE_URL` para uma porta livre. O prefixo
`postgresql+psycopg://` é necessário porque o `langchain-postgres` usa o psycopg 3.

## Execução

Suba o banco:

```bash
docker compose up -d
```

Rode a ingestão do PDF:

```bash
python src/ingest.py
```

Abra o chat:

```bash
python src/chat.py
```

## Exemplo de uso

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Para sair, digite `sair`.

## Como funciona

A ingestão (`src/ingest.py`) lê o PDF com o `PyPDFLoader`, quebra o texto em chunks de
1000 caracteres com overlap de 150 usando o `RecursiveCharacterTextSplitter`, gera os
embeddings e grava tudo no pgVector. A coleção é recriada a cada execução, então rodar
a ingestão mais de uma vez não duplica os dados.

A busca (`src/search.py`) monta uma chain que vetoriza a pergunta, recupera os 10 chunks
mais próximos com `similarity_search_with_score(query, k=10)`, concatena esse conteúdo no
prompt e envia para a LLM. O prompt instrui o modelo a responder apenas com base no
contexto recuperado.

O `src/providers.py` isola a escolha entre OpenAI e Gemini: os demais módulos pedem o
modelo de embeddings, a LLM ou o vector store sem saber qual provider está ativo.

## Observações

Trocar o provider exige rodar a ingestão novamente. Os embeddings da OpenAI têm 1536
dimensões e os do Gemini 768, então uma coleção gerada com um deles não pode ser
consultada com o outro.

Se a conexão com o banco falhar, confira se o container está de pé com `docker compose ps`.
O `docker-compose.yml` já cria a extensão `vector` automaticamente na subida.