# Themis Backend

Bem-vindo ao Themis-back, a API Backend para análise e pesquisa de precedentes jurídicos. Este projeto, desenvolvido pela Equipe Skyfall, utiliza Java e Python para oferecer um serviço robusto e escalável. Siga os passos abaixo para configurar e rodar o projeto localmente. 🚀

## Requisitos

- Python 3.10+
- MongoDB Atlas com um índice vetorial configurado
- Chave de API da OpenAI

## Como Executar Localmente

### 1. Clone o repositório

```bash
git clone <url-do-repositório>
cd themis-back
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Preencha os valores no `.env`.

### 4. Inicie o servidor

```bash
python run.py
```

O servidor sobe em `http://localhost:8000`.

## Documentação da API

Com o servidor em execução, acesse a documentação interativa em:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Endpoints

### `POST /petition/analyze`

Recebe um PDF de petição e retorna os precedentes jurídicos mais relevantes, classificados por aplicabilidade.

**Request:** `multipart/form-data` com o campo `file` contendo o PDF.

**Response:**
```json
{
  "results": [
    {
      "id": "string",
      "tipo": "string | null",
      "orgao": "string | null",
      "tese": "string | null",
      "questao": "string | null",
      "textoEmenta": "string | null",
      "textoDecisao": "string | null",
      "relevance_label": "aplicavel | possivelmente aplicavel | nao aplicavel",
      "explanation": "string | null",
      "similarity_score": "number | null"
    }
  ]
}
```

## Estrutura do Projeto

```
themis/
├── app.py              # Inicialização do FastAPI
├── config.py           # Variáveis de ambiente e clientes (MongoDB, OpenAI)
├── routes.py           # Definição das rotas
├── controller.py       # Orquestração do pipeline
├── models/
│   └── responses.py    # Modelos de resposta (Pydantic)
└── services/
    ├── pdf_extractor.py # Extração de texto do PDF
    ├── retrieval.py     # Busca vetorial no Atlas
    └── judge.py         # Classificação dos precedentes via LLM

run.py                  # Ponto de entrada (uvicorn)
requirements.txt
.env.example
```

## Variáveis de Ambiente

| Variável | Descrição |
|---|---|
| `MONGO_URI` | URI de conexão do MongoDB Atlas |
| `OPENAI_API_KEY` | Chave de API da OpenAI |
| `EMBEDDING_MODEL` | Modelo de embedding (ex: `text-embedding-3-large`) |
| `QUERY_MODEL` | Modelo para extração de queries da petição |
| `JUDGE_MODEL` | Modelo para classificação dos precedentes |
| `VECTOR_INDEX` | Nome do índice vetorial no Atlas |
| `DB_NAME` | Nome do banco de dados |
| `COLLECTION_NAME` | Nome da coleção de precedentes |
| `CANDIDATES` | Número de candidatos recuperados antes do judge |
| `VECTOR_SCORE_THRESHOLD` | Score mínimo de similaridade para inclusão |

## Licença

MIT
