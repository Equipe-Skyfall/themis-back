# Themis Backend

Bem-vindo ao Themis-back, a API Backend para análise e pesquisa de precedentes jurídicos. Este projeto, desenvolvido pela Equipe Skyfall, utiliza Python para oferecer um serviço robusto e escalável. Siga os passos abaixo para configurar e rodar o projeto localmente.

## Requisitos

- Python 3.10+
- MongoDB Atlas com um índice vetorial configurado
- Chave de API da OpenAI
- Chave de API da Groq (opcional)
- Conta no Langfuse (observabilidade)

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

## Autenticação

Todos os endpoints exigem um JWT emitido pelo microserviço de autenticação.

```
Authorization: Bearer <token>
```

O token é validado localmente usando o `JWT_SECRET` compartilhado entre os serviços. Nenhuma chamada ao serviço de autenticação é feita em tempo de execução.

## Documentação da API

Com o servidor em execução, acesse a documentação interativa em:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Endpoints

### `POST /petition/analyze`

Recebe um PDF de petição e retorna os precedentes jurídicos mais relevantes, classificados por aplicabilidade.

**Parâmetros:**
- `file` (multipart/form-data) — PDF da petição

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
      "confidence_score": "number | null"
    }
  ]
}
```

| Campo | Descrição |
|---|---|
| `relevance_label` | Classificação do juiz LLM |
| `explanation` | Justificativa de uma frase |
| `confidence_score` | Confiança do sistema na classificação (0–100): 70% peso do rótulo + 30% peso da similaridade semântica |

### `POST /petition/evaluate`

Executa o pipeline completo e mede a qualidade do resultado comparando com um precedente correto conhecido.

**Parâmetros:**
- `file` (multipart/form-data) — PDF da petição
- `expected_id` (form) — ID do precedente correto

Retorna as métricas de avaliação: `retrieved`, `retrieval_rank`, `pipeline_rank`, `classification`, `hit_at_k`, `reciprocal_rank`. Consulte [EVALUATION.md](./EVALUATION.md) para a explicação detalhada de cada métrica.

## Avaliação do Sistema

O sistema foi avaliado sobre um conjunto de 15–17 petições rotuladas usando Mean Reciprocal Rank (MRR) como métrica principal.

| Métrica | OpenAI | Groq |
|---|---|---|
| MRR | 0.806 | 0.882 |
| Retrieved rate | 100% | 100% |
| Hit@5 | 87% | 94% |
| Classified `aplicavel` | 100% | 97% |

**MRR** mede a posição média do precedente correto nos resultados. Um MRR de 0.882 significa que, em média, o precedente correto aparece próximo à posição 1. Veja a explicação matemática completa em [EVALUATION.md](./EVALUATION.md).

Para rodar a avaliação:
```bash
python evaluate_batch.py --provider openai
python query_results.py   # consulta resultados já salvos no Langfuse
```

## Estrutura do Projeto

```
themis/
├── app.py              # Inicialização do FastAPI
├── auth.py             # Validação do JWT (dependência compartilhada)
├── config.py           # Variáveis de ambiente, clientes e registry de providers
├── controller.py       # Orquestração do pipeline
├── routes.py           # Definição das rotas
├── models/
│   └── responses.py    # Modelos de resposta (Pydantic)
└── services/
    ├── pdf_extractor.py # Extração de texto do PDF
    ├── providers.py     # Strategy pattern para providers LLM (OpenAI, Groq)
    ├── retrieval.py     # Busca vetorial no Atlas
    ├── judge.py         # Classificação dos precedentes via LLM
    └── evaluator.py     # Cálculo de métricas e logging no Langfuse

run.py                  # Ponto de entrada (uvicorn)
evaluate_batch.py       # Avaliação em lote sobre petições rotuladas
query_results.py        # Consulta de resultados agregados no Langfuse
requirements.txt
.env.example
EVALUATION.md           # Explicação detalhada das métricas
```

## Variáveis de Ambiente

| Variável | Descrição |
|---|---|
| `MONGO_URI` | URI de conexão do MongoDB Atlas |
| `OPENAI_API_KEY` | Chave de API da OpenAI |
| `GROQ_API_KEY` | Chave de API da Groq (opcional) |
| `EMBEDDING_MODEL` | Modelo de embedding (ex: `text-embedding-3-large`) |
| `QUERY_MODEL` | Modelo OpenAI para extração de queries da petição |
| `JUDGE_MODEL` | Modelo OpenAI para classificação dos precedentes |
| `GROQ_QUERY_MODEL` | Modelo Groq para extração de queries |
| `GROQ_JUDGE_MODEL` | Modelo Groq para classificação dos precedentes |
| `VECTOR_INDEX` | Nome do índice vetorial no Atlas |
| `DB_NAME` | Nome do banco de dados |
| `COLLECTION_NAME` | Nome da coleção de precedentes |
| `CANDIDATES` | Número de candidatos recuperados antes do judge |
| `VECTOR_SCORE_THRESHOLD` | Score mínimo de similaridade para inclusão |
| `PROVIDER` | Provider LLM ativo: `openai` ou `groq` |
| `JWT_SECRET` | Segredo compartilhado com o microserviço de autenticação |
| `LANGFUSE_PUBLIC_KEY` | Chave pública do Langfuse |
| `LANGFUSE_SECRET_KEY` | Chave secreta do Langfuse |
| `LANGFUSE_HOST` | Host do Langfuse (ex: `https://us.cloud.langfuse.com`) |

## Licença

MIT
