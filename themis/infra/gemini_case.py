import asyncio
import io
import json
import logging
from enum import Enum

from google import genai
from google.genai import types
from pydantic import BaseModel
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

_SEGMENTATION_PROMPT = (
    "Você é um analista jurídico especializado em processos judiciais brasileiros.\n\n"
    "Analise o documento processual completo e execute as seguintes tarefas:\n\n"
    "1. **Segmentação**: Identifique APENAS as peças da AÇÃO PRINCIPAL do processo.\n"
    "   Um processo pode conter várias ações (execuções fiscais, ações declaratórias, ações de ressarcimento, etc.), "
    "mas você deve identificar SOMENTE as peças da ação principal que originou o processo.\n"
    "   Peças a identificar (UMA de cada tipo, no máximo):\n"
    "   - peticao_inicial: a PRIMEIRA petição que inicia a ação principal do processo\n"
    "   - contestacao: resposta do réu à petição inicial da ação principal\n"
    "   - replica: resposta do autor à contestação da ação principal\n"
    "   - sentenca: decisão de mérito de primeira instância da ação principal (NÃO inclua acórdãos, decisões monocráticas, despachos ou decisões de ações incidentais)\n"
    "   - apelacao: recurso de apelação contra a sentença da ação principal\n"
    "   - contrarrazoes: resposta à apelação da ação principal\n\n"
    "   IGNORE completamente: peças de ações incidentais, execuções fiscais, ações de ressarcimento, "
    "ações declaratórias, acórdãos, agravos, despachos, certidões, mandados, "
    "procurações, guias, decisões monocráticas, embargos e peças acessórias.\n"
    "   Para cada peça principal encontrada, informe:\n"
    "   - `type`: um dos tipos acima\n"
    "   - `title`: título descritivo da peça\n"
    "   - `start_page`: número da primeira página (começando em 1)\n"
    "   - `end_page`: número da última página (começando em 1)\n"
    "   - `summary`: resumo conciso de 2-3 frases da peça\n\n"
    "2. **Resumo geral**: Produza um resumo estruturado do caso em no máximo 20 linhas, "
    "cobrindo: tipo de ação, partes (apenas papéis processuais, sem dados pessoais), "
    "fundamento jurídico, pedidos e contexto fático.\n\n"
    "IMPORTANTE:\n"
    "- Use os números de página do PDF (começando em 1)\n"
    "- start_page e end_page devem corresponder EXATAMENTE ao início e fim daquela peça específica, "
    "não ao intervalo até a próxima peça\n"
    "- Uma petição inicial típica tem 5-50 páginas, uma contestação 5-40 páginas, uma sentença 3-30 páginas. "
    "Intervalos de centenas de páginas para uma única peça estão quase certamente errados.\n"
    "- Não inclua nomes, CPFs, CNPJs ou dados pessoais\n"
    "- Se uma peça principal não estiver presente, simplesmente não a inclua\n"
    "- A lista de documents deve ter NO MÁXIMO 6 itens\n\n"
    "Responda SOMENTE com JSON válido no seguinte formato:\n"
    "{\n"
    '  "case_summary": "resumo geral do caso",\n'
    '  "documents": [\n'
    "    {\n"
    '      "type": "peticao_inicial",\n'
    '      "title": "Petição Inicial",\n'
    '      "start_page": 1,\n'
    '      "end_page": 45,\n'
    '      "summary": "resumo da peça"\n'
    "    }\n"
    "  ]\n"
    "}\n"
)


class DocumentType(str, Enum):
    peticao_inicial = "peticao_inicial"
    contestacao = "contestacao"
    replica = "replica"
    sentenca = "sentenca"
    apelacao = "apelacao"
    contrarrazoes = "contrarrazoes"


class DocumentSegmentSchema(BaseModel):
    type: DocumentType
    title: str
    start_page: int
    end_page: int
    summary: str


class CaseAnalysisSchema(BaseModel):
    case_summary: str
    documents: list[DocumentSegmentSchema]


MAX_PAGES_PER_REQUEST = 1000


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        obj, _ = json.JSONDecoder().raw_decode(text.strip())
        return obj
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Last resort: log and re-raise
    logger.error("Failed to parse JSON response (first 500 chars): %s", text[:500])
    raise


_INLINE_MAX_BYTES = 20 * 1024 * 1024  # 20MB


async def _prepare_pdf_part(client: genai.Client, pdf_bytes: bytes) -> types.Part:
    if len(pdf_bytes) <= _INLINE_MAX_BYTES:
        return types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    logger.info("PDF is %d MB, uploading via File API.", len(pdf_bytes) // (1024 * 1024))
    uploaded = client.files.upload(
        file=io.BytesIO(pdf_bytes),
        config=types.UploadFileConfig(mime_type="application/pdf"),
    )
    while uploaded.state.name == "PROCESSING":
        logger.info("File still processing, waiting...")
        await asyncio.sleep(5)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name == "FAILED":
        raise RuntimeError(f"Gemini file processing failed: {uploaded.name}")
    logger.info("File ready: %s", uploaded.name)
    return types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)


_gemini_retry = retry(
    retry=retry_if_exception_type(ServerError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    before_sleep=lambda rs: logger.warning(
        "Gemini 503, retrying (attempt %d)...", rs.attempt_number,
    ),
    reraise=True,
)


@_gemini_retry
async def analyze_pdf(client: genai.Client, model: str, pdf_bytes: bytes) -> dict:
    pdf_part = await _prepare_pdf_part(client, pdf_bytes)

    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=_SEGMENTATION_PROMPT),
                    pdf_part,
                ],
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=CaseAnalysisSchema,
            max_output_tokens=65536,
        ),
    )
    return _parse_json(response.text)


_MERGE_PROMPT = (
    "Você é um analista jurídico sênior. Recebeu os resultados de uma análise processual "
    "que foi dividida em múltiplas partes por limitações técnicas.\n\n"
    "Sua tarefa é consolidar esses resultados parciais em um resultado final coerente:\n\n"
    "1. **Resumo geral (case_summary)**: Produza UM ÚNICO resumo unificado do caso completo, "
    "com no máximo 40 linhas. Cubra: tipo de ação, partes (apenas papéis processuais, sem dados pessoais), "
    "fundamento jurídico, pedidos, contexto fático e desfecho (se houver sentença).\n\n"
    "2. **Documentos (documents)**: Consolide a lista de peças processuais.\n"
    "   - Se a MESMA peça foi cortada na divisão (ex: uma contestação que termina em uma parte "
    "e continua na próxima com o MESMO conteúdo), una-a em um único registro.\n"
    "   - Se existem peças DIFERENTES do mesmo tipo (ex: duas petições iniciais de ações diferentes), "
    "mantenha APENAS a da ação principal e descarte as de ações incidentais.\n"
    "   - NUNCA una peças diferentes em uma só. Cada peça deve ter start_page e end_page "
    "correspondendo EXATAMENTE às páginas daquela peça específica, não ao intervalo entre peças.\n"
    "   - O resultado final deve ter NO MÁXIMO UMA peça de cada tipo (peticao_inicial, contestacao, "
    "replica, sentenca, apelacao, contrarrazoes), totalizando no máximo 6 documentos.\n"
    "   - Se houver múltiplas peças do mesmo tipo de ações diferentes, mantenha APENAS a da ação principal.\n\n"
    "IMPORTANTE:\n"
    "- Mantenha os números de página originais (já ajustados)\n"
    "- Não invente peças que não existem nos dados\n"
    "- Não inclua nomes, CPFs, CNPJs ou dados pessoais\n"
    "- Uma petição inicial típica tem 5-50 páginas, uma sentença 3-30 páginas. "
    "Se uma peça tem centenas de páginas, o intervalo provavelmente está errado.\n\n"
    "Resultados parciais:\n{batch_results}\n\n"
    "Responda SOMENTE com JSON válido no seguinte formato:\n"
    "{{\n"
    '  "case_summary": "resumo unificado do caso (máximo 40 linhas)",\n'
    '  "documents": [\n'
    "    {{\n"
    '      "type": "peticao_inicial",\n'
    '      "title": "Petição Inicial",\n'
    '      "start_page": 1,\n'
    '      "end_page": 45,\n'
    '      "summary": "resumo da peça"\n'
    "    }}\n"
    "  ]\n"
    "}}\n"
)


@_gemini_retry
async def merge_batch_results(client: genai.Client, model: str, batch_results: list[dict]) -> dict:
    formatted = json.dumps(batch_results, ensure_ascii=False, indent=2)
    response = await client.aio.models.generate_content(
        model=model,
        contents=_MERGE_PROMPT.format(batch_results=formatted),
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=CaseAnalysisSchema,
        ),
    )
    return _parse_json(response.text)


_PETITION_SUMMARY_PROMPT = (
    "Você é um analista jurídico sênior especializado em direito brasileiro.\n\n"
    "Analise a petição inicial abaixo e produza um resumo detalhado e estruturado com no máximo 20 linhas, "
    "cobrindo obrigatoriamente:\n\n"
    "1. **Tipo de ação** (ex.: Ação Declaratória, Mandado de Segurança, etc.)\n"
    "2. **Partes** — use apenas papéis processuais (autor, réu, etc.), sem dados pessoais\n"
    "3. **Fatos relevantes** — contexto fático que motivou a ação\n"
    "4. **Fundamento jurídico** — dispositivos legais, princípios e teses invocados\n"
    "5. **Pedidos** — principal e acessórios (tutela de urgência, danos morais, etc.)\n"
    "6. **Causa de pedir** — nexo entre os fatos e o direito invocado\n\n"
    "Regras:\n"
    "- Escreva em parágrafos corridos, sem bullet points\n"
    "- Use linguagem técnica porém acessível\n"
    "- Não inclua nomes, CPFs, CNPJs ou dados pessoais\n"
    "- Não inclua saudações, comentários ou texto além do resumo\n"
)


_MINUTA_PROMPT = (
    "Você é um juiz federal brasileiro altamente experiente.\n\n"
    "Com base nos resumos do processo judicial e nos precedentes vinculantes abaixo, "
    "elabore uma **minuta de sentença** completa e bem fundamentada.\n\n"
    "A minuta DEVE conter obrigatoriamente as três seções a seguir, nesta ordem:\n\n"
    "## I — RELATÓRIO\n"
    "Resumo objetivo e cronológico dos fatos processuais: partes (apenas papéis processuais, "
    "sem dados pessoais), pedidos, argumentos do autor e do réu, provas produzidas, "
    "e incidentes processuais relevantes.\n\n"
    "## II — FUNDAMENTAÇÃO\n"
    "Análise jurídica detalhada de cada questão controvertida. Para cada ponto:\n"
    "- Identifique a tese do autor e a do réu\n"
    "- Aplique a legislação pertinente\n"
    "- Utilize os precedentes fornecidos abaixo para fundamentar sua decisão, "
    "citando-os expressamente (tipo, órgão e tese)\n"
    "- Para cada precedente citado, faça a **análise de aderência ou distinção**:\n"
    "  - Se ADERENTE: explique por que a tese do precedente se aplica ao caso concreto\n"
    "  - Se DISTINGUÍVEL: explique as diferenças fáticas ou jurídicas que afastam sua aplicação\n"
    "- Conclua sobre o acolhimento ou rejeição de cada pedido\n\n"
    "## III — DISPOSITIVO\n"
    "Decisão final, clara e objetiva, indicando:\n"
    "- Procedência total, parcial ou improcedência dos pedidos\n"
    "- Condenações específicas (se houver)\n"
    "- Honorários advocatícios e custas processuais\n"
    "- Demais providências cabíveis\n\n"
    "REGRAS:\n"
    "- Use linguagem técnica e formal, compatível com decisões judiciais brasileiras\n"
    "- Não inclua nomes, CPFs, CNPJs ou dados pessoais — use apenas papéis processuais\n"
    "- Fundamente com os precedentes fornecidos sempre que aplicável\n"
    "- Se um precedente não for relevante para a decisão, não o cite\n"
    "- Escreva em português jurídico formal\n\n"
    "RESUMO GERAL DO CASO:\n{case_summary}\n\n"
    "RESUMO DETALHADO DA PETIÇÃO INICIAL:\n{petition_summary}\n\n"
    "PEÇAS PROCESSUAIS IDENTIFICADAS:\n{documents}\n\n"
    "PRECEDENTES VINCULANTES ENCONTRADOS:\n{precedentes}\n"
)


@_gemini_retry
async def generate_minuta(
    client: genai.Client,
    model: str,
    case_summary: str,
    petition_summary: str,
    documents: list[dict],
    precedents: list[dict],
) -> str:
    precedents_text = "\n\n".join(
        f"---\nTipo: {p.get('tipo', 'N/A')}\n"
        f"Órgão: {p.get('orgao', 'N/A')}\n"
        f"Tese: {p.get('tese', 'N/A')}\n"
        f"Ementa: {p.get('textoEmenta', 'N/A')}\n"
        f"Relevância: {p.get('relevance_label', 'N/A')}\n"
        f"Explicação: {p.get('explanation', 'N/A')}"
        for p in precedents
    )

    documents_text = "\n\n".join(
        f"- {d.get('title', d.get('type', 'N/A'))} (páginas {d.get('start_page')}-{d.get('end_page')})\n"
        f"  Resumo: {d.get('summary', 'N/A')}"
        for d in documents
    )

    prompt = _MINUTA_PROMPT.format(
        case_summary=case_summary,
        petition_summary=petition_summary or "Não disponível",
        documents=documents_text,
        precedentes=precedents_text,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=65536,
        ),
    )
    return response.text


@_gemini_retry
async def summarize_petition_pdf(client: genai.Client, model: str, pdf_bytes: bytes) -> str:
    pdf_part = await _prepare_pdf_part(client, pdf_bytes)
    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=_PETITION_SUMMARY_PROMPT),
                    pdf_part,
                ],
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )
    return response.text
