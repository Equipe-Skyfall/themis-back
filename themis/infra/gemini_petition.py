import logging

from themis.interfaces.providers import ChatProvider

logger = logging.getLogger(__name__)

_PETICAO_PROMPT = (
    "Você é um advogado brasileiro altamente experiente.\n\n"
    "Com base na descrição do caso e nos precedentes jurídicos fornecidos, "
    "elabore uma **petição inicial** completa e bem fundamentada.\n\n"
    "A petição DEVE conter obrigatoriamente as seguintes seções, nesta ordem:\n\n"
    "## I — FATOS\n"
    "Exposição cronológica e detalhada dos fatos relevantes do caso. "
    "Estruture de forma clara e objetiva, destacando datas, eventos e circunstâncias "
    "que fundamentam a pretensão.\n\n"
    "## II — FUNDAMENTOS JURÍDICOS\n"
    "Análise jurídica detalhada. Para cada fundamento:\n"
    "- Cite a legislação aplicável (artigos, leis, códigos)\n"
    "- Cite os precedentes fornecidos que sustentam a tese, indicando expressamente "
    "tipo, órgão e tese de cada precedente utilizado\n"
    "- Para cada precedente citado, faça a **análise de aderência**: explique por que "
    "a tese do precedente se aplica ao caso concreto, demonstrando a similitude fática e jurídica\n"
    "- Se um precedente tiver distinções relevantes, reconheça-as e explique por que "
    "ainda assim é aplicável ou, alternativamente, por que não o é\n\n"
    "## III — TESE CENTRAL\n"
    "Síntese clara e objetiva da tese jurídica principal que sustenta a pretensão, "
    "reunindo os fundamentos legais e jurisprudenciais apresentados.\n\n"
    "## IV — PEDIDOS\n"
    "Lista clara e específica dos pedidos:\n"
    "- Pedido principal\n"
    "- Pedidos acessórios (tutela de urgência, danos morais/materiais, honorários, custas, etc.)\n"
    "- Requerimentos processuais (citação, produção de provas, etc.)\n\n"
    "REGRAS:\n"
    "- Use linguagem técnica e formal, compatível com petições judiciais brasileiras\n"
    "- Não inclua nomes reais, CPFs, CNPJs ou dados pessoais — use papéis processuais "
    "(Autor, Réu, Requerente, Requerido)\n"
    "- Fundamente com os precedentes fornecidos sempre que aplicável\n"
    "- Se nenhum precedente for relevante, fundamente apenas na legislação\n"
    "- Escreva em português jurídico formal\n\n"
    "DESCRIÇÃO DO CASO:\n{case_description}\n\n"
    "PRECEDENTES ENCONTRADOS:\n{precedentes}\n"
    "{instructions_section}"
)

_REFINE_PROMPT = (
    "Você é um advogado brasileiro altamente experiente.\n\n"
    "O advogado redigiu a petição abaixo e deseja que você a revise "
    "de acordo com as instruções fornecidas.\n\n"
    "REGRAS:\n"
    "- Mantenha a estrutura e conteúdo originais, alterando apenas o que "
    "as instruções pedem\n"
    "- Use linguagem técnica e formal, compatível com petições judiciais brasileiras\n"
    "- Não inclua nomes reais, CPFs, CNPJs ou dados pessoais\n"
    "- Escreva em português jurídico formal\n\n"
    "DESCRIÇÃO DO CASO:\n{case_description}\n\n"
    "PETIÇÃO ATUAL:\n{petition_text}\n\n"
    "INSTRUÇÕES DO ADVOGADO:\n{instructions}\n"
)


def refine_peticao(
    provider: ChatProvider,
    case_description: str,
    petition_text: str,
    instructions: str,
) -> str:
    prompt = _REFINE_PROMPT.format(
        case_description=case_description,
        petition_text=petition_text,
        instructions=instructions,
    )
    return provider.complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )


def generate_peticao(
    provider: ChatProvider,
    case_description: str,
    precedents: list[dict],
    instructions: str | None = None,
) -> str:
    precedents_text = "\n\n".join(
        f"---\nTipo: {p.get('tipo', 'N/A')}\n"
        f"Órgão: {p.get('orgao', 'N/A')}\n"
        f"Tese: {p.get('tese', 'N/A')}\n"
        f"Ementa: {p.get('textoEmenta', 'N/A')}\n"
        f"Relevância: {p.get('relevance_label', 'N/A')}\n"
        f"Explicação: {p.get('explanation', 'N/A')}"
        for p in precedents
    ) if precedents else "Nenhum precedente encontrado."

    instructions_section = (
        f"\nINSTRUÇÕES ADICIONAIS DO ADVOGADO:\n{instructions}\n"
        if instructions else ""
    )

    prompt = _PETICAO_PROMPT.format(
        case_description=case_description,
        precedentes=precedents_text,
        instructions_section=instructions_section,
    )

    return provider.complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
