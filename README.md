# Themis Backend

Bem-vindo ao Themis-back, a API Backend para análise e pesquisa de precedentes jurídicos. Este projeto, desenvolvido pela Equipe Skyfall, utiliza Java e Python para oferecer um serviço robusto e escalável. Siga os passos abaixo para configurar e rodar o projeto localmente. 🚀

## Requisitos

## Como Executar Localmente

### 1. Clone o repositório

```bash
git clone <url-do-repositório>
cd themis-back
```

### 2. Instale as dependências

## Documentação da API

Com o servidor em execução, você pode acessar a documentação interativa da API em:
- **Swagger UI**: `http://localhost:3000/docs`

## Endpoints da API

## Fluxo de Desenvolvimento

## Estrutura do Projeto

```
prisma/
├── migrations/       # Arquivos de migração do banco de dados
└── schema.prisma     # Definição do schema do banco de dados

src/
├── alerts/           # Módulo de alertas
├── config/           # Arquivos de configuração (database, swagger, prisma)
├── controllers/      # Handlers de requisições HTTP
├── factories/        # Factories de entidades
├── middleware/       # Middleware Express (validação, tratamento de erros)
├── repositories/     # Camada de acesso a dados (usando Prisma)
├── routes/           # Definições de rotas
├── services/         # Camada de lógica de negócio
├── types/            # Interfaces e tipos TypeScript
├── container/        # Container de injeção de dependências
└── server.ts         # Ponto de entrada da aplicação

tests/
└── unit/            # Testes unitários
```

## Princípios de Arquitetura

Este projeto segue os princípios de arquitetura limpa:

- **Princípios SOLID**: Responsabilidade única, aberto/fechado, substituição de Liskov, segregação de interface, inversão de dependência
- **Padrão Repository**: Abstrai a lógica de acesso a dados
- **Padrão Factory**: Cria instâncias de entidades
- **Injeção de Dependências**: Gerencia dependências e possibilita testes
- **Camada de Serviço**: Contém a lógica de negócio
- **Camada de Validação**: Garante a integridade dos dados

## Licença

MIT
