# Hugging Face Models por TheBloke/GPTQ

Este repositório fornece uma implementação do modelo GPTQ por TheBloke, com diferentes versões. O código pode ser modificado para suportar outros modelos pré-treinados em versão GPTQ.

## Modelos Necessários

Antes de executar os scripts, é necessário baixar os modelos conforme as instruções no README2, que é o readme original de um dos modelos. Certifique-se de ter o modelo baixado antes de executar os scripts.

Creditos ao TheBloke pelo modelo de LLM. Parabenizo também pelo empenho em entregar conteúdo à comunidade.

## Scripts Disponíveis

- **chat.py:** Este script possui um prompt fixo e foi projetado para receber uma instrução fixa e analisar um texto subsequente. Ele é utilizado para tentar criar um efeito de NER (no meu caso, uma reverificação textual).
  
- **chat2.py:** Primeira versão de conversação em tempo real com o modelo.

- **chat3.py:** Implementa o uso da biblioteca `torch` para possibilitar o uso da GPU durante a execução.

## Como Usar

1. Clone o repositório:

   ```bash
   git clone https://github.com/Delevati/chat_llm_GPTQ.git

1. Instale as dependencias:

   ```bash
    pip install -r requirements.txt


1. Execute o Script de sua escolha:

   ```bash
    python chat.py
    python chat2.py
    python chat3.py