from search import search_prompt

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar).\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() == "sair":
            break

        try:
            answer = chain.invoke(question)
        except Exception as erro:
            print(f"Erro ao consultar: {erro}\n")
            continue

        print(f"RESPOSTA: {answer}\n")
        print("-" * 40 + "\n")

    print("Encerrado.")

if __name__ == "__main__":
    main()