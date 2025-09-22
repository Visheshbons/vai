import csv
import os
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

CSV_FILE = "data.csv"

def ensure_csv():
    """Ensure the CSV file exists with correct headers."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "response"])
            writer.writeheader()

def add_entry(prompt, answer):
    """Append a single entry to the CSV file."""
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "response"])
        writer.writerow({"input": prompt, "response": answer})

def run_cli():
    """Run the interactive CLI loop."""
    print(f"{Fore.CYAN}=== Chatbot Dataset Builder ===")
    print(f"{Fore.YELLOW}Type 'exit' at any prompt to quit.\n")

    while True:
        prompt = input(f"{Fore.GREEN}Prompt: {Style.RESET_ALL}").strip()
        if prompt.lower() in ["exit", "quit"]:
            print(f"{Fore.GREEN}Goodbye!")
            break
        answer = input(f"{Fore.BLUE}Answer: {Style.RESET_ALL}").strip()
        if answer.lower() in ["exit", "quit"]:
            print(f"{Fore.GREEN}Goodbye!")
            break
        add_entry(prompt, answer)
        print(f"{Fore.GREEN}✓ Added pair: {Fore.WHITE}[{prompt}] → [{answer}]\n")

if __name__ == "__main__":
    print(f"{Fore.MAGENTA}Initializing dataset file...")
    ensure_csv()
    print(f"{Fore.GREEN}Dataset file ready!\n")
    run_cli()
