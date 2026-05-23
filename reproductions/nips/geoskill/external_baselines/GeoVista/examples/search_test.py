from gpt_researcher.search_worker import run_search, _print_results; 
# from utils_search import run_search, _print_results; 

results = run_search("Chille", 
                     retriever_name="tavily", 
                     max_results=10, 
                     expand=True); _print_results(results)
