from rag.retriever import retrieve, load_index
load_index()
docs, score = retrieve('fine for no helmet in Pune', k=5, city='Kharakvasla', state='Maharashtra', country='India')
for i, d in enumerate(docs):
    tier = d.metadata.get('corpus_tier', '?')
    region = d.metadata.get('region', '?')
    section = d.metadata.get('law_section', '')
    source = d.metadata.get('source', '')
    print(f'--- Doc {i+1} | tier={tier} | region={region} | section={section} | source={source} ---')
    print(d.page_content[:250])
    print()
