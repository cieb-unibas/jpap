from pipeline.preprocessing import DataRetriever

#### connect to JPOD
retriever = DataRetriever(db_path="C:/Users/matth/Desktop/jpod_test.db")

#### get postings and language information:
postings = retriever.get_postings()
postings_lan = {}
for lan in ["eng", "ger", "fre", "ita"]:
    postings_lan[lan] = postings[postings["text_language"] == lan]
    print("%d postings stored for language '%s'" % (len(postings_lan[lan]), lan))

#### translate postings to 'eng':
    


