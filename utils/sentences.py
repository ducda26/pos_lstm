class GetSentence(object):
    def __init__(self, data):
        self.n_sentence=1
        self.data=data
        self.empty = False
        function=lambda d:[(w, p) for w, p in zip(d["WORD"].values.tolist(),
                                                        d["POS"].values.tolist())]
        
        self.group_sent = self.data.groupby("ID").apply(function)
        self.all_sentences = [d for d in self.group_sent] 

    def sentences(self, data):
        get=GetSentence(data)
        sentences=get.all_sentences
        return sentences