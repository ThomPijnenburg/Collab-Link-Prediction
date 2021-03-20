def create_pipeline(list_functions):
    def pipeline(*input):
        res = input
        for function in list_functions:
            res = function(*res)
        return res

    return pipeline
