@app.route("/re_search")
def re_search():
    '''
    compile regex pattern and use it to search on target
    '''
    pattern = request.args.get('pattern')
    target = request.args.get('target')