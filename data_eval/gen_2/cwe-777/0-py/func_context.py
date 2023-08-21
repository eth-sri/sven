@app.route("/re_search")
def re_search():
    '''
    search regex pattern on target
    '''
    pattern = request.args.get('pattern')
    target = request.args.get('target')