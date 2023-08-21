@app.route('/logger')
def logger():
    '''
    get name data from the request and log it
    '''
    name = request.args.get('name')