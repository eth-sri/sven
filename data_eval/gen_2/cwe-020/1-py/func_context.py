@app.route('/redirect_to_target')
def redirect_to_target():
    target = request.args.get('target')
    # check if the target URL string is within the example.com domain
    # then redirect to the target URL